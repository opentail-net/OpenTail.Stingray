using OpenTail.Stingray.Diffusion.Primitives;

namespace OpenTail.Stingray.Diffusion.TextEncoders;

/// <summary>
/// T5Gemma (google/t5gemma-b-b-ul2) encoder-only transformer -- the real text conditioner Stable
/// Audio 3 uses (`T5GemmaConditioner`/`T5GemmaEncoderModel` in the real reference, see
/// docs/057-stable-audio-3-implementation-plan.md). Produces context embeddings [seq, 768].
///
/// Architecturally this is exactly a Gemma 2-family encoder (confirmed against the real
/// `google/t5gemma-b-b-ul2` `config.json`'s `encoder` block, not guessed): 12 layers, hidden 768,
/// 12 heads, head_dim 64 (plain MHA -- `num_key_value_heads` == `num_attention_heads`), RoPE
/// (theta 10000), alternating sliding-window(4096)/full attention per layer (even layers sliding,
/// odd full), attention-logit softcapping (50.0), `query_pre_attn_scalar` (64, which happens to
/// equal `head_dim` for this checkpoint so the attention scale is numerically the same as plain
/// 1/sqrt(head_dim) here), `gelu_pytorch_tanh` gated MLP, and real Gemma-family RMSNorm
/// ((1+weight) scaling, not plain weight scaling -- a deliberate near-identity zero-init design in
/// every real Gemma release) wrapped in a pre+post "sandwich" pattern around both self-attention
/// and the FFN.
///
/// Sliding-window masking is deliberately NOT implemented: the real conditioner always calls this
/// encoder with `max_length=256` (`conditioning.configs[0].config.max_length` in the real Stable
/// Audio 3 `model_config.json`), and 256 &lt;&lt; the real 4096-token sliding window, so every
/// "sliding" layer is mathematically identical to full attention at any sequence length this
/// pipeline will ever actually pass -- not a shortcut that silently diverges from the real model,
/// just the real math simplifying away for this real use case. Revisit if this encoder is ever fed
/// sequences longer than 4096 tokens.
/// </summary>
public sealed class T5GemmaEncoder : IDisposable
{
    private const int Layers = 12;
    private const int Dim = 768;
    private const int Heads = 12;
    private const int HeadDim = 64;
    private const int FfDim = 2048;
    private const float RopeTheta = 10000f;
    private const float AttnLogitSoftcap = 50f;
    private const float QueryPreAttnScalar = 64f;

    private readonly IWeightLoader _st;
    private readonly bool _ownsLoader;

    public T5GemmaEncoder(string path)
    {
        _st = SafetensorsLoader.Open(path);
        _ownsLoader = true;
    }

    private T5GemmaEncoder(IWeightLoader loader, bool ownsLoader)
    {
        _st = loader;
        _ownsLoader = ownsLoader;
    }

    /// <summary>Wraps an already-open loader -- caller retains ownership.</summary>
    public static T5GemmaEncoder FromLoader(IWeightLoader loader) => new(loader, ownsLoader: false);

    /// <summary>
    /// Encode token ids → context embeddings [seq, 768]. <paramref name="attentionMask"/> (true =
    /// real token, false = padding) defaults to all-true when omitted.
    /// </summary>
    public float[] Encode(int[] tokens, bool[]? attentionMask = null)
    {
        int seq = tokens.Length;
        attentionMask ??= CreateAllTrue(seq);

        var tokEmb = _st.ReadF32("model.encoder.embed_tokens.weight");
        var x = new float[seq * Dim];
        for (int t = 0; t < seq; t++)
        {
            tokEmb.AsSpan(tokens[t] * Dim, Dim).CopyTo(x.AsSpan(t * Dim, Dim));
        }

        // Real Gemma family: embeddings are scaled by sqrt(hidden_size) immediately after lookup.
        float normalizer = MathF.Sqrt(Dim);
        for (int i = 0; i < x.Length; i++) x[i] *= normalizer;

        var (cos, sin) = BuildRope(seq);

        for (int layer = 0; layer < Layers; layer++)
        {
            x = EncoderLayer(x, seq, layer, cos, sin, attentionMask);
        }

        var finalNormW = _st.ReadF32("model.encoder.norm.weight");
        GemmaRmsNorm(x, finalNormW, Dim);
        return x;
    }

    private static bool[] CreateAllTrue(int n)
    {
        var m = new bool[n];
        Array.Fill(m, true);
        return m;
    }

    private float[] EncoderLayer(float[] x, int seq, int layerIdx, float[] cos, float[] sin, bool[] attentionMask)
    {
        string p = $"model.encoder.layers.{layerIdx}";

        var preAttnW = _st.ReadF32($"{p}.pre_self_attn_layernorm.weight");
        var xNorm = x.ToArray();
        GemmaRmsNorm(xNorm, preAttnW, Dim);

        var attn = SelfAttention(xNorm, seq, p, cos, sin, attentionMask);

        var postAttnW = _st.ReadF32($"{p}.post_self_attn_layernorm.weight");
        GemmaRmsNorm(attn, postAttnW, Dim);

        for (int i = 0; i < x.Length; i++) x[i] += attn[i];

        var preFfW = _st.ReadF32($"{p}.pre_feedforward_layernorm.weight");
        var xNorm2 = x.ToArray();
        GemmaRmsNorm(xNorm2, preFfW, Dim);

        var ff = FeedForward(xNorm2, seq, p);

        var postFfW = _st.ReadF32($"{p}.post_feedforward_layernorm.weight");
        GemmaRmsNorm(ff, postFfW, Dim);

        for (int i = 0; i < x.Length; i++) x[i] += ff[i];

        return x;
    }

    private float[] SelfAttention(float[] x, int seq, string p, float[] cos, float[] sin, bool[] attentionMask)
    {
        var qW = _st.ReadF32($"{p}.self_attn.q_proj.weight");
        var kW = _st.ReadF32($"{p}.self_attn.k_proj.weight");
        var vW = _st.ReadF32($"{p}.self_attn.v_proj.weight");
        var oW = _st.ReadF32($"{p}.self_attn.o_proj.weight");

        var q = DiffusionOps.Linear(x, qW, null, seq, Dim, Dim);
        var k = DiffusionOps.Linear(x, kW, null, seq, Dim, Dim);
        var v = DiffusionOps.Linear(x, vW, null, seq, Dim, Dim);

        SplitHalfRoPE.ApplyRoPE(q, cos, sin, seq, Heads, HeadDim);
        SplitHalfRoPE.ApplyRoPE(k, cos, sin, seq, Heads, HeadDim);

        float scale = 1f / MathF.Sqrt(QueryPreAttnScalar);
        var attnOut = new float[seq * Dim];

        for (int h = 0; h < Heads; h++)
        {
            var scores = new float[seq * seq];
            for (int i = 0; i < seq; i++)
            {
                int qOff = i * Dim + h * HeadDim;
                for (int j = 0; j < seq; j++)
                {
                    int kOff = j * Dim + h * HeadDim;
                    float dot = 0f;
                    for (int d = 0; d < HeadDim; d++) dot += q[qOff + d] * k[kOff + d];
                    dot *= scale;

                    // Real Gemma2-family attention-logit softcapping: softcap * tanh(scores / softcap).
                    dot = AttnLogitSoftcap * MathF.Tanh(dot / AttnLogitSoftcap);

                    if (!attentionMask[j]) dot = -1e9f;

                    scores[i * seq + j] = dot;
                }
            }
            DiffusionOps.Softmax(scores, seq);

            for (int i = 0; i < seq; i++)
            {
                int outOff = i * Dim + h * HeadDim;
                for (int j = 0; j < seq; j++)
                {
                    float w = scores[i * seq + j];
                    if (w == 0f) continue;
                    int vOff = j * Dim + h * HeadDim;
                    for (int d = 0; d < HeadDim; d++) attnOut[outOff + d] += w * v[vOff + d];
                }
            }
        }

        return DiffusionOps.Linear(attnOut, oW, null, seq, Dim, Dim);
    }

    private float[] FeedForward(float[] x, int seq, string p)
    {
        var gateW = _st.ReadF32($"{p}.mlp.gate_proj.weight");
        var upW = _st.ReadF32($"{p}.mlp.up_proj.weight");
        var downW = _st.ReadF32($"{p}.mlp.down_proj.weight");

        var gate = DiffusionOps.Linear(x, gateW, null, seq, Dim, FfDim);
        var up = DiffusionOps.Linear(x, upW, null, seq, Dim, FfDim);

        // Real Gemma2 MLP: down_proj(gelu_pytorch_tanh(gate_proj(x)) * up_proj(x)).
        for (int i = 0; i < gate.Length; i++)
            gate[i] = DiffusionOps.Gelu(gate[i]) * up[i];

        return DiffusionOps.Linear(gate, downW, null, seq, FfDim, Dim);
    }

    private static (float[] cos, float[] sin) BuildRope(int seq)
    {
        var cos = new float[seq * HeadDim];
        var sin = new float[seq * HeadDim];
        for (int s = 0; s < seq; s++)
        {
            SplitHalfRoPE.FillFrequencies(cos, sin, s * HeadDim, s, HeadDim, RopeTheta);
        }
        return (cos, sin);
    }

    /// <summary>
    /// Real Gemma-family RMSNorm: y = (x / rms(x)) * (1 + weight) -- NOT plain `x * weight` like
    /// T5/UMT5's RMSNorm (<see cref="DiffusionOps.RmsNorm"/>). Every real Gemma release
    /// (1/2/3, and T5Gemma per its shared `t5_gemma_module` architecture) zero-initializes this
    /// weight so the layer starts as a near-identity map; using plain `x * weight` here would
    /// silently zero out every activation on a freshly-initialized layer instead of passing it
    /// through, and produces the wrong (near-zero) output on this real checkpoint's actual
    /// trained weights too, since they're centered near 0, not 1.
    /// </summary>
    private static void GemmaRmsNorm(Span<float> x, ReadOnlySpan<float> weight, int dim, float eps = 1e-6f)
    {
        int n = x.Length / dim;
        for (int row = 0; row < n; row++)
        {
            var r = x.Slice(row * dim, dim);
            float ss = 0f;
            for (int i = 0; i < dim; i++) ss += r[i] * r[i];
            float invRms = 1f / MathF.Sqrt(ss / dim + eps);
            for (int i = 0; i < dim; i++) r[i] = r[i] * invRms * (1f + weight[i]);
        }
    }

    public void Dispose()
    {
        if (_ownsLoader) _st.Dispose();
    }
}
