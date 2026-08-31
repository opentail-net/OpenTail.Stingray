
namespace OpenTail.Stingray.Diffusion.TextEncoders;

/// <summary>
/// UMT5-XXL encoder-only transformer (`google/umt5-xxl`), the real text encoder Wan 2.1/2.2 uses
/// (`WanPipeline._get_t5_prompt_embeds` in `examples/diffusers/src/diffusers/pipelines/wan/
/// pipeline_wan.py`). Produces context embeddings [seq, 4096] for the Wan DiT's cross-attention.
///
/// Loads weights from `models_t5_umt5-xxl-enc-bf16.pth` (converted to safetensors; Wan's own
/// checkpoint, `Wan-AI/Wan2.1-T2V-1.3B`).
///
/// <para>Same overall shape as <see cref="T5Encoder"/> (T5-XXL, used by FLUX) -- SAME real
/// dimensions (24 layers, d_model=4096, 64 heads, head_dim=64, d_ff=10240), SAME real GELU-gated
/// FFN, SAME real bidirectional relative-position-bucket formula (both confirmed against the real
/// `google/umt5-xxl`/`google/t5-v1_1-xxl` `config.json`s directly: `feed_forward_proj: "gated-
/// gelu"`, `dense_act_fn: "gelu_new"`, `relative_attention_num_buckets: 32`,
/// `relative_attention_max_distance: 128` -- identical on both). The ONE real, load-bearing
/// architectural difference (confirmed against the actual `transformers.models.umt5.modeling_umt5`
/// source, not assumed from the "U" in the name): UMT5's `UMT5LayerSelfAttention.__init__` sets
/// `has_relative_attention_bias=True` for EVERY encoder block (`relative_attention_bias.weight`
/// exists once per layer), whereas plain T5 only has this on layer 0 and shares that single bias
/// across every other layer. Real vocab size is also much larger (256384 vs T5-XXL's 32128,
/// UMT5 is a multilingual model) -- irrelevant to the math itself, just the embedding table size.
/// </para>
/// </summary>
public sealed class UMT5Encoder : IDisposable
{
    private const int Layers      = 24;
    private const int Dim         = 4096;
    private const int Heads       = 64;
    private const int HeadDim     = 64;
    private const int FfDim       = 10240;
    private const int RelPosBuckets = 32;
    private const int MaxRelPos   = 128;

    private readonly SafetensorsLoader _st;

    public UMT5Encoder(string path) => _st = SafetensorsLoader.Open(path);

    /// <summary>Encode token ids -> context embeddings [seq, 4096].</summary>
    public float[] Encode(int[] tokens)
    {
        int seq = tokens.Length;
        var tokEmb = _st.ReadF32("shared.weight");

        var x = new float[seq * Dim];
        for (int t = 0; t < seq; t++)
        {
            int off = tokens[t] * Dim;
            tokEmb.AsSpan(off, Dim).CopyTo(x.AsSpan(t * Dim, Dim));
        }

        for (int i = 0; i < Layers; i++)
            x = EncoderBlock(x, seq, i);

        var fnW = _st.ReadF32("encoder.final_layer_norm.weight");
        DiffusionOps.RmsNorm(x, fnW, Dim);
        return x;
    }

    private float[] EncoderBlock(float[] x, int seq, int blockIdx)
    {
        string p = $"encoder.block.{blockIdx}.layer";

        // Real UMT5: EVERY block has its own relative_attention_bias (unlike plain T5, which
        // only has one on block 0 and reuses it everywhere).
        var rpW = _st.ReadF32($"{p}.0.SelfAttention.relative_attention_bias.weight");
        var relPosBias = ComputeRelPosBias(rpW, seq, Heads);

        var lnW0 = _st.ReadF32($"{p}.0.layer_norm.weight");
        var xNorm = x.ToArray();
        DiffusionOps.RmsNorm(xNorm, lnW0, Dim);
        var attn = SelfAttention(xNorm, relPosBias, seq, $"{p}.0.SelfAttention");
        for (int i = 0; i < x.Length; i++) x[i] += attn[i];

        var lnW1 = _st.ReadF32($"{p}.1.layer_norm.weight");
        var xNorm2 = x.ToArray();
        DiffusionOps.RmsNorm(xNorm2, lnW1, Dim);
        var ff = FeedForward(xNorm2, seq, $"{p}.1.DenseReluDense");
        for (int i = 0; i < x.Length; i++) x[i] += ff[i];

        return x;
    }

    private float[] SelfAttention(float[] x, float[] relBias, int seq, string p)
    {
        var qW = _st.ReadF32($"{p}.q.weight");
        var kW = _st.ReadF32($"{p}.k.weight");
        var vW = _st.ReadF32($"{p}.v.weight");
        var oW = _st.ReadF32($"{p}.o.weight");

        var q = DiffusionOps.Linear(x, qW, null, seq, Dim, Dim);
        var k = DiffusionOps.Linear(x, kW, null, seq, Dim, Dim);
        var v = DiffusionOps.Linear(x, vW, null, seq, Dim, Dim);

        // Real T5-family attention: NO 1/sqrt(headDim) scaling (query/key projections have no
        // bias and the model is trained without softmax scaling; the relative position bias is
        // the only additive term). Confirmed: T5Attention.forward computes scores as a raw
        // matmul, scaling is folded into initialization instead.
        var attnOut = new float[seq * Dim];

        for (int h = 0; h < Heads; h++)
        {
            var scores = new float[seq * seq];
            for (int i = 0; i < seq; i++)
            {
                for (int j = 0; j < seq; j++)
                {
                    float dot = 0f;
                    int qOff = i * Dim + h * HeadDim;
                    int kOff = j * Dim + h * HeadDim;
                    for (int d = 0; d < HeadDim; d++) dot += q[qOff + d] * k[kOff + d];
                    scores[i * seq + j] = dot + relBias[(h * seq + i) * seq + j];
                }
            }
            DiffusionOps.Softmax(scores, seq);

            for (int i = 0; i < seq; i++)
            {
                int outOff = i * Dim + h * HeadDim;
                for (int j = 0; j < seq; j++)
                {
                    float w = scores[i * seq + j];
                    int vOff = j * Dim + h * HeadDim;
                    for (int d = 0; d < HeadDim; d++) attnOut[outOff + d] += w * v[vOff + d];
                }
            }
        }

        return DiffusionOps.Linear(attnOut, oW, null, seq, Dim, Dim);
    }

    private float[] FeedForward(float[] x, int seq, string p)
    {
        // Real gated-gelu FFN (matches T5Encoder's, see this class's doc comment).
        var wi0W = _st.ReadF32($"{p}.wi_0.weight");
        var wi1W = _st.ReadF32($"{p}.wi_1.weight");
        var woW  = _st.ReadF32($"{p}.wo.weight");

        var gate = DiffusionOps.Linear(x, wi0W, null, seq, Dim, FfDim);
        var val  = DiffusionOps.Linear(x, wi1W, null, seq, Dim, FfDim);

        for (int i = 0; i < gate.Length; i++)
            gate[i] = DiffusionOps.Gelu(gate[i]) * val[i];

        return DiffusionOps.Linear(gate, woW, null, seq, FfDim, Dim);
    }

    private static float[] ComputeRelPosBias(float[] biasWeight, int seq, int nHeads)
    {
        var bias = new float[nHeads * seq * seq];
        for (int i = 0; i < seq; i++)
        {
            for (int j = 0; j < seq; j++)
            {
                int bucket = RelPosBucket(j - i);
                for (int h = 0; h < nHeads; h++)
                    bias[(h * seq + i) * seq + j] = biasWeight[bucket * nHeads + h];
            }
        }
        return bias;
    }

    private static int RelPosBucket(int relPos)
    {
        bool negative = relPos < 0;
        int pos = negative ? -relPos : relPos;
        int numBuckets = RelPosBuckets / 2;
        int maxExact = numBuckets / 2;

        int bucket;
        if (pos < maxExact)
        {
            bucket = pos;
        }
        else
        {
            float log = MathF.Log((float)pos / maxExact) / MathF.Log((float)MaxRelPos / maxExact);
            bucket = maxExact + (int)(log * (numBuckets - maxExact));
            bucket = Math.Min(bucket, numBuckets - 1);
        }

        return relPos > 0 ? numBuckets + bucket : bucket;
    }

    public void Dispose() => _st.Dispose();
}
