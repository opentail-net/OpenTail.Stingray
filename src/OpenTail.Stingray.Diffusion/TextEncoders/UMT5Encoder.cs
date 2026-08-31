
namespace OpenTail.Stingray.Diffusion.TextEncoders;

/// <summary>
/// UMT5-XXL encoder-only transformer, the real text encoder Wan 2.1/2.2 uses
/// (`WanPipeline._get_t5_prompt_embeds` in `examples/diffusers/src/diffusers/pipelines/wan/
/// pipeline_wan.py`). Produces context embeddings [seq, 4096] for the Wan DiT's cross-attention.
///
/// Loads weights from `models_t5_umt5-xxl-enc-bf16.pth` (converted to safetensors; Wan's own
/// checkpoint, `Wan-AI/Wan2.1-T2V-1.3B`).
///
/// <para><b>Real tensor names, confirmed directly against the actual downloaded checkpoint (NOT
/// the standard HF `transformers.models.umt5` naming this class originally assumed before
/// inspecting the real file)</b>: Wan ships its own reimplementation/re-export of UMT5, not a
/// literal HF `UMT5EncoderModel` state dict. Real keys: `token_embedding.weight` [256384,4096],
/// `blocks.{i}.norm1`/`norm2.weight`, `blocks.{i}.attn.{q,k,v,o}.weight`,
/// `blocks.{i}.ffn.gate.0.weight` (the GELU-activated branch, `wi_0` in HF's naming),
/// `blocks.{i}.ffn.fc1.weight` (the linear/value branch, `wi_1`), `blocks.{i}.ffn.fc2.weight`
/// (output projection, `wo`), `blocks.{i}.pos_embedding.embedding.weight` [32,64] (real, genuine
/// PER-LAYER relative position bias -- confirmed present on every one of the 24 blocks, unlike
/// plain T5 which only has this on block 0 and shares it everywhere), `norm.weight` (final layer
/// norm). No biases anywhere (T5-family convention, all real Linear layers are bias=False).</para>
///
/// <para>Math itself (dims, GELU-gated FFN, unscaled attention + additive relative-position bias,
/// bidirectional bucket formula) matches <see cref="T5Encoder"/> (T5-XXL, FLUX's encoder) --
/// confirmed identical between the real `google/umt5-xxl` and `google/t5-v1_1-xxl` configs
/// (`feed_forward_proj: "gated-gelu"`, `dense_act_fn: "gelu_new"`,
/// `relative_attention_num_buckets: 32`, `relative_attention_max_distance: 128`, 24 layers,
/// d_model=4096, 64 heads, d_ff=10240). Real vocab is far larger (256384 vs 32128, UMT5 is
/// multilingual) -- irrelevant to the math, just the embedding table size.</para>
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
        var tokEmb = _st.ReadF32("token_embedding.weight");

        var x = new float[seq * Dim];
        for (int t = 0; t < seq; t++)
        {
            int off = tokens[t] * Dim;
            tokEmb.AsSpan(off, Dim).CopyTo(x.AsSpan(t * Dim, Dim));
        }

        for (int i = 0; i < Layers; i++)
            x = EncoderBlock(x, seq, i);

        var fnW = _st.ReadF32("norm.weight");
        DiffusionOps.RmsNorm(x, fnW, Dim);
        return x;
    }

    private float[] EncoderBlock(float[] x, int seq, int blockIdx)
    {
        string p = $"blocks.{blockIdx}";

        // Real Wan UMT5: every block has its own relative position bias.
        var rpW = _st.ReadF32($"{p}.pos_embedding.embedding.weight");
        var relPosBias = ComputeRelPosBias(rpW, seq, Heads);

        var lnW0 = _st.ReadF32($"{p}.norm1.weight");
        var xNorm = x.ToArray();
        DiffusionOps.RmsNorm(xNorm, lnW0, Dim);
        var attn = SelfAttention(xNorm, relPosBias, seq, $"{p}.attn");
        for (int i = 0; i < x.Length; i++) x[i] += attn[i];

        var lnW1 = _st.ReadF32($"{p}.norm2.weight");
        var xNorm2 = x.ToArray();
        DiffusionOps.RmsNorm(xNorm2, lnW1, Dim);
        var ff = FeedForward(xNorm2, seq, $"{p}.ffn");
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

        // Real T5-family attention: raw matmul, NO 1/sqrt(head_dim) scaling (see T5Encoder's
        // matching fix/doc comment -- confirmed against the real T5Attention.forward source).
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
        // Real gated-gelu FFN. Wan's own naming: ffn.gate.0 = the GELU-activated branch (wi_0 in
        // HF naming), ffn.fc1 = the linear/value branch (wi_1), ffn.fc2 = output projection (wo).
        var gateW = _st.ReadF32($"{p}.gate.0.weight");
        var fc1W  = _st.ReadF32($"{p}.fc1.weight");
        var fc2W  = _st.ReadF32($"{p}.fc2.weight");

        var gate = DiffusionOps.Linear(x, gateW, null, seq, Dim, FfDim);
        var val  = DiffusionOps.Linear(x, fc1W, null, seq, Dim, FfDim);

        for (int i = 0; i < gate.Length; i++)
            gate[i] = DiffusionOps.Gelu(gate[i]) * val[i];

        return DiffusionOps.Linear(gate, fc2W, null, seq, FfDim, Dim);
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
