
using System.Buffers;
using System.Numerics.Tensors;

namespace OpenTail.Stingray.Diffusion.TextEncoders;

/// <summary>
/// T5-XXL encoder-only transformer.
/// Produces context embeddings [seq, 4096] for the FLUX DiT.
///
/// Loads weights from t5xxl_fp16.safetensors or t5xxl_fp8.safetensors.
/// Architecture: 24 encoder layers, d_model=4096, 64 heads, head_dim=64, d_ff=10240.
/// Notable differences from standard transformers:
///   - T5LayerNorm = RMSNorm without mean-centering or bias
///   - Relative position bias (bias[head, query_pos - key_pos + max_dist])
///   - GELU("gelu_new", tanh-approx) gated FFN, real `google/t5-v1_1-xxl` config
///     (`feed_forward_proj: "gated-gelu"`, `dense_act_fn: "gelu_new"`, confirmed against the
///     real HF config -- NOT SiLU, despite this class's own prior doc comment claiming so; fixed
///     after finding the same activation choice needed re-verifying for Wan's UMT5 encoder, which
///     shares this exact FFN shape): h = gelu_new(w1*x) * (w3*x); out = w2*h
///   - No absolute position embeddings
/// </summary>
public sealed class T5Encoder : IDisposable
{
    private const int Layers      = 24;
    private const int Dim         = 4096;
    private const int Heads       = 64;
    private const int HeadDim     = 64;
    private const int FfDim       = 10240;
    private const int VocabSize   = 32128;
    private const int RelPosBuckets = 32;
    private const int MaxRelPos   = 128;

    private readonly IWeightLoader _st;
    private readonly bool _ownsLoader;
    private float[]? _relPosBias; // lazy cached

    // Perf (2026-09-11): same fix as ClipLEncoder/OpenClipGEncoder -- _st.ReadF32 does a real
    // file.Seek+ReadExactly disk read under a lock on EVERY call, no caching. T5-XXL is a much
    // bigger model (24 layers, 4096-dim) than CLIP, so this is likely even more significant here.
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    private float[] Wt(string name)
    {
        if (_weightCache.TryGetValue(name, out var w)) return w;
        w = _st.ReadF32(name);
        _weightCache[name] = w;
        return w;
    }

    public T5Encoder(string path)
    {
        _st = SafetensorsLoader.Open(path);
        _ownsLoader = true;
    }

    private T5Encoder(IWeightLoader loader, bool ownsLoader)
    {
        _st = loader;
        _ownsLoader = ownsLoader;
    }

    /// <summary>Wraps an already-open loader (e.g. <see cref="SafetensorsLoader.OpenDirectory"/> for
    /// a sharded HF-format checkpoint like `Lightricks/LTX-Video`'s own `text_encoder/` folder) --
    /// caller retains ownership and must dispose it themselves; this instance's own
    /// <see cref="Dispose"/> is then a no-op.</summary>
    public static T5Encoder FromLoader(IWeightLoader loader) => new(loader, ownsLoader: false);

    /// <summary>
    /// Encode token ids → context embeddings [seq, 4096].
    /// </summary>
    public float[] Encode(int[] tokens)
    {
        int seq = tokens.Length;
        var tokEmb = Wt("shared.weight");

        var x = new float[seq * Dim];
        for (int t = 0; t < seq; t++)
        {
            int off = tokens[t] * Dim;
            x.AsSpan().Slice(t * Dim, Dim).Fill(0);
            tokEmb.AsSpan(off, Dim).CopyTo(x.AsSpan(t * Dim, Dim));
        }

        // Precompute relative position bias from first block (shared across all blocks)
        if (_relPosBias is null)
        {
            var rpW = Wt("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight");
            // rpW: [num_buckets, num_heads] = [32, 64]
            _relPosBias = ComputeRelPosBias(rpW, seq, Heads);
        }
        else if (_relPosBias.Length != seq * seq * Heads)
        {
            // Recompute if sequence length changed
            var rpW = Wt("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight");
            _relPosBias = ComputeRelPosBias(rpW, seq, Heads);
        }

        for (int i = 0; i < Layers; i++)
            x = EncoderBlock(x, _relPosBias, seq, i);

        // Final layer norm
        var fnW = Wt("encoder.final_layer_norm.weight");
        DiffusionOps.RmsNorm(x, fnW, Dim);
        return x;  // [seq, 4096]
    }

    private float[] EncoderBlock(float[] x, float[] relPosBias, int seq, int blockIdx)
    {
        string p = $"encoder.block.{blockIdx}.layer";

        // Self-attention sub-layer
        var lnW0 = Wt($"{p}.0.layer_norm.weight");
        var xNorm = (float[])x.Clone();
        DiffusionOps.RmsNorm(xNorm, lnW0, Dim);
        var attn = SelfAttention(xNorm, relPosBias, seq, $"{p}.0.SelfAttention", blockIdx);
        TensorPrimitives.Add(x, attn, x);

        // Feed-forward sub-layer
        var lnW1 = Wt($"{p}.1.layer_norm.weight");
        x.CopyTo(xNorm.AsSpan());
        DiffusionOps.RmsNorm(xNorm, lnW1, Dim);
        var ff = FeedForward(xNorm, seq, $"{p}.1.DenseReluDense");
        TensorPrimitives.Add(x, ff, x);

        return x;
    }

    private float[] SelfAttention(float[] x, float[] relBias, int seq, string p, int blockIdx)
    {
        var qW = Wt($"{p}.q.weight");
        var kW = Wt($"{p}.k.weight");
        var vW = Wt($"{p}.v.weight");
        var oW = Wt($"{p}.o.weight");

        var q = DiffusionOps.Linear(x, qW, null, seq, Dim, Dim);
        var k = DiffusionOps.Linear(x, kW, null, seq, Dim, Dim);
        var v = DiffusionOps.Linear(x, vW, null, seq, Dim, Dim);

        var attnOut = new float[seq * Dim];

        Parallel.For(0, Heads, h =>
        {
            var scoresArr = ArrayPool<float>.Shared.Rent(seq * seq);
            var scores = scoresArr.AsSpan(0, seq * seq);
            try
            {
                for (int i = 0; i < seq; i++)
                {
                    int qOff = i * Dim + h * HeadDim;
                    var qSpan = q.AsSpan(qOff, HeadDim);
                    int relRowOff = (h * seq + i) * seq;

                    for (int j = 0; j < seq; j++)
                    {
                        int kOff = j * Dim + h * HeadDim;
                        float dot = TensorPrimitives.Dot(qSpan, k.AsSpan(kOff, HeadDim));
                        scores[i * seq + j] = dot + relBias[relRowOff + j];
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
                        for (int d = 0; d < HeadDim; d++)
                            attnOut[outOff + d] += w * v[vOff + d];
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(scoresArr);
            }
        });

        return DiffusionOps.Linear(attnOut, oW, null, seq, Dim, Dim);
    }

    private float[] FeedForward(float[] x, int seq, string p)
    {
        // Flan-T5 gated SiLU FFN: h = silu(wi_0 * x) * (wi_1 * x); out = wo * h
        var wi0W = Wt($"{p}.wi_0.weight");  // [FfDim, Dim]
        var wi1W = Wt($"{p}.wi_1.weight");  // [FfDim, Dim]
        var woW  = Wt($"{p}.wo.weight");    // [Dim, FfDim]

        var gate = DiffusionOps.Linear(x, wi0W, null, seq, Dim, FfDim);
        var val  = DiffusionOps.Linear(x, wi1W, null, seq, Dim, FfDim);

        // h = gelu_new(gate) * val -- real T5v1.1/UMT5 "gated-gelu" FFN (verified against the
        // real google/t5-v1_1-xxl and google/umt5-xxl configs: dense_act_fn="gelu_new").
        DiffusionOps.GeluInPlace(gate);
        TensorPrimitives.Multiply(gate, val, gate);

        return DiffusionOps.Linear(gate, woW, null, seq, FfDim, Dim);
    }

    /// <summary>Compute T5 relative position bias for a sequence length.</summary>
    private static float[] ComputeRelPosBias(float[] biasWeight, int seq, int nHeads)
    {
        // biasWeight: [RelPosBuckets, nHeads] = [32, 64]
        var bias = new float[nHeads * seq * seq];
        for (int i = 0; i < seq; i++)
        {
            for (int j = 0; j < seq; j++)
            {
                // Real T5 `compute_bias`: relative_position = memory_position(j) - context_position(i).
                // Was `i - j` (sign-flipped) -- swaps which bucket half ("to the left" vs "to the
                // right" of the query) a given position pair lands in without erroring, a subtle
                // directional bug caught while re-deriving this exact formula for Wan's UMT5 encoder.
                int bucket = RelPosBucket(j - i);
                for (int h = 0; h < nHeads; h++)
                    bias[(h * seq + i) * seq + j] = biasWeight[bucket * nHeads + h];
            }
        }
        return bias;
    }

    private static int RelPosBucket(int relPos)
    {
        // T5 bidirectional relative position bucketing (32 total buckets = 16 positive, 16 negative)
        bool negative = relPos < 0;
        int pos = negative ? -relPos : relPos;
        int numBuckets = RelPosBuckets / 2; // 16 buckets per direction
        int maxExact = numBuckets / 2;     // 8 exact positions (0..7)

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

    public void Dispose()
    {
        if (_ownsLoader) _st.Dispose();
    }
}
