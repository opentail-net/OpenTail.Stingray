
using System.Buffers;
using System.Numerics.Tensors;
using OpenTail.Stingray.Core;

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

    private T5GpuWeights? _gpuWeights;
    private T5GpuWorkspace? _gpuWorkspace;

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
    /// Pre-allocates GPU weights and execution workspace for fast GPU encoding.
    /// </summary>
    private int _gpuValidLen = -1;

    public void InitGpu(IVisionOpsBackend backend, int maxSeqLen = 256) => InitGpu(backend, maxSeqLen, maxSeqLen);

    public void InitGpu(IVisionOpsBackend backend, int maxSeqLen, int validLen)
    {
        if (_gpuWeights is null)
        {
            _gpuWeights = new T5GpuWeights(backend, Wt, Layers, Dim, FfDim);
        }

        if (_gpuWorkspace is null || _gpuWorkspace.SeqLen != maxSeqLen || _gpuValidLen != validLen)
        {
            _gpuWorkspace?.Dispose();
            var rpW = Wt("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight");
            _relPosBias = ComputeRelPosBias(rpW, maxSeqLen, Heads, validLen);
            _gpuWorkspace = new T5GpuWorkspace(backend, maxSeqLen, _relPosBias, Dim, Heads, HeadDim, FfDim);
            _gpuValidLen = validLen;
        }
    }

    /// <summary>
    /// Encodes token ids → context embeddings [seq, 4096] directly on GPU with zero round trips per layer.
    /// </summary>
    public float[] EncodeGpu(int[] tokens, IVisionOpsBackend backend) => EncodeGpu(tokens, backend, tokens.Length);

    /// <param name="validLen">Real (non-padding) token count, including the trailing EOS token.
    /// Real bug found and fixed 2026-09-21 (SD3.5 composition-bug investigation): callers pad
    /// `tokens` to a fixed length (e.g. 256) with the real `&lt;pad&gt;` token id (0, a genuine
    /// embedding row, not a sentinel), and this encoder's self-attention previously let every
    /// padding position fully participate as both query and key with no mask at all -- for a short
    /// prompt (~10 real tokens in a 256-slot buffer), ~96% of the bidirectional self-attention
    /// signal at every layer was mixing in meaningless `&lt;pad&gt;` embeddings, corrupting the real
    /// tokens' own hidden states before they ever reach the DiT. Confirmed against the real
    /// reference (`stable-diffusion.cpp`'s T5 path builds an explicit additive attention mask from
    /// padding and applies it in every self-attention layer). Fixed by masking key positions
    /// `&gt;= validLen` to -inf in the attention scores before softmax, baked directly into the
    /// (per-call, not cached) relative-position bias tensor.</param>
    public float[] EncodeGpu(int[] tokens, IVisionOpsBackend backend, int validLen)
    {
        int seq = tokens.Length;
        InitGpu(backend, seq, validLen);

        var ws = _gpuWorkspace!;
        var weights = _gpuWeights!;

        // 1. Host token embedding lookup & upload to ws.X
        var tokEmb = Wt("shared.weight");
        var xHost = new float[seq * Dim];
        for (int t = 0; t < seq; t++)
        {
            int off = tokens[t] * Dim;
            tokEmb.AsSpan(off, Dim).CopyTo(xHost.AsSpan(t * Dim, Dim));
        }
        using (var xInit = backend.Upload(xHost, TensorShape.D2(seq, Dim), exact: true))
        {
            ((IImageOpsBackend)backend).ScaleInPlace(ws.X, 0f);
            backend.AddInPlace(ws.X, xInit);
        }

        // 2. 24 Transformer Blocks entirely on GPU with chunked batch recording
        var imageOps = backend as IImageOpsBackend;
        const int chunkLayers = 4;
        for (int i = 0; i < Layers; i += chunkLayers)
        {
            imageOps?.BeginBatch();
            int end = Math.Min(i + chunkLayers, Layers);
            for (int l = i; l < end; l++)
            {
                var lw = weights.Layers[l];

                // ── Self-Attention Sub-layer ──
                // Pre-norm: xNorm = RmsNorm(x, ln0, eps=1e-6)
                backend.RmsNormBatched(ws.XNorm, ws.X, lw.LayerNorm0Weight, Dim, seq, eps: 1e-6f);

                // Q, K, V projections via 64x128 tiled SgemmF16
                backend.Sgemm(ws.Q, ws.XNorm, lw.QWeight, seq, Dim, Dim);
                backend.Sgemm(ws.K, ws.XNorm, lw.KWeight, seq, Dim, Dim);
                backend.Sgemm(ws.V, ws.XNorm, lw.VWeight, seq, Dim, Dim);

                // Multi-head attention with relative position bias
                backend.T5MultiHeadAttentionRelBias(ws.AttnOut, ws.Q, ws.K, ws.V, ws.RelPosBias, seq, seq, Heads, HeadDim);

                // O projection: xNorm = Sgemm(AttnOut, O)
                backend.Sgemm(ws.XNorm, ws.AttnOut, lw.OWeight, seq, Dim, Dim);

                // Residual add: X += XNorm
                backend.AddInPlace(ws.X, ws.XNorm);

                // ── Feed-Forward Sub-layer ──
                // Pre-norm: xNorm = RmsNorm(x, ln1, eps=1e-6)
                backend.RmsNormBatched(ws.XNorm, ws.X, lw.LayerNorm1Weight, Dim, seq, eps: 1e-6f);

                // wi_0, wi_1 up-projections: [seq, Dim] -> [seq, FfDim]
                backend.Sgemm(ws.Gate, ws.XNorm, lw.Wi0Weight, seq, Dim, FfDim);
                backend.Sgemm(ws.Val, ws.XNorm, lw.Wi1Weight, seq, Dim, FfDim);

                // Gated GELU: gate = gelu_new(gate) * val
                backend.GeluTanhMul(ws.Gate, ws.Val);

                // Down-projection: FfOut = Sgemm(Gate, wo)
                backend.Sgemm(ws.FfOut, ws.Gate, lw.WoWight, seq, FfDim, Dim);

                // Residual add: X += FfOut
                backend.AddInPlace(ws.X, ws.FfOut);
            }
            imageOps?.EndBatch();
        }

        // 3. Final LayerNorm
        backend.RmsNormBatched(ws.X, ws.X, weights.FinalLayerNormWeight, Dim, seq, eps: 1e-6f);

        // 4. Download result to host
        var output = new float[seq * Dim];
        backend.Download(ws.X, output);
        return output;
    }

    /// <summary>
    /// Encode token ids → context embeddings [seq, 4096].
    /// </summary>
    public float[] Encode(int[] tokens) => Encode(tokens, tokens.Length);

    private int _cpuValidLen = -1;

    /// <param name="validLen">See <see cref="EncodeGpu(int[],IVisionOpsBackend,int)"/>'s doc for why
    /// this must be the real (non-padding) token count, not <c>tokens.Length</c>.</param>
    public float[] Encode(int[] tokens, int validLen)
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

        // Diagnostic dump E0: token embedding output (before any blocks)
        DiagnosticDump("E0", x);

        // Precompute relative position bias (with the real padding mask baked in -- see
        // EncodeGpu's doc comment). Not safely cacheable across calls with different validLen
        // (cond vs. uncond prompts have different real lengths), so recompute whenever either
        // seq or validLen changes.
        if (_relPosBias is null || _relPosBias.Length != seq * seq * Heads || _cpuValidLen != validLen)
        {
            var rpW = Wt("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight");
            _relPosBias = ComputeRelPosBias(rpW, seq, Heads, validLen);
            _cpuValidLen = validLen;
        }

        for (int i = 0; i < Layers; i++)
        {
            x = EncoderBlock(x, _relPosBias, seq, i);
            
            // Diagnostic dumps for block 0 checkpoints
            if (i == 0)
            {
                DiagnosticDump("F0", x); // F0: output after block 0's FFN + residual
            }
        }

        // Final layer norm
        var fnW = Wt("encoder.final_layer_norm.weight");
        DiffusionOps.RmsNorm(x, fnW, Dim);
        
        // Diagnostic dump: final output after all blocks + final RMSNorm
        DiagnosticDump("final", x);
        
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
        
        // Diagnostic dump A0: output after block 0's self-attention + residual (before FFN)
        if (blockIdx == 0)
        {
            DiagnosticDump("A0", x);
        }

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

    /// <summary>Compute T5 relative position bias for a sequence length, with padding keys masked
    /// out (additive -1e9 for any key position j &gt;= <paramref name="validLen"/> -- see
    /// <see cref="EncodeGpu(int[],IVisionOpsBackend,int)"/>'s doc comment for why).</summary>
    private static float[] ComputeRelPosBias(float[] biasWeight, int seq, int nHeads, int validLen)
    {
        // biasWeight: [RelPosBuckets, nHeads] = [32, 64]
        const float MaskBias = -1e9f;
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
                bool isPadKey = j >= validLen;
                for (int h = 0; h < nHeads; h++)
                    bias[(h * seq + i) * seq + j] = isPadKey ? MaskBias : biasWeight[bucket * nHeads + h];
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

    private static int _diagnosticCallCount = 0;
    
    /// <summary>
    /// Diagnostic helper for T5 encoder bisection (2026-09-21). Dumps internal checkpoints to disk
    /// when STINGRAY_T5_DUMP_PREFIX env var is set, for direct comparison against the C++ reference.
    /// Checkpoints: E0 (token embeddings), A0 (after block 0 attn), F0 (after block 0 FFN), final.
    /// </summary>
    private static void DiagnosticDump(string checkpoint, float[] data)
    {
        string? prefix = Environment.GetEnvironmentVariable("STINGRAY_T5_DUMP_PREFIX");
        if (string.IsNullOrEmpty(prefix)) return;
        
        // Append a call counter to distinguish cond vs uncond passes (matches C++ SD_DUMP_CONDITION_PREFIX convention)
        string path = $"{prefix}_{checkpoint}_{_diagnosticCallCount}.bin";
        _diagnosticCallCount++;
        
        var bytes = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
        Console.WriteLine($"[T5 Diagnostic] Dumped {checkpoint}: {data.Length} floats ({data.Length / 4096} tokens x 4096) -> {path}");
    }

    /// <summary>
    /// Frees the GPU-resident weights/workspace AND the host-side FP32 weight cache (~19GB for T5-XXL).
    /// The next Encode/EncodeGpu call reloads them on demand. Call this after encoding when the next
    /// stage needs the memory. On a shared-memory iGPU, keeping T5 resident alongside FLUX.1's ~24GB
    /// of FP16 DiT weights exhausts device memory.
    /// </summary>
    public void ReleaseMemory()
    {
        _gpuWorkspace?.Dispose();
        _gpuWorkspace = null;
        _gpuWeights?.Dispose();
        _gpuWeights = null;
        _gpuValidLen = -1;
        _weightCache.Clear();
    }

    public void Dispose()
    {
        _gpuWorkspace?.Dispose();
        _gpuWeights?.Dispose();
        if (_ownsLoader) _st.Dispose();
    }
}
