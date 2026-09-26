using OpenTail.Stingray.Core;

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
    private const int Layers        = 24;
    private const int Dim           = 4096;
    private const int Heads         = 64;
    private const int HeadDim       = 64;
    private const int FfDim         = 10240;
    private const int RelPosBuckets = 32;
    private const int MaxRelPos     = 128;

    private readonly IWeightLoader _st;
    private readonly bool _ownsLoader;

    private UMT5GpuWeights? _gpuWeights;
    private UMT5GpuWorkspace? _gpuWorkspace;
    private float[][]? _relPosBiases;

    // Perf (2026-09-11): same fix as ClipLEncoder/OpenClipGEncoder/T5Encoder -- _st.ReadF32 does a
    // real file.Seek+ReadExactly disk read under a lock on EVERY call, no caching.
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    /// <summary>
    /// Token-embedding lookup that converts only the looked-up rows. token_embedding.weight is
    /// [256384, 4096] — ~2 GB BF16 on disk, ~4 GB as F32 — and every encode path used to read and
    /// convert the whole table to fetch a few dozen rows (twice per Wan generation on the GPU path,
    /// and the CPU path then kept the 4 GB copy cached for the encoder's lifetime). Reads rows
    /// straight from the memory-mapped safetensors file when the loader exposes it; otherwise
    /// falls back to the full read.
    /// </summary>
    private unsafe float[] LookupTokenEmbeddings(int[] tokens)
    {
        var x = new float[tokens.Length * Dim];
        if (_st is SafetensorsLoader sl
            && sl.TryGetMappedPointer("token_embedding.weight", out byte* p, out long bytes, out string dt)
            && dt is "BF16" or "F16" or "F32")
        {
            int elem = dt == "F32" ? 4 : 2;
            for (int t = 0; t < tokens.Length; t++)
            {
                long off = (long)tokens[t] * Dim * elem;
                if (off < 0 || off + (long)Dim * elem > bytes)
                    throw new ArgumentOutOfRangeException(nameof(tokens), $"token id {tokens[t]} outside the embedding table");
                byte* row = p + off;
                var dst = x.AsSpan(t * Dim, Dim);
                if (dt == "F32")
                    new ReadOnlySpan<float>(row, Dim).CopyTo(dst);
                else if (dt == "BF16")
                    for (int d = 0; d < Dim; d++) dst[d] = BitConverter.Int32BitsToSingle(((ushort*)row)[d] << 16);
                else
                    for (int d = 0; d < Dim; d++) dst[d] = (float)((Half*)row)[d];
            }
            return x;
        }

        var tokEmb = _st.ReadF32("token_embedding.weight");
        for (int t = 0; t < tokens.Length; t++)
            tokEmb.AsSpan(tokens[t] * Dim, Dim).CopyTo(x.AsSpan(t * Dim, Dim));
        return x;
    }

    private float[] Wt(string name)
    {
        if (_weightCache.TryGetValue(name, out var w)) return w;
        w = _st.ReadF32(name);
        _weightCache[name] = w;
        return w;
    }

    public UMT5Encoder(string path)
    {
        _st = SafetensorsLoader.Open(path);
        _ownsLoader = true;
    }

    public UMT5Encoder(IWeightLoader loader, bool ownsLoader = false)
    {
        _st = loader;
        _ownsLoader = ownsLoader;
    }

    public static UMT5Encoder FromLoader(IWeightLoader loader) => new(loader, ownsLoader: false);

    /// <summary>
    /// Pre-allocates GPU weights and execution workspace for fast GPU encoding.
    /// </summary>
    public void InitGpu(IVisionOpsBackend backend, int maxSeqLen = 256)
    {
        if (_gpuWeights is null)
        {
            _gpuWeights = new UMT5GpuWeights(backend, _st.ReadF32, Layers, Dim, FfDim);
        }

        if (_gpuWorkspace is null || _gpuWorkspace.SeqLen != maxSeqLen)
        {
            _gpuWorkspace?.Dispose();
            _relPosBiases = new float[Layers][];
            for (int i = 0; i < Layers; i++)
            {
                var rpW = Wt($"blocks.{i}.pos_embedding.embedding.weight");
                _relPosBiases[i] = ComputeRelPosBias(rpW, maxSeqLen, Heads);
            }
            _gpuWorkspace = new UMT5GpuWorkspace(backend, maxSeqLen, _relPosBiases, Dim, Heads, HeadDim, FfDim);
        }
    }

    /// <summary>
    /// Encodes token ids -> context embeddings [seq, 4096] directly on GPU with layer streaming (VRAM-safe).
    /// </summary>
    public float[] EncodeGpu(int[] tokens, IVisionOpsBackend backend)
    {
        int seq = tokens.Length;

        _relPosBiases = new float[Layers][];
        for (int i = 0; i < Layers; i++)
        {
            var rpW = Wt($"blocks.{i}.pos_embedding.embedding.weight");
            _relPosBiases[i] = ComputeRelPosBias(rpW, seq, Heads);
        }

        using var ws = new UMT5GpuWorkspace(backend, seq, _relPosBiases, Dim, Heads, HeadDim, FfDim);

        // 1. Host token embedding lookup (only the needed rows) & upload to ws.X.
        var xHost = LookupTokenEmbeddings(tokens);

        using (var xInit = backend.Upload(xHost, TensorShape.D2(seq, Dim), exact: true))
        {
            ((IImageOpsBackend)backend).ScaleInPlace(ws.X, 0f);
            backend.AddInPlace(ws.X, xInit);
        }

        // 2. 24 Transformer Blocks with layer-by-layer weight streaming / batching.
        bool hasGpuWeights = _gpuWeights != null;
        var imageOps = backend as IImageOpsBackend;
        if (hasGpuWeights)
        {
            const int chunkLayers = 4;
            for (int i = 0; i < Layers; i += chunkLayers)
            {
                imageOps?.BeginBatch();
                int end = Math.Min(i + chunkLayers, Layers);
                for (int l = i; l < end; l++)
                {
                    ExecuteLayerGpu(ws, _gpuWeights!.Layers[l], l, seq, backend);
                }
                imageOps?.EndBatch();
            }
        }
        else
        {
            for (int i = 0; i < Layers; i++)
            {
                var lw = new UMT5GpuWeights.UMT5LayerGpuWeights(backend, _st.ReadF32, i, Dim, FfDim);
                try
                {
                    imageOps?.BeginBatch();
                    ExecuteLayerGpu(ws, lw, i, seq, backend);
                    imageOps?.EndBatch();
                }
                finally
                {
                    lw.Dispose();
                }
            }
        }

        // 3. Final RMSNorm
        using var finalNorm = backend.Upload(Wt("norm.weight"), TensorShape.D1(Dim), exact: true);
        backend.RmsNormBatched(ws.X, ws.X, finalNorm, Dim, seq, eps: 1e-6f);

        // 4. Download result
        var result = new float[seq * Dim];
        backend.Download(ws.X, result);
        return result;
    }

    /// <summary>
    /// Encodes both conditional and unconditional token sequences in a single pass over the 24 layers,
    /// halving disk reads and GC pauses.
    /// </summary>
    public (float[] Cond, float[] Uncond) EncodePairGpu(int[] condTokens, int[] uncondTokens, IVisionOpsBackend backend)
    {
        int seqCond = condTokens.Length;
        int seqUncond = uncondTokens.Length;

        // 1. RelPosBiases
        var relPosCond = new float[Layers][];
        var relPosUncond = (seqCond == seqUncond) ? relPosCond : new float[Layers][];
        for (int i = 0; i < Layers; i++)
        {
            var rpW = Wt($"blocks.{i}.pos_embedding.embedding.weight");
            relPosCond[i] = ComputeRelPosBias(rpW, seqCond, Heads);
            if (seqCond != seqUncond)
            {
                relPosUncond[i] = ComputeRelPosBias(rpW, seqUncond, Heads);
            }
        }

        using var wsCond = new UMT5GpuWorkspace(backend, seqCond, relPosCond, Dim, Heads, HeadDim, FfDim);
        using var wsUncond = new UMT5GpuWorkspace(backend, seqUncond, relPosUncond, Dim, Heads, HeadDim, FfDim);

        // 2. Token embedding lookup (only the needed rows)
        var xCondHost = LookupTokenEmbeddings(condTokens);
        var xUncondHost = LookupTokenEmbeddings(uncondTokens);

        using (var xInitCond = backend.Upload(xCondHost, TensorShape.D2(seqCond, Dim), exact: true))
        {
            ((IImageOpsBackend)backend).ScaleInPlace(wsCond.X, 0f);
            backend.AddInPlace(wsCond.X, xInitCond);
        }
        using (var xInitUncond = backend.Upload(xUncondHost, TensorShape.D2(seqUncond, Dim), exact: true))
        {
            ((IImageOpsBackend)backend).ScaleInPlace(wsUncond.X, 0f);
            backend.AddInPlace(wsUncond.X, xInitUncond);
        }

        // 3. 24 Transformer Blocks - stream each layer ONCE for both sequences
        bool hasGpuWeights = _gpuWeights != null;
        var imageOps = backend as IImageOpsBackend;
        if (hasGpuWeights)
        {
            const int chunkLayers = 4;
            for (int i = 0; i < Layers; i += chunkLayers)
            {
                imageOps?.BeginBatch();
                int end = Math.Min(i + chunkLayers, Layers);
                for (int l = i; l < end; l++)
                {
                    var lw = _gpuWeights!.Layers[l];
                    ExecuteLayerGpu(wsCond, lw, l, seqCond, backend);
                    ExecuteLayerGpu(wsUncond, lw, l, seqUncond, backend);
                }
                imageOps?.EndBatch();
            }
        }
        else
        {
            for (int i = 0; i < Layers; i++)
            {
                var lw = new UMT5GpuWeights.UMT5LayerGpuWeights(backend, _st.ReadF32, i, Dim, FfDim);
                try
                {
                    imageOps?.BeginBatch();
                    ExecuteLayerGpu(wsCond, lw, i, seqCond, backend);
                    ExecuteLayerGpu(wsUncond, lw, i, seqUncond, backend);
                    imageOps?.EndBatch();
                }
                finally
                {
                    lw.Dispose();
                }
            }
        }

        // 4. Final RMSNorm
        using var finalNorm = backend.Upload(Wt("norm.weight"), TensorShape.D1(Dim), exact: true);
        backend.RmsNormBatched(wsCond.X, wsCond.X, finalNorm, Dim, seqCond, eps: 1e-6f);
        backend.RmsNormBatched(wsUncond.X, wsUncond.X, finalNorm, Dim, seqUncond, eps: 1e-6f);

        // 5. Download results
        var resultCond = new float[seqCond * Dim];
        var resultUncond = new float[seqUncond * Dim];
        backend.Download(wsCond.X, resultCond);
        backend.Download(wsUncond.X, resultUncond);

        return (resultCond, resultUncond);
    }

    private static void ExecuteLayerGpu(UMT5GpuWorkspace ws, UMT5GpuWeights.UMT5LayerGpuWeights lw, int i, int seq, IVisionOpsBackend backend)
    {
        // -- Self-Attention Sub-layer --
        backend.RmsNormBatched(ws.XNorm, ws.X, lw.LayerNorm0Weight, Dim, seq, eps: 1e-6f);

        backend.Sgemm(ws.Q, ws.XNorm, lw.QWeight, seq, Dim, Dim);
        backend.Sgemm(ws.K, ws.XNorm, lw.KWeight, seq, Dim, Dim);
        backend.Sgemm(ws.V, ws.XNorm, lw.VWeight, seq, Dim, Dim);

        backend.T5MultiHeadAttentionRelBias(ws.AttnOut, ws.Q, ws.K, ws.V, ws.RelPosBias[i], seq, seq, Heads, HeadDim);

        backend.Sgemm(ws.XNorm, ws.AttnOut, lw.OWeight, seq, Dim, Dim);
        backend.AddInPlace(ws.X, ws.XNorm);

        // -- Feed-Forward Sub-layer --
        backend.RmsNormBatched(ws.XNorm, ws.X, lw.LayerNorm1Weight, Dim, seq, eps: 1e-6f);

        backend.Sgemm(ws.Gate, ws.XNorm, lw.Wi0Weight, seq, Dim, FfDim);
        backend.Sgemm(ws.Val, ws.XNorm, lw.Wi1Weight, seq, Dim, FfDim);

        backend.GeluTanhMul(ws.Gate, ws.Val);

        backend.Sgemm(ws.FfOut, ws.Gate, lw.WoWeight, seq, FfDim, Dim);
        backend.AddInPlace(ws.X, ws.FfOut);
    }

    /// <summary>Encode token ids -> context embeddings [seq, 4096] on CPU.</summary>
    public float[] Encode(int[] tokens)
    {
        int seq = tokens.Length;
        var x = LookupTokenEmbeddings(tokens);

        for (int i = 0; i < Layers; i++)
            x = EncoderBlock(x, seq, i);

        var fnW = Wt("norm.weight");
        DiffusionOps.RmsNorm(x, fnW, Dim);
        return x;
    }

    private float[] EncoderBlock(float[] x, int seq, int blockIdx)
    {
        string p = $"blocks.{blockIdx}";

        // Real Wan UMT5: every block has its own relative position bias.
        var rpW = Wt($"{p}.pos_embedding.embedding.weight");
        var relPosBias = ComputeRelPosBias(rpW, seq, Heads);

        var lnW0 = Wt($"{p}.norm1.weight");
        var xNorm = x.ToArray();
        DiffusionOps.RmsNorm(xNorm, lnW0, Dim);
        var attn = SelfAttention(xNorm, relPosBias, seq, $"{p}.attn");
        for (int i = 0; i < x.Length; i++) x[i] += attn[i];

        var lnW1 = Wt($"{p}.norm2.weight");
        var xNorm2 = x.ToArray();
        DiffusionOps.RmsNorm(xNorm2, lnW1, Dim);
        var ff = FeedForward(xNorm2, seq, $"{p}.ffn");
        for (int i = 0; i < x.Length; i++) x[i] += ff[i];

        return x;
    }

    private float[] SelfAttention(float[] x, float[] relBias, int seq, string p)
    {
        var qW = Wt($"{p}.q.weight");
        var kW = Wt($"{p}.k.weight");
        var vW = Wt($"{p}.v.weight");
        var oW = Wt($"{p}.o.weight");

        var q = DiffusionOps.Linear(x, qW, null, seq, Dim, Dim);
        var k = DiffusionOps.Linear(x, kW, null, seq, Dim, Dim);
        var v = DiffusionOps.Linear(x, vW, null, seq, Dim, Dim);

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
        var gateW = Wt($"{p}.gate.0.weight");
        var fc1W  = Wt($"{p}.fc1.weight");
        var fc2W  = Wt($"{p}.fc2.weight");

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

    public void Dispose()
    {
        if (_ownsLoader) _st.Dispose();
        _gpuWorkspace?.Dispose();
        _gpuWeights?.Dispose();
    }
}
