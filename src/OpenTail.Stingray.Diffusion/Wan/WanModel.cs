using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.Wan;

/// <summary>
/// Native C# Wan 2.1 / 2.2 Video Diffusion Transformer (DiT).
/// Reference: stable-diffusion.cpp:src/model/diffusion/wan.hpp:WanModel
/// </summary>
public sealed class WanModel : IDisposable
{
    private readonly IWeightLoader _weights;
    private readonly string _prefix;
    private readonly IComputeBackend? _backend;
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    private readonly Dictionary<string, CoreTensor>? _gpuWeights;
    private readonly int _numLayers;
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _ffnDim;
    private bool _disposed;

    public const int InChannels = 64;   // 16 * 2 * 2
    public const int OutChannels = 16;
    public const int TextDim = 4096;    // UMT5 / T5-XXL text dimension

    public int NumLayers => _numLayers;
    public int Dim => _dim;
    public int NumHeads => _numHeads;
    public int HeadDim => _headDim;
    public int FfnDim => _ffnDim;
    public IComputeBackend? Backend => _backend;

    private WanGpuWeights? _gpuWeightsResident;

    public WanGpuWeights GetOrCreateGpuWeights()
    {
        if (_backend is null) throw new InvalidOperationException("No compute backend configured for WanModel GPU residency.");
        return _gpuWeightsResident ??= new WanGpuWeights(_backend, _weights, _prefix, _numLayers, _dim, _ffnDim);
    }

    public WanModel(IWeightLoader weights, string prefix = "", int numLayers = 30, int dim = 1536, int numHeads = 12, IComputeBackend? backend = null)
    {
        _weights = weights;
        _prefix = prefix;
        _backend = backend;
        (_numLayers, _dim, _numHeads) = DetectConfig(weights, prefix, numLayers, dim, numHeads);
        _headDim = _dim / _numHeads;
        _ffnDim = _dim == 1536 ? 8960 : (_dim == 5120 ? 13824 : _dim * 4);
        if (backend is not null)
            _gpuWeights = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
    }

    private static (int numLayers, int dim, int numHeads) DetectConfig(IWeightLoader weights, string prefix, int defLayers, int defDim, int defHeads)
    {
        int detectedLayers = defLayers;
        for (int i = 60; i >= 0; i--)
        {
            string key = $"{prefix}blocks.{i}.self_attn.q.weight";
            if (weights.Contains(key) || weights.Contains("model.diffusion_model." + key))
            {
                detectedLayers = i + 1;
                break;
            }
        }

        int detectedDim = defDim;
        string patchKey = $"{prefix}patch_embedding.weight";
        if (!weights.Contains(patchKey)) patchKey = "model.diffusion_model." + patchKey;
        if (weights.Contains(patchKey))
        {
            var w = weights.ReadF32(patchKey);
            detectedDim = w.Length / InChannels;
        }

        int detectedHeads = detectedDim == 1536 ? 12 : (detectedDim == 5120 ? 40 : defHeads);
        return (detectedLayers, detectedDim, detectedHeads);
    }

    private string Resolve(string name)
    {
        string direct = _prefix + name;
        if (_weights.Contains(direct)) return direct;
        if (_weights.Contains("model.diffusion_model." + direct)) return "model.diffusion_model." + direct;
        if (_weights.Contains("diffusion_model." + direct)) return "diffusion_model." + direct;
        return direct;
    }

    private float[] GetWeight(string name)
    {
        string fullName = Resolve(name);
        if (_weightCache.TryGetValue(fullName, out var cached)) return cached;
        var data = _weights.ReadF32(fullName);
        _weightCache[fullName] = data;
        return data;
    }

    private float[]? TryGetWeight(string name)
    {
        string fullName = Resolve(name);
        if (_weightCache.TryGetValue(fullName, out var cached)) return cached;
        if (_weights.Contains(fullName))
        {
            var data = _weights.ReadF32(fullName);
            _weightCache[fullName] = data;
            return data;
        }
        return null;
    }

    /// <summary>
    /// Precomputes and caches cross-attention K and V projections for all transformer blocks for a given text context.
    /// Invariant across timesteps and CFG branches.
    /// </summary>
    public void PrecomputeCrossKvCache(float[] textContext, WanWorkspace ws)
    {
        var txtProj = ComputeTextEmbedding(textContext);
        int numTxtTokens = textContext.Length / TextDim;

        for (int b = 0; b < _numLayers; b++)
        {
            string p = $"blocks.{b}";
            var k = new float[numTxtTokens * _dim];
            var v = new float[numTxtTokens * _dim];

            Linear($"{p}.cross_attn.k", txtProj, k.AsSpan(), _dim, _dim);
            Linear($"{p}.cross_attn.v", txtProj, v.AsSpan(), _dim, _dim);

            var normK = TryGetWeight($"{p}.cross_attn.norm_k.weight");
            if (normK is not null) RmsNormHeads(k, numTxtTokens, _numHeads, _headDim, normK);

            ws.CrossKvCache[b] = (k, v);
        }
    }

    /// <summary>
    /// Precomputes and caches cross-attention K and V projections on GPU for all transformer blocks.
    /// Invariant across timesteps and CFG branches.
    /// </summary>
    public void PrecomputeCrossKvCacheGpu(
        float[] textContext,
        WanGpuWorkspace gpuWs,
        WanGpuWeights gpuWeights,
        IImageOpsBackend imageOps)
    {
        var txtProj = ComputeTextEmbedding(textContext);
        int numTxtTokens = textContext.Length / TextDim;

        using var txtProjGpu = imageOps.Upload(txtProj, TensorShape.D2(numTxtTokens, _dim));

        for (int b = 0; b < _numLayers; b++)
        {
            var block = gpuWeights.Blocks[b];
            imageOps.Sgemm(gpuWs.CrossKvCache[b].K, txtProjGpu, block.CrossAttnK, numTxtTokens, _dim, _dim);
            imageOps.Sgemm(gpuWs.CrossKvCache[b].V, txtProjGpu, block.CrossAttnV, numTxtTokens, _dim, _dim);

            if (block.CrossAttnNormK is not null)
            {
                imageOps.RmsNorm(gpuWs.CrossKvCache[b].K, gpuWs.CrossKvCache[b].K, block.CrossAttnNormK);
            }
        }
    }

    /// <summary>
    /// Executes the GPU-resident forward pass of the Wan DiT model using WanGpuWeights and WanGpuWorkspace.
    /// </summary>
    public float[] ForwardGpu(
        float[] latent,
        float timestep,
        float[] textContext,
        int numFrames,
        int latH,
        int latW,
        WanGpuWorkspace gpuWs,
        WanGpuWeights gpuWeights,
        IImageOpsBackend imageOps)
    {
        int patchH = latH / 2;
        int patchW = latW / 2;
        int numTokens = numFrames * patchH * patchW;

        // 1. Pack latents [16, numFrames, latH, latW] -> [numTokens, 64] and upload to GPU
        var packed = PackLatents(latent, numFrames, latH, latW);
        using var packedGpu = imageOps.Upload(packed, TensorShape.D2(numTokens, InChannels));

        // 2. Patch input projection on GPU
        imageOps.Sgemm(gpuWs.X, packedGpu, gpuWeights.PatchEmbedding, numTokens, InChannels, _dim);

        // 3. Timestep embedding (sinusoidal 256 -> linear dim -> silu -> linear dim) -- tiny
        // (dim-sized) vectors, computed on host; not worth a GPU round-trip.
        var tEmb = ComputeTimestepEmbedding(timestep);
        var timeProjSilu = (float[])tEmb.Clone();
        DiffusionOps.SiluInPlace(timeProjSilu);
        var timestepProj = Linear("time_projection.1", timeProjSilu, _dim, _dim * 6);

        var visionOps = (IVisionOpsBackend)imageOps;

        // 4. Transformer Blocks -- fully GPU-resident (see TransformerBlockGpu): no per-block
        // host round-trips. gpuWs.RopeCos/RopeSin were uploaded once at WanGpuWorkspace
        // 4. Transformer Blocks -- fully GPU-resident (see TransformerBlockGpu): no per-block
        // host round-trips. gpuWs.RopeCos/RopeSin were uploaded once at WanGpuWorkspace
        // construction (see WanRoPE.Compute3DRoPECompact).
        var hostMod = new float[_dim * 6];

        // Pre-write all layer modulations upfront
        for (int b = 0; b < _numLayers; b++)
        {
            var bw = gpuWeights.Blocks[b];
            var modParam = bw.HostModulation;
            for (int i = 0; i < hostMod.Length; i++) hostMod[i] = modParam[i] + timestepProj[i];
            imageOps.WritePinned(gpuWs.LayerMods[b], hostMod);
        }

        // Full-graph batching: record all 30 blocks per command buffer submission to eliminate driver fence bubbles
        const int batchBlocks = 30;
        for (int chunkStart = 0; chunkStart < _numLayers; chunkStart += batchBlocks)
        {
            int chunkEnd = Math.Min(_numLayers, chunkStart + batchBlocks);
            imageOps.BeginBatch();
            for (int b = chunkStart; b < chunkEnd; b++)
            {
                var bw = gpuWeights.Blocks[b];
                TransformerBlockGpu(b, bw, gpuWs, gpuWs.LayerMods[b], numTokens, imageOps, visionOps);
            }
            imageOps.EndBatch();
        }

        // 5. Final Layer (AdaLN + Linear dim -> 64)
        var headModParam = gpuWeights.HostHeadModulation;
        var headMod = new float[_dim * 2];
        for (int d = 0; d < _dim; d++)
        {
            headMod[d] = headModParam[d] + tEmb[d];
            headMod[_dim + d] = headModParam[_dim + d] + tEmb[d];
        }
        using var headModGpu = imageOps.Upload(headMod, TensorShape.D1(_dim * 2), exact: true);
        visionOps.AdaLNModulate(gpuWs.Normed1, gpuWs.X, headModGpu, numTokens, _dim, shiftOffset: 0, scaleOffset: _dim, isRmsNorm: false, eps: 1e-6f);

        imageOps.Sgemm(gpuWs.OutPacked, gpuWs.Normed1, gpuWeights.HeadWeight, numTokens, _dim, InChannels);

        var outPacked = new float[numTokens * InChannels];
        imageOps.Download(gpuWs.OutPacked, outPacked);

        // 6. Unpack patches [numTokens, 64] -> [16, numFrames, latH, latW]
        return UnpackLatents(outPacked, numFrames, latH, latW);
    }

    /// <summary>
    /// GPU-resident equivalent of <see cref="TransformerBlock"/> -- same math, every op dispatched
    /// on <paramref name="imageOps"/>/<paramref name="visionOps"/> against <paramref name="ws"/>'s
    /// preallocated device buffers, zero host round-trips. <paramref name="modTensor"/> holds this layer's
    /// `modulation + timestepProj`.
    /// </summary>
    private void TransformerBlockGpu(
        int layerIdx,
        WanGpuWeights.BlockWeights bw,
        WanGpuWorkspace ws,
        CoreTensor modTensor,
        int numTokens,
        IImageOpsBackend imageOps,
        IVisionOpsBackend visionOps)
    {
        int d = _dim;

        // 1. Self-attention: affine-free LayerNorm -> AdaLN modulate -> fused QKV GEMM -> fused Norm+RoPE -> self-attn -> gated residual.
        visionOps.AdaLNModulate(ws.Normed1, ws.X, modTensor, numTokens, d, shiftOffset: 0, scaleOffset: d, isRmsNorm: false, eps: 1e-6f);

        // Fused QKV GEMM [numTokens, d] x [d, d*3] -> [numTokens, d*3]
        imageOps.Sgemm(ws.Qkv, ws.Normed1, bw.SelfAttnQkv, numTokens, d, d * 3);

        // Fused QKV Split + per-head RMSNorm + 3D RoPE (GPT-NeoX adjacent pair rotation)
        visionOps.WanQkvSplitNormRoPE(ws.Qkv, ws.Q, ws.K, ws.V, ws.RopeCos, ws.RopeSin,
            bw.SelfAttnNormQ, bw.SelfAttnNormK, numTokens, _numHeads, _headDim, eps: 1e-6f);

        imageOps.MultiHeadAttentionTiled(ws.AttnOut, ws.Q, ws.K, ws.V, numTokens, numTokens, _numHeads, _headDim);
        imageOps.Sgemm(ws.CrossAttnOut, ws.AttnOut, bw.SelfAttnO, numTokens, d, d);
        visionOps.ScaleGateAdd(ws.X, ws.CrossAttnOut, modTensor, numTokens, d, gateOffset: 2 * d);

        // 2. Cross-Attention with T5/UMT5 text tokens (using precomputed K/V cache).
        imageOps.LayerNormGpu(ws.NormedCross, ws.X, bw.Norm3Weight, bw.Norm3Bias, numTokens, d);
        imageOps.Sgemm(ws.CrossQ, ws.NormedCross, bw.CrossAttnQ, numTokens, d, d);
        if (bw.CrossAttnNormQ is not null) visionOps.RmsNormBatched(ws.CrossQ, ws.CrossQ, bw.CrossAttnNormQ, d, numTokens);

        var (cachedK, cachedV) = ws.CrossKvCache[layerIdx];
        int ctxLen = (int)cachedK.Shape.Dims[0];
        imageOps.MultiHeadAttentionTiled(ws.CrossAttnOut, ws.CrossQ, cachedK, cachedV, numTokens, ctxLen, _numHeads, _headDim);
        imageOps.Sgemm(ws.AttnOut, ws.CrossAttnOut, bw.CrossAttnO, numTokens, d, d);
        imageOps.AddInPlace(ws.X, ws.AttnOut);

        // 3. Modulated FeedForward (GELU approx tanh): affine-free LayerNorm -> AdaLN modulate -> FFN -> gated residual.
        visionOps.AdaLNModulate(ws.Normed2, ws.X, modTensor, numTokens, d, shiftOffset: 3 * d, scaleOffset: 4 * d, isRmsNorm: false, eps: 1e-6f);
        imageOps.Sgemm(ws.Ffn1, ws.Normed2, bw.Ffn0, numTokens, d, _ffnDim);
        visionOps.VisionGeluInPlace(ws.Ffn1);
        imageOps.Sgemm(ws.FfnOut, ws.Ffn1, bw.Ffn2, numTokens, _ffnDim, d);
        visionOps.ScaleGateAdd(ws.X, ws.FfnOut, modTensor, numTokens, d, gateOffset: 5 * d);
    }

    /// <summary>
    /// Executes the forward pass of the Wan DiT model.
    /// </summary>
    public float[] Forward(
        float[] latent,
        float timestep,
        float[] textContext,
        int numFrames,
        int latH,
        int latW,
        WanWorkspace? ws = null)
    {
        int patchH = latH / 2;
        int patchW = latW / 2;
        int numTokens = numFrames * patchH * patchW;
        int numTxtTokens = textContext.Length / TextDim;

        ws ??= new WanWorkspace(numTokens, _dim, _ffnDim, _numLayers);
        if (ws.CrossKvCache == null || ws.CrossKvCache.Length < _numLayers || ws.CrossKvCache[0].K == null)
        {
            PrecomputeCrossKvCache(textContext, ws);
        }

        // 1. Pack 16-channel video latent into 64-channel patches [numTokens, 64]
        var packed = PackLatents(latent, numFrames, latH, latW);

        // 2. Patch input projection
        var x = Linear("patch_embedding", packed, InChannels, _dim);

        // 3. Timestep embedding (sinusoidal 256 -> linear dim -> silu -> linear dim)
        var tEmb = ComputeTimestepEmbedding(timestep);
        var timeProjSilu = (float[])tEmb.Clone();
        DiffusionOps.SiluInPlace(timeProjSilu);
        var timestepProj = Linear("time_projection.1", timeProjSilu, _dim, _dim * 6);

        // 4. 3D-RoPE positional frequencies
        var (cos, sin) = WanRoPE.Compute3DRoPE(numFrames, patchH, patchW, _headDim);

        // 5. Transformer Blocks
        for (int b = 0; b < _numLayers; b++)
        {
            string p = $"blocks.{b}";
            TransformerBlock(b, p, x, timestepProj, cos, sin, numTokens, numTxtTokens, ws);
        }

        // 6. Final Layer (AdaLN + Linear dim -> 64)
        var headModParam = GetWeight("head.modulation");
        for (int d = 0; d < _dim; d++)
        {
            ws.HeadShift[d] = headModParam[d] + tEmb[d];
            ws.HeadScale[d] = headModParam[_dim + d] + tEmb[d];
        }

        DiffusionOps.LayerNormNoAffine(x.AsSpan(0, numTokens * _dim), ws.Norm1.AsSpan(0, numTokens * _dim), _dim);
        DiffusionOps.ModulateRows(ws.Norm1.AsSpan(0, numTokens * _dim), ws.Normed1.AsSpan(0, numTokens * _dim), numTokens, _dim, ws.HeadShift, ws.HeadScale);

        var outPacked = Linear("head.head", ws.Normed1, _dim, InChannels);

        // 7. Unpack patches [numTokens, 64] -> [16, numFrames, latH, latW]
        return UnpackLatents(outPacked, numFrames, latH, latW);
    }

    private void TransformerBlock(
        int layerIdx,
        string prefix,
        float[] x,
        float[] timestepProj,
        float[] cos,
        float[] sin,
        int numTokens,
        int numTxt,
        WanWorkspace ws)
    {
        var modParam = GetWeight($"{prefix}.modulation");
        for (int i = 0; i < ws.Mod.Length; i++) ws.Mod[i] = modParam[i] + timestepProj[i];
        var s1 = ws.Mod.AsSpan(0 * _dim, _dim);
        var sc1 = ws.Mod.AsSpan(1 * _dim, _dim);
        var g1 = ws.Mod.AsSpan(2 * _dim, _dim);
        var s2 = ws.Mod.AsSpan(3 * _dim, _dim);
        var sc2 = ws.Mod.AsSpan(4 * _dim, _dim);
        var g2 = ws.Mod.AsSpan(5 * _dim, _dim);

        // 1. Self-attention: affine-free LayerNorm -> AdaLN modulate -> self-attn (3D-RoPE) -> gated residual.
        DiffusionOps.LayerNormNoAffine(x.AsSpan(0, numTokens * _dim), ws.Norm1.AsSpan(0, numTokens * _dim), _dim);
        DiffusionOps.ModulateRows(ws.Norm1.AsSpan(0, numTokens * _dim), ws.Normed1.AsSpan(0, numTokens * _dim), numTokens, _dim, s1, sc1);
        SelfAttention($"{prefix}.self_attn", ws.Normed1, ws, cos, sin, numTokens);
        DiffusionOps.ApplyGatedResidualRows(x.AsSpan(0, numTokens * _dim), ws.CrossAttnOut.AsSpan(0, numTokens * _dim), numTokens, _dim, g1);

        // 2. Cross-Attention with T5/UMT5 text tokens (using precomputed K/V cache)
        var norm3W = GetWeight($"{prefix}.norm3.weight");
        var norm3B = GetWeight($"{prefix}.norm3.bias");
        DiffusionOps.LayerNorm(x.AsSpan(0, numTokens * _dim), ws.NormedCross.AsSpan(0, numTokens * _dim), norm3W, norm3B, _dim);
        CrossAttention(layerIdx, $"{prefix}.cross_attn", ws.NormedCross, ws, numTokens, numTxt);
        TensorPrimitives.Add(x.AsSpan(0, numTokens * _dim), ws.AttnOut.AsSpan(0, numTokens * _dim), x.AsSpan(0, numTokens * _dim));

        // 3. Modulated FeedForward (GELU approx tanh): affine-free LayerNorm -> AdaLN modulate -> FFN -> gated residual.
        DiffusionOps.LayerNormNoAffine(x.AsSpan(0, numTokens * _dim), ws.Norm2.AsSpan(0, numTokens * _dim), _dim);
        DiffusionOps.ModulateRows(ws.Norm2.AsSpan(0, numTokens * _dim), ws.Normed2.AsSpan(0, numTokens * _dim), numTokens, _dim, s2, sc2);
        FeedForward($"{prefix}.ffn", ws.Normed2, ws, numTokens);
        DiffusionOps.ApplyGatedResidualRows(x.AsSpan(0, numTokens * _dim), ws.FfnOut.AsSpan(0, numTokens * _dim), numTokens, _dim, g2);
    }

    private void SelfAttention(string prefix, float[] x, WanWorkspace ws, float[] cos, float[] sin, int seqLen)
    {
        Linear($"{prefix}.q", x, ws.Q.AsSpan(0, seqLen * _dim), _dim, _dim);
        Linear($"{prefix}.k", x, ws.K.AsSpan(0, seqLen * _dim), _dim, _dim);
        Linear($"{prefix}.v", x, ws.V.AsSpan(0, seqLen * _dim), _dim, _dim);

        // RMSNorm on Q and K per head
        var normQ = TryGetWeight($"{prefix}.norm_q.weight");
        if (normQ is not null) RmsNormHeads(ws.Q, seqLen, _numHeads, _headDim, normQ);
        var normK = TryGetWeight($"{prefix}.norm_k.weight");
        if (normK is not null) RmsNormHeads(ws.K, seqLen, _numHeads, _headDim, normK);

        // Apply 3D-RoPE
        WanRoPE.ApplyRoPE(ws.Q, cos, sin, seqLen, _numHeads, _headDim);
        WanRoPE.ApplyRoPE(ws.K, cos, sin, seqLen, _numHeads, _headDim);

        WanAttention.TiledMultiHeadAttention(ws.Q, ws.K, ws.V, ws.AttnOut.AsSpan(0, seqLen * _dim), seqLen, seqLen, _numHeads, _headDim);
        Linear($"{prefix}.o", ws.AttnOut, ws.CrossAttnOut.AsSpan(0, seqLen * _dim), _dim, _dim);
    }

    private void CrossAttention(int layerIdx, string prefix, float[] x, WanWorkspace ws, int seqLen, int ctxLen)
    {
        Linear($"{prefix}.q", x, ws.CrossQ.AsSpan(0, seqLen * _dim), _dim, _dim);

        var normQ = TryGetWeight($"{prefix}.norm_q.weight");
        if (normQ is not null) RmsNormHeads(ws.CrossQ, seqLen, _numHeads, _headDim, normQ);

        var (cachedK, cachedV) = ws.CrossKvCache[layerIdx];
        WanAttention.TiledMultiHeadAttention(ws.CrossQ, cachedK, cachedV, ws.CrossAttnOut.AsSpan(0, seqLen * _dim), seqLen, ctxLen, _numHeads, _headDim);
        Linear($"{prefix}.o", ws.CrossAttnOut, ws.AttnOut.AsSpan(0, seqLen * _dim), _dim, _dim);
    }

    /// <summary>Real Wan QK-norm: `torch.nn.RMSNorm(dim_head * heads, ...)` -- ONE RMS statistic
    /// over the FULL projected `dim`-length row (all heads concatenated), computed and applied
    /// BEFORE the conceptual split into heads, with the full `dim`-length learned gamma indexed
    /// contiguously across the whole row (`gamma[h*headDim+d]`, not `gamma[d]` reused per head).
    /// Confirmed against `WanAttention.__init__`/`forward` (`query = attn.norm_q(query)` runs on
    /// the un-split `(seq, dim)` tensor, `unflatten(2, (heads, -1))` happens only afterward) --
    /// found and fixed 2026-08-31 after this file's previous per-head-RMS-with-truncated-gamma
    /// version was root-caused as corrupting every attention call in every block. Note: since Q/K
    /// buffers are already row-major `[seqLen, dim]` with each token's dim-length row itself
    /// contiguous across heads (`h*headDim+d` sweeps 0..dim-1), "the full row" and "all heads
    /// concatenated" are the same contiguous span -- no data layout change needed here.</summary>
    private static void RmsNormHeads(float[] qk, int seqLen, int numHeads, int headDim, float[] gamma)
    {
        int dim = numHeads * headDim;
        Parallel.For(0, seqLen, s =>
        {
            int rowOff = s * dim;
            var rowSpan = qk.AsSpan(rowOff, dim);
            float sumSq = TensorPrimitives.SumOfSquares(rowSpan);
            float invRms = 1.0f / MathF.Sqrt(sumSq / dim + 1e-6f);
            for (int d = 0; d < dim; d++)
                qk[rowOff + d] = qk[rowOff + d] * invRms * gamma[d];
        });
    }

    private void FeedForward(string prefix, float[] x, WanWorkspace ws, int seqLen)
    {
        Linear($"{prefix}.0", x, ws.Ffn1.AsSpan(0, seqLen * _ffnDim), _dim, _ffnDim);
        DiffusionOps.GeluInPlace(ws.Ffn1.AsSpan(0, seqLen * _ffnDim));
        Linear($"{prefix}.2", ws.Ffn1, ws.FfnOut.AsSpan(0, seqLen * _dim), _ffnDim, _dim);
    }

    private float[] Modulate(float[] x, int seqLen, ReadOnlySpan<float> shift, ReadOnlySpan<float> scale)
        => DiffusionOps.ModulateRows(x, seqLen, _dim, shift, scale);

    private void ApplyGatedResidual(float[] x, float[] branch, int seqLen, ReadOnlySpan<float> gate)
        => DiffusionOps.ApplyGatedResidualRows(x, branch, seqLen, _dim, gate);

    private float[] ComputeTimestepEmbedding(float timestep)
    {
        var emb = DiffusionOps.SinusoidalTimestepEmbedding(timestep);
        var t0 = Linear("time_embedding.0", emb, 256, _dim);
        DiffusionOps.SiluInPlace(t0);
        return Linear("time_embedding.2", t0, _dim, _dim);
    }

    // Real Wan DiT text conditioning projection: `PixArtAlphaTextProjection(text_embed_dim, dim,
    // act_fn="gelu_tanh")` in transformer_wan.py's `WanTimeTextImageEmbedding` -- linear_1 -> GELU
    // (tanh-approx) -> linear_2, NOT a single linear (confirmed real GGUF tensor names are
    // `text_embedding.0`/`text_embedding.2`, matching `time_embedding`'s own two-layer shape).
    private float[] ComputeTextEmbedding(float[] textContext)
    {
        var t0 = Linear("text_embedding.0", textContext, TextDim, _dim);
        for (int i = 0; i < t0.Length; i++) t0[i] = DiffusionOps.Gelu(t0[i]);
        return Linear("text_embedding.2", t0, _dim, _dim);
    }

    private static float[] DiffusionOpsSilu(float[] x)
    {
        var res = (float[])x.Clone();
        DiffusionOps.SiluInPlace(res);
        return res;
    }

    private unsafe void Linear(string name, float[] x, Span<float> output, int inDim, int outDim)
    {
        var w = GetWeight($"{name}.weight");
        var b = TryGetWeight($"{name}.bias");
        int rows = x.Length / inDim;

        fixed (float* pOut = output, pIn = x, pW = w)
        {
            if (b is not null)
            {
                fixed (float* pB = b)
                {
                    SimdKernels.MatMulBatchedF32(pOut, pW, pIn, rows, outDim, inDim, pB);
                }
            }
            else
            {
                SimdKernels.MatMulBatchedF32(pOut, pW, pIn, rows, outDim, inDim, null);
            }
        }
    }

    private float[] Linear(string name, float[] x, int inDim, int outDim)
    {
        int rows = x.Length / inDim;
        var outF = new float[rows * outDim];
        Linear(name, x, outF.AsSpan(), inDim, outDim);
        return outF;
    }

    public static float[] PackLatents(float[] latents, int numFrames, int latH, int latW)
    {
        int patchH = latH / 2;
        int patchW = latW / 2;
        int numTokens = numFrames * patchH * patchW;
        var packed = new float[numTokens * InChannels];

        for (int f = 0; f < numFrames; f++)
        {
            for (int ph = 0; ph < patchH; ph++)
            {
                for (int pw = 0; pw < patchW; pw++)
                {
                    int tokenIdx = (f * patchH + ph) * patchW + pw;
                    int tokenOff = tokenIdx * InChannels;
                    int chanOffset = 0;

                    for (int c = 0; c < OutChannels; c++)
                    {
                        for (int dy = 0; dy < 2; dy++)
                        {
                            for (int dx = 0; dx < 2; dx++)
                            {
                                int y = ph * 2 + dy;
                                int x = pw * 2 + dx;
                                int srcIdx = ((c * numFrames + f) * latH + y) * latW + x;
                                packed[tokenOff + chanOffset++] = latents[srcIdx];
                            }
                        }
                    }
                }
            }
        }
        return packed;
    }

    public static float[] UnpackLatents(float[] packed, int numFrames, int latH, int latW)
    {
        int patchH = latH / 2;
        int patchW = latW / 2;
        var unpacked = new float[OutChannels * numFrames * latH * latW];

        for (int f = 0; f < numFrames; f++)
        {
            for (int ph = 0; ph < patchH; ph++)
            {
                for (int pw = 0; pw < patchW; pw++)
                {
                    int tokenIdx = (f * patchH + ph) * patchW + pw;
                    int tokenOff = tokenIdx * InChannels;

                    for (int dy = 0; dy < 2; dy++)
                    {
                        for (int dx = 0; dx < 2; dx++)
                        {
                            int y = ph * 2 + dy;
                            int x = pw * 2 + dx;
                            int spatialSubOff = (dy * 2 + dx) * OutChannels;

                            for (int c = 0; c < OutChannels; c++)
                            {
                                int dstIdx = ((c * numFrames + f) * latH + y) * latW + x;
                                unpacked[dstIdx] = packed[tokenOff + spatialSubOff + c];
                            }
                        }
                    }
                }
            }
        }
        return unpacked;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _gpuWeightsResident?.Dispose();
            _gpuWeightsResident = null;
            if (_gpuWeights is not null)
            {
                foreach (var tensor in _gpuWeights.Values)
                    _backend?.Free(tensor);
                _gpuWeights.Clear();
            }
            _weights.Dispose();
        }
    }
}
