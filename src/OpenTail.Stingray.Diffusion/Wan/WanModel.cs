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

        // Cross-reference debug override (2026-09-14, docs/081): when STINGRAY_WAN_DEBUG_CROSSREF=1,
        // substitute the C++ reference's own dumped POST-projection context (wandbg_context.bin --
        // i.e. the exact equivalent of this method's own `txtProj`) in place of this port's UMT5
        // output, so cross-attention's own math can be isolated from any difference between the two
        // UMT5 encoder files (C++ used a Q8_0 GGUF, this port its own bf16 safetensors -- a real
        // confound otherwise). Self-attention was already independently verified byte-identical
        // (cosine 0.9999999) without this override; this closes the same gap for cross-attention.
        if (Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF") == "1")
        {
            string dir = Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF_DIR") ?? "examples/stable-diffusion.cpp";
            string path = Path.Combine(dir, "wan_cpp_dump_wandbg_context.bin");
            if (File.Exists(path))
            {
                txtProj = ReadCrossRefBin(path);
                numTxtTokens = txtProj.Length / _dim;
                Console.Error.WriteLine($"[WanCrossRef] loaded context (post-projection) from {path}, numTxtTokens={numTxtTokens}");
            }
        }

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

        // Cross-reference debug override (2026-09-14, docs/081): mirrors the CPU path's own
        // identical override in PrecomputeCrossKvCache -- see that method's doc comment for the
        // full rationale (substitutes the C++ reference's own dumped post-projection context to
        // eliminate the UMT5-encoder-format confound between the two implementations).
        if (Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF") == "1")
        {
            string dir = Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF_DIR") ?? "examples/stable-diffusion.cpp";
            string path = Path.Combine(dir, "wan_cpp_dump_wandbg_context.bin");
            if (File.Exists(path))
            {
                txtProj = ReadCrossRefBin(path);
                numTxtTokens = txtProj.Length / _dim;
                Console.Error.WriteLine($"[WanCrossRef][GPU] loaded context (post-projection) from {path}, numTxtTokens={numTxtTokens}");
            }
        }

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

        // Cross-reference debug override (2026-09-14, docs/081): mirrors the CPU Forward()'s own
        // identical override -- loads the C++ reference's own dumped input latent/timestep in place
        // of this call's own, for a truly controlled, apples-to-apples comparison.
        if (Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF") == "1")
        {
            string dir = Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF_DIR") ?? "examples/stable-diffusion.cpp";
            string latentPath = Path.Combine(dir, "wan_cpp_dump_wandbg_input_latent.bin");
            if (File.Exists(latentPath))
            {
                latent = ReadCrossRefBin(latentPath);
                Console.Error.WriteLine($"[WanCrossRef][GPU] loaded input latent from {latentPath} ({latent.Length} floats)");
            }
            string timestepPath = Path.Combine(dir, "wan_cpp_dump_wandbg_timestep.bin");
            if (File.Exists(timestepPath))
            {
                var tArr = ReadCrossRefBin(timestepPath);
                if (tArr.Length > 0)
                {
                    timestep = tArr[0];
                    Console.Error.WriteLine($"[WanCrossRef][GPU] loaded timestep from {timestepPath} = {timestep}");
                }
            }
        }

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

        // Cross-reference debug dump (2026-09-14, docs/081): env-gated (STINGRAY_WAN_DEBUG_CROSSREF=1),
        // zero cost when unset. Forces batchBlocks=1 (one BeginBatch/EndBatch per block instead of
        // all 30 in one submission) so mid-block Download() calls inside TransformerBlockGpu are
        // safe (a batched command buffer isn't submitted until EndBatch(); downloading mid-batch
        // throws -- see this file's own earlier per-block CPU-round-trip precedent). Real timing
        // cost only when this env var is set; the real (non-debug) path is unaffected.
        bool debugCrossRefGpu = Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF") == "1";
        string debugCrossRefGpuDir = Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF_DIR") ?? "examples/stable-diffusion.cpp";
        if (debugCrossRefGpu)
        {
            var xHost = new float[numTokens * _dim];
            imageOps.Download(gpuWs.X, xHost);
            WriteCrossRefBin(debugCrossRefGpuDir, "wandbg_patchembed_gpu", xHost);
        }

        // Full-graph batching: record all 30 blocks per command buffer submission to eliminate driver fence bubbles
        int batchBlocks = debugCrossRefGpu ? 1 : 30;
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

            if (debugCrossRefGpu && chunkStart == 0)
            {
                var block0Host = new float[numTokens * _dim];
                imageOps.Download(gpuWs.X, block0Host);
                WriteCrossRefBin(debugCrossRefGpuDir, "wandbg_block0_gpu", block0Host);
            }
            if (debugCrossRefGpu && chunkEnd == _numLayers)
            {
                var blockLastHost = new float[numTokens * _dim];
                imageOps.Download(gpuWs.X, blockLastHost);
                WriteCrossRefBin(debugCrossRefGpuDir, "wandbg_blocklast_gpu", blockLastHost);
            }
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

        bool debugBlock0Gpu = layerIdx == 0 && Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF") == "1";
        string debugBlock0GpuDir = Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF_DIR") ?? "examples/stable-diffusion.cpp";
        if (debugBlock0Gpu)
        {
            imageOps.EndBatch();
            var selfAttnHost = new float[numTokens * d];
            imageOps.Download(ws.X, selfAttnHost);
            WriteCrossRefBin(debugBlock0GpuDir, "wandbg_block0_selfattn_gpu", selfAttnHost);
            imageOps.BeginBatch();
        }

        // 2. Cross-Attention with T5/UMT5 text tokens (using precomputed K/V cache).
        imageOps.LayerNormGpu(ws.NormedCross, ws.X, bw.Norm3Weight, bw.Norm3Bias, numTokens, d);
        imageOps.Sgemm(ws.CrossQ, ws.NormedCross, bw.CrossAttnQ, numTokens, d, d);
        if (bw.CrossAttnNormQ is not null) visionOps.RmsNormBatched(ws.CrossQ, ws.CrossQ, bw.CrossAttnNormQ, d, numTokens);

        var (cachedK, cachedV) = ws.CrossKvCache[layerIdx];
        int ctxLen = (int)cachedK.Shape.Dims[0];
        imageOps.MultiHeadAttentionTiled(ws.CrossAttnOut, ws.CrossQ, cachedK, cachedV, numTokens, ctxLen, _numHeads, _headDim);
        if (debugBlock0Gpu)
        {
            imageOps.EndBatch();
            var crossAttnRawHost = new float[numTokens * d];
            imageOps.Download(ws.CrossAttnOut, crossAttnRawHost);
            WriteCrossRefBin(debugBlock0GpuDir, "wandbg_block0_crossattn_raw_gpu", crossAttnRawHost);
            var qHost = new float[numTokens * d];
            imageOps.Download(ws.CrossQ, qHost);
            WriteCrossRefBin(debugBlock0GpuDir, "wandbg_block0_crossq_gpu", qHost);
            var kHost = new float[ctxLen * d];
            imageOps.Download(cachedK, kHost);
            WriteCrossRefBin(debugBlock0GpuDir, "wandbg_block0_crossk_gpu", kHost);
            imageOps.BeginBatch();
        }
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

        // Cross-reference debug dump (2026-09-14, docs/081): env-gated (STINGRAY_WAN_DEBUG_CROSSREF=1),
        // zero cost when unset. Mirrors the real C++ reference's own equivalent instrumentation
        // (examples/stable-diffusion.cpp's wan.hpp WanDebugDump + ggml_extend.hpp capture_tensor
        // dump) so intermediate tensors from both implementations, given the EXACT SAME input, can
        // be diffed byte-for-byte -- closing the "only reference formulas, never reference values"
        // gap this whole investigation was blocked on. When enabled, overrides `latent`/`timestep`
        // with the C++ reference's own dumped input (bypassing this port's own RNG, which won't
        // reproduce identical values from the same --seed as the C++ binary's RNG) so the comparison
        // starts from a truly controlled, apples-to-apples state.
        bool debugCrossRef = Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF") == "1";
        string crossRefDir = Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF_DIR")
            ?? "examples/stable-diffusion.cpp";
        if (debugCrossRef)
        {
            string latentPath = Path.Combine(crossRefDir, "wan_cpp_dump_wandbg_input_latent.bin");
            if (File.Exists(latentPath))
            {
                latent = ReadCrossRefBin(latentPath);
                Console.Error.WriteLine($"[WanCrossRef] loaded input latent from {latentPath} ({latent.Length} floats)");
            }
            string timestepPath = Path.Combine(crossRefDir, "wan_cpp_dump_wandbg_timestep.bin");
            if (File.Exists(timestepPath))
            {
                var tArr = ReadCrossRefBin(timestepPath);
                if (tArr.Length > 0)
                {
                    timestep = tArr[0];
                    Console.Error.WriteLine($"[WanCrossRef] loaded timestep from {timestepPath} = {timestep}");
                }
            }
        }

        ws ??= new WanWorkspace(numTokens, _dim, _ffnDim, _numLayers);
        if (ws.CrossKvCache == null || ws.CrossKvCache.Length < _numLayers || ws.CrossKvCache[0].K == null)
        {
            PrecomputeCrossKvCache(textContext, ws);
        }

        // 1. Pack 16-channel video latent into 64-channel patches [numTokens, 64]
        var packed = PackLatents(latent, numFrames, latH, latW);

        // 2. Patch input projection
        var x = Linear("patch_embedding", packed, InChannels, _dim);
        if (debugCrossRef) WriteCrossRefBin(crossRefDir, "wandbg_patchembed", x);

        // 3. Timestep embedding (sinusoidal 256 -> linear dim -> silu -> linear dim)
        var tEmb = ComputeTimestepEmbedding(timestep);
        var timeProjSilu = (float[])tEmb.Clone();
        DiffusionOps.SiluInPlace(timeProjSilu);
        var timestepProj = Linear("time_projection.1", timeProjSilu, _dim, _dim * 6);

        // 4. 3D-RoPE positional frequencies
        var (cos, sin) = WanRoPE.Compute3DRoPE(numFrames, patchH, patchW, _headDim);

        // 5. Transformer Blocks
        bool debugPerBlock = Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_PERBLOCK") == "1";
        for (int b = 0; b < _numLayers; b++)
        {
            string p = $"blocks.{b}";
            TransformerBlock(b, p, x, timestepProj, cos, sin, numTokens, numTxtTokens, ws);

            if (debugCrossRef && b == 0) WriteCrossRefBin(crossRefDir, "wandbg_block0", x);
            if (debugCrossRef && b == _numLayers - 1) WriteCrossRefBin(crossRefDir, "wandbg_blocklast", x);

            if (debugPerBlock)
            {
                float mean = 0, sumSq = 0, maxAbs = 0;
                for (int i = 0; i < x.Length; i++)
                {
                    mean += x[i]; sumSq += x[i] * x[i];
                    float a = MathF.Abs(x[i]);
                    if (a > maxAbs) maxAbs = a;
                }
                mean /= x.Length;
                float std = MathF.Sqrt(Math.Max(0, sumSq / x.Length - mean * mean));
                bool bad = float.IsNaN(mean) || float.IsInfinity(mean) || maxAbs > 1e6f;
                Console.WriteLine($"[WanDebug][PerBlock] block={b} mean={mean:F6} std={std:F6} maxAbs={maxAbs:F4}{(bad ? " <<< PATHOLOGICAL" : "")}");
            }
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
        if (debugCrossRef) WriteCrossRefBin(crossRefDir, "wandbg_head", outPacked);

        // 7. Unpack patches [numTokens, 64] -> [16, numFrames, latH, latW]
        return UnpackLatents(outPacked, numFrames, latH, latW);
    }

    private static float[] ReadCrossRefBin(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var arr = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);
        return arr;
    }

    private static void WriteCrossRefBin(string dir, string name, float[] data)
    {
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, $"wan_cs_dump_{name}.bin");
        var bytes = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
        Console.Error.WriteLine($"[WanCrossRef] wrote {path} ({data.Length} floats)");
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
        if (layerIdx == 0 && Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF") == "1")
        {
            WriteCrossRefBin(
                Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF_DIR") ?? "examples/stable-diffusion.cpp",
                "wandbg_block0_selfattn", x.AsSpan(0, numTokens * _dim).ToArray());
        }

        // 2. Cross-Attention with T5/UMT5 text tokens (using precomputed K/V cache)
        var norm3W = GetWeight($"{prefix}.norm3.weight");
        var norm3B = GetWeight($"{prefix}.norm3.bias");
        DiffusionOps.LayerNorm(x.AsSpan(0, numTokens * _dim), ws.NormedCross.AsSpan(0, numTokens * _dim), norm3W, norm3B, _dim);
        CrossAttention(layerIdx, $"{prefix}.cross_attn", ws.NormedCross, ws, numTokens, numTxt);
        TensorPrimitives.Add(x.AsSpan(0, numTokens * _dim), ws.AttnOut.AsSpan(0, numTokens * _dim), x.AsSpan(0, numTokens * _dim));
        if (layerIdx == 0 && Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF") == "1")
        {
            WriteCrossRefBin(
                Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF_DIR") ?? "examples/stable-diffusion.cpp",
                "wandbg_block0_crossattn", x.AsSpan(0, numTokens * _dim).ToArray());
        }

        // 3. Modulated FeedForward (GELU approx tanh): affine-free LayerNorm -> AdaLN modulate -> FFN -> gated residual.
        DiffusionOps.LayerNormNoAffine(x.AsSpan(0, numTokens * _dim), ws.Norm2.AsSpan(0, numTokens * _dim), _dim);
        DiffusionOps.ModulateRows(ws.Norm2.AsSpan(0, numTokens * _dim), ws.Normed2.AsSpan(0, numTokens * _dim), numTokens, _dim, s2, sc2);
        bool debugFfn0 = layerIdx == 0 && Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF") == "1";
        string debugFfnDir = Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF_DIR") ?? "examples/stable-diffusion.cpp";
        if (debugFfn0) WriteCrossRefBin(debugFfnDir, "wandbg_block0_ffnin", ws.Normed2.AsSpan(0, numTokens * _dim).ToArray());
        FeedForward($"{prefix}.ffn", ws.Normed2, ws, numTokens, debugFfn0 ? debugFfnDir : null);
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

        if (Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_ATTNMAP") == "1")
        {
            InspectSelfAttentionMap(prefix, ws.Q, ws.K, seqLen);
        }

        WanAttention.TiledMultiHeadAttention(ws.Q, ws.K, ws.V, ws.AttnOut.AsSpan(0, seqLen * _dim), seqLen, seqLen, _numHeads, _headDim);
        Linear($"{prefix}.o", ws.AttnOut, ws.CrossAttnOut.AsSpan(0, seqLen * _dim), _dim, _dim);
    }

    private void CrossAttention(int layerIdx, string prefix, float[] x, WanWorkspace ws, int seqLen, int ctxLen)
    {
        Linear($"{prefix}.q", x, ws.CrossQ.AsSpan(0, seqLen * _dim), _dim, _dim);

        var normQ = TryGetWeight($"{prefix}.norm_q.weight");
        if (normQ is not null) RmsNormHeads(ws.CrossQ, seqLen, _numHeads, _headDim, normQ);

        var (cachedK, cachedV) = ws.CrossKvCache[layerIdx];
        // 2026-09-14 fix (docs/081): derive the real context length from the precomputed K cache's
        // own actual size rather than trusting the separately-threaded `ctxLen` parameter (computed
        // once, early, in Forward() from the ORIGINAL textContext length) -- found via a real
        // cross-reference bisection against the C++ reference where a debug-only context-override
        // path updated PrecomputeCrossKvCache's own local token count without updating Forward()'s
        // separate copy, silently mismatching `ctxLen` against the cache's true size and causing
        // TiledMultiHeadAttention to attend over the wrong number of context tokens. Matches the GPU
        // path's own already-correct pattern (`WanModel.cs` GPU TransformerBlock: `int ctxLen =
        // (int)cachedK.Shape.Dims[0]`), which was never vulnerable to this because it derives ctxLen
        // from the cache directly instead of accepting it as a caller-tracked parameter.
        ctxLen = cachedK.Length / _dim;

        bool debugCross0 = layerIdx == 0 && Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF") == "1";
        string debugCrossDir = Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_CROSSREF_DIR") ?? "examples/stable-diffusion.cpp";
        if (debugCross0)
        {
            WriteCrossRefBin(debugCrossDir, "wandbg_block0_crossq", ws.CrossQ.AsSpan(0, seqLen * _dim).ToArray());
            WriteCrossRefBin(debugCrossDir, "wandbg_block0_crossk", cachedK);
            WriteCrossRefBin(debugCrossDir, "wandbg_block0_crossv", cachedV);
        }

        WanAttention.TiledMultiHeadAttention(ws.CrossQ, cachedK, cachedV, ws.CrossAttnOut.AsSpan(0, seqLen * _dim), seqLen, ctxLen, _numHeads, _headDim);
        if (debugCross0) WriteCrossRefBin(debugCrossDir, "wandbg_block0_crossattn_raw", ws.CrossAttnOut.AsSpan(0, seqLen * _dim).ToArray());
        Linear($"{prefix}.o", ws.CrossAttnOut, ws.AttnOut.AsSpan(0, seqLen * _dim), _dim, _dim);
    }

    private void InspectSelfAttentionMap(string prefix, float[] qArr, float[] kArr, int seqLen)
    {
        // Only inspect representative blocks (0, 15, 29) to keep output concise and fast
        if (prefix != "blocks.0.self_attn" && prefix != "blocks.15.self_attn" && prefix != "blocks.29.self_attn")
            return;

        int gridW = (int)Math.Round(Math.Sqrt(seqLen));
        int gridH = seqLen / gridW;
        if (gridW * gridH != seqLen)
        {
            gridW = 16;
            gridH = Math.Max(1, seqLen / 16);
        }

        var queryTokens = new (string Name, int Y, int X)[]
        {
            ("Corner (0,0)", 0, 0),
            ("Edge (0,W/2)", 0, gridW / 2),
            ("Center (H/2,W/2)", gridH / 2, gridW / 2),
            ("Corner (H-1,W-1)", gridH - 1, gridW - 1)
        };

        float scale = 1.0f / MathF.Sqrt(_headDim);
        float uniformWeight = 1.0f / seqLen;
        int h = 0; // head 0

        Console.WriteLine($"\n==========================================================================");
        Console.WriteLine($"[WanAttentionMap] {prefix} (Head {h}, seqLen={seqLen}, grid={gridH}x{gridW}, uniform={uniformWeight * 100:F3}%)");
        Console.WriteLine($"==========================================================================");

        var scores = new float[seqLen];
        var weights = new float[seqLen];

        foreach (var (name, qy, qx) in queryTokens)
        {
            int qIdx = qy * gridW + qx;
            if (qIdx >= seqLen) continue;

            int qOff = (qIdx * _numHeads + h) * _headDim;

            // 1. Raw Dot Products & Logits
            float maxScore = float.NegativeInfinity;
            float minScore = float.PositiveInfinity;
            for (int k = 0; k < seqLen; k++)
            {
                int kOff = (k * _numHeads + h) * _headDim;
                float dot = 0f;
                for (int d = 0; d < _headDim; d++)
                {
                    dot += qArr[qOff + d] * kArr[kOff + d];
                }
                float s = dot * scale;
                scores[k] = s;
                if (s > maxScore) maxScore = s;
                if (s < minScore) minScore = s;
            }

            // 2. Softmax
            float sumExp = 0f;
            for (int k = 0; k < seqLen; k++)
            {
                weights[k] = MathF.Exp(scores[k] - maxScore);
                sumExp += weights[k];
            }
            float invSum = 1.0f / sumExp;
            for (int k = 0; k < seqLen; k++)
            {
                weights[k] *= invSum;
            }

            // 3. Statistics
            float maxW = 0f;
            int maxK = 0;
            int count2x = 0;
            int count5x = 0;
            float entropy = 0f;

            for (int k = 0; k < seqLen; k++)
            {
                float w = weights[k];
                if (w > maxW) { maxW = w; maxK = k; }
                if (w > 2.0f * uniformWeight) count2x++;
                if (w > 5.0f * uniformWeight) count5x++;
                if (w > 1e-12f) entropy -= w * MathF.Log(w);
            }

            // Spatially nearest (Manhattan dist == 1)
            float nearWeightSum = 0f;
            int nearCount = 0;
            // Spatially far (Manhattan dist >= maxDist - 1)
            int maxDist = (gridH - 1) + (gridW - 1);
            float farWeightSum = 0f;
            int farCount = 0;

            for (int k = 0; k < seqLen; k++)
            {
                int ky = k / gridW;
                int kx = k % gridW;
                int dist = Math.Abs(ky - qy) + Math.Abs(kx - qx);
                if (dist == 1)
                {
                    nearWeightSum += weights[k];
                    nearCount++;
                }
                else if (dist >= maxDist - 1)
                {
                    farWeightSum += weights[k];
                    farCount++;
                }
            }

            float selfWeight = weights[qIdx];
            float nearAvg = nearCount > 0 ? (nearWeightSum / nearCount) : 0f;
            float farAvg = farCount > 0 ? (farWeightSum / farCount) : 0f;
            int oppY = gridH - 1 - qy;
            int oppX = gridW - 1 - qx;
            int oppIdx = Math.Min(seqLen - 1, oppY * gridW + oppX);
            float oppCornerWeight = weights[oppIdx];

            int maxKy = maxK / gridW;
            int maxKx = maxK % gridW;

            Console.WriteLine($"--- Query: {name} (idx={qIdx}, pos=[{qy},{qx}]) ---");
            Console.WriteLine($"  Logit Range: [{minScore:F3} .. {maxScore:F3}] (span={maxScore - minScore:F3})");
            Console.WriteLine($"  Max Weight:  {maxW * 100:F2}% (token {maxK} at [{maxKy},{maxKx}], dist={Math.Abs(maxKy - qy) + Math.Abs(maxKx - qx)})");
            Console.WriteLine($"  Self Weight: {selfWeight * 100:F2}% ({selfWeight / uniformWeight:F2}x uniform)");
            Console.WriteLine($"  Concentrated Tokens: >2x uniform: {count2x}/{seqLen} ({(float)count2x / seqLen * 100:F1}%), >5x: {count5x}/{seqLen}");
            Console.WriteLine($"  Near Neighbors (dist=1, n={nearCount}): avg={nearAvg * 100:F3}% ({nearAvg / uniformWeight:F2}x uniform), total={nearWeightSum * 100:F2}%");
            Console.WriteLine($"  Far Tokens (dist>={maxDist - 1}, n={farCount}): avg={farAvg * 100:F3}% ({farAvg / uniformWeight:F2}x uniform)");
            Console.WriteLine($"  Opposite Corner [{oppY},{oppX}]: {oppCornerWeight * 100:F3}% ({oppCornerWeight / uniformWeight:F2}x uniform)");
            Console.WriteLine($"  Near/Far Ratio: {(farAvg > 1e-8f ? (nearAvg / farAvg).ToString("F2") : "inf")}x");
            Console.WriteLine($"  Entropy: {entropy:F3} nats (uniform = {MathF.Log(seqLen):F3} nats)");
        }
        Console.WriteLine($"==========================================================================\n");
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

    private void FeedForward(string prefix, float[] x, WanWorkspace ws, int seqLen, string? debugDumpDir = null)
    {
        Linear($"{prefix}.0", x, ws.Ffn1.AsSpan(0, seqLen * _ffnDim), _dim, _ffnDim);
        if (debugDumpDir is not null) WriteCrossRefBin(debugDumpDir, "wandbg_block0_ffn0", ws.Ffn1.AsSpan(0, seqLen * _ffnDim).ToArray());
        DiffusionOps.GeluInPlace(ws.Ffn1.AsSpan(0, seqLen * _ffnDim));
        if (debugDumpDir is not null) WriteCrossRefBin(debugDumpDir, "wandbg_block0_gelu", ws.Ffn1.AsSpan(0, seqLen * _ffnDim).ToArray());
        Linear($"{prefix}.2", ws.Ffn1, ws.FfnOut.AsSpan(0, seqLen * _dim), _ffnDim, _dim);
        if (debugDumpDir is not null) WriteCrossRefBin(debugDumpDir, "wandbg_block0_ffn2", ws.FfnOut.AsSpan(0, seqLen * _dim).ToArray());
    }

    private float[] Modulate(float[] x, int seqLen, ReadOnlySpan<float> shift, ReadOnlySpan<float> scale)
        => DiffusionOps.ModulateRows(x, seqLen, _dim, shift, scale);

    private void ApplyGatedResidual(float[] x, float[] branch, int seqLen, ReadOnlySpan<float> gate)
        => DiffusionOps.ApplyGatedResidualRows(x, branch, seqLen, _dim, gate);

    private float[] ComputeTimestepEmbedding(float timestep)
    {
        // flipSinToCos: true -- the real reference (ggml_compute_forward_timestep_embedding_f32,
        // examples/stable-diffusion.cpp/ggml/src/ggml-cpu/ops.cpp:8302-8303) writes
        // embed_data[j]=cos(arg) for the FIRST half and embed_data[j+half]=sin(arg) for the
        // second half -- i.e. [cos, sin] order. The default (flipSinToCos: false) produces the
        // opposite [sin, cos] order, which silently swaps which half of the input `time_embedding.0`
        // linear layer's trained weights sees cos vs sin -- found 2026-09-14 during the Priority-0
        // Wan accuracy investigation (docs/081): the "half-sinusoid" columns each end up multiplied
        // by the wrong learned coefficient (same class of bug as the PackLatents column-permutation
        // fix earlier in this file), corrupting the model's actual sense of the current noise level.
        var emb = DiffusionOps.SinusoidalTimestepEmbedding(timestep, flipSinToCos: true);
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
        // 2026-09-14: two "reference" sources disagree here -- diffusers' transformer_wan.py uses
        // `PixArtAlphaTextProjection(..., act_fn="gelu_tanh")` (tanh-approx, the default below),
        // but wan.hpp's own comment says plain `nn.GELU()` (exact, no args = exact in PyTorch).
        // STINGRAY_WAN_DEBUG_EXACT_GELU=1 switches to the exact/erf-based variant to test which
        // one is real empirically rather than trusting either doc comment blindly (docs/081).
        bool exactGelu = Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_EXACT_GELU") == "1";
        for (int i = 0; i < t0.Length; i++)
            t0[i] = exactGelu ? DiffusionOps.GeluExact(t0[i]) : DiffusionOps.Gelu(t0[i]);
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
                    // patch_embedding is a real Conv3d(in_dim=16, kernel=patch_size=(1,2,2)) in the
                    // reference (wan.hpp:537), NOT a hand-rolled pixel-major pack -- reinterpreting
                    // its [out_dim, in_dim, kt, kh, kw] weight as a flat Linear matrix means the
                    // packed input vector's element order MUST match the weight's own trailing-dim
                    // flatten order: in_channel OUTER, spatial (dy, dx) INNER (slot = c*4 + dy*2 + dx).
                    // The previous (dy*2+dx)*OutChannels+c ordering was a column permutation of the
                    // patch-embedding matrix relative to what the checkpoint was trained with --
                    // self-consistent with UnpackLatents but wrong against the real weight layout,
                    // which explains plausible-magnitude-but-fully-scrambled DiT output (see
                    // docs/081, 2026-09-14 update: VAE isolation test proved the VAE decodes
                    // structured input faithfully, ruling it out and pointing back at this packing).
                    for (int dy = 0; dy < 2; dy++)
                    {
                        for (int dx = 0; dx < 2; dx++)
                        {
                            for (int c = 0; c < OutChannels; c++)
                            {
                                int y = ph * 2 + dy;
                                int x = pw * 2 + dx;
                                int srcIdx = ((c * numFrames + f) * latH + y) * latW + x;
                                int slot = c * 4 + dy * 2 + dx;
                                packed[tokenOff + slot] = latents[srcIdx];
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

                    // 2026-09-14 correction: head.head is a plain Linear (NOT a Conv3d like
                    // patch_embedding), and the reference's own `unpatchify` (wan.hpp:610-625)
                    // reshapes its 64-length output as ggml `(C, pw*ph*pt)` -- ggml's ne[0] is the
                    // FASTEST/most-contiguous axis (per this repo's own AGENTS.md tensor-layout
                    // note), so C is the INNERMOST component there, spatial offset (dy,dx) OUTER.
                    // This is the OPPOSITE convention from patch_embedding's real Conv3d weight
                    // (whose raw PyTorch storage makes in_channel the outer, slower-varying axis
                    // of its flattened kernel taps) -- an initial fix here wrongly assumed
                    // PackLatents and UnpackLatents must share one convention; they don't, because
                    // the encode (Conv3d) and decode (Linear + separately-coded unpatchify) sides
                    // are independent design choices in the original model. Keep PackLatents at
                    // channel-outer (c*4+dy*2+dx) but UnpackLatents at spatial-outer
                    // ((dy*2+dx)*OutChannels+c), matching each side's own real reference.
                    for (int c = 0; c < OutChannels; c++)
                    {
                        for (int dy = 0; dy < 2; dy++)
                        {
                            for (int dx = 0; dx < 2; dx++)
                            {
                                int y = ph * 2 + dy;
                                int x = pw * 2 + dx;
                                int offset = (dy * 2 + dx) * OutChannels + c;
                                int dstIdx = ((c * numFrames + f) * latH + y) * latW + x;
                                unpacked[dstIdx] = packed[tokenOff + offset];
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
