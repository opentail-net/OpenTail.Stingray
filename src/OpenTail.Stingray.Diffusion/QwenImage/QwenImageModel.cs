using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.QwenImage;

/// <summary>
/// Native C# Qwen Image 60-layer MM-DiT diffusion transformer model.
/// Supports standard Text-to-Image and Qwen Image Edit (reference visual conditioning).
/// Reference: stable-diffusion.cpp:src/model/diffusion/qwen_image.hpp:QwenImageModel
/// </summary>
public sealed class QwenImageModel : IDisposable
{
    private readonly IWeightLoader _weights;
    private readonly string _prefix;
    private readonly IComputeBackend? _backend;
    private readonly Dictionary<string, CoreTensor>? _gpuWeights;
    private readonly int _numLayers;
    private readonly QuantizedWeightCache _cache;
    private bool _gpuDisabled;
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, float[]?> _biasCache = new(StringComparer.Ordinal);
    private bool _disposed;
    private QwenImageGpuWeights? _residentGpuWeights;
    private QwenImageGpuWorkspace? _residentGpuWorkspace;

    // Per-block diagnostic hooks (docs/094 Phase 2's GPU-bug bisection, 2026-09-20), same
    // convention already proven in WanModel (OnBlockOutputCpu/OnBlockOutputGpu) -- lets a real test
    // dump the [img] hidden state after each of the 60 blocks on both paths and diff them to find
    // the FIRST divergence point, rather than guessing at op-level candidates (8 already ruled out
    // that way for this exact bug, see docs/094).
    public Action<int, float[]>? OnBlockOutputCpu { get; set; }
    public Action<int, float[]>? OnBlockOutputGpu { get; set; }

    /// <summary>Same event as <see cref="OnBlockOutputCpu"/> but also exposes the [txt] stream's
    /// post-block state (docs/094 Phase 2 follow-up, 2026-09-20) -- needed to freeze a real
    /// mid-trajectory (img, txt) pair as the common input for a single-block cross-injection
    /// experiment (<see cref="RunSingleBlockGpuForTest"/>/<see cref="RunSingleBlockCpuForTest"/>),
    /// since Qwen Image's joint attention consumes both streams and the older img-only hook can't
    /// reproduce a real block's actual input.</summary>
    public Action<int, float[], float[]>? OnBlockStateCpu { get; set; }

    // RoPE cache (docs/094 FLUX.2 GPU optimization wave's cross-model caching survey, 2026-09-20):
    // Compute3DRoPE depends only on (numTxtTokens, patchH, patchW, HeadDim) -- fixed for the whole
    // generation -- but both Forward() and ForwardGpu() recomputed it via fresh trig evaluation on
    // every single denoising step. Same bug class already found and fixed in Flux2DiT/WanModel/
    // HunyuanVideoModel this same pass. Shared by both CPU and GPU paths since the math (and the
    // key it depends on) is identical either way.
    private readonly object _ropeCacheLock = new();
    private (int numTxtTokens, int patchH, int patchW, int headDim)? _cachedRopeKey;
    private (float[] cos, float[] sin)? _cachedRope;

    private (float[] cos, float[] sin) GetOrComputeRope(int numTxtTokens, int patchH, int patchW)
    {
        var key = (numTxtTokens, patchH, patchW, HeadDim);
        lock (_ropeCacheLock)
        {
            if (_cachedRopeKey == key && _cachedRope is not null) return _cachedRope.Value;
            var rope = QwenImageRoPE.Compute3DRoPE(numTxtTokens, patchH, patchW, HeadDim);
            _cachedRopeKey = key;
            _cachedRope = rope;
            return rope;
        }
    }

    public const int HiddenDim = 3072;
    public const int NumHeads = 24;
    public const int HeadDim = 128;
    public const int PatchSize = 2;
    public const int InChannels = 64;   // 16 * 2 * 2
    public const int OutChannels = 16;
    public const int ContextDim = 3584; // Qwen2.5-VL text dimension

    public int NumLayers => _numLayers;

    public QwenImageModel(IWeightLoader weights, string prefix = "", int numLayers = 60, IComputeBackend? backend = null)
    {
        _weights = weights;
        _prefix = prefix;
        _backend = backend;
        _cache = new QuantizedWeightCache(weights, "");
        _numLayers = DetectNumLayers(weights, prefix, numLayers);
        if (backend is not null)
            _gpuWeights = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
    }

    private static int DetectNumLayers(IWeightLoader weights, string prefix, int fallback)
    {
        for (int i = 80; i >= 0; i--)
        {
            string key = $"{prefix}transformer_blocks.{i}.attn.to_q.weight";
            if (weights.Contains(key) || weights.Contains("model.diffusion_model." + key))
                return i + 1;
        }
        return fallback;
    }

    private string Resolve(string name)
    {
        string direct = _prefix + name;
        if (_weights.Contains(direct)) return direct;
        if (_weights.Contains("model.diffusion_model." + direct)) return "model.diffusion_model." + direct;
        if (_weights.Contains("transformer." + direct)) return "transformer." + direct;
        return direct;
    }

    // NOT cached, deliberately -- a single 60-layer Forward() pass touches every distinct
    // per-block tensor exactly once (no reuse WITHIN one pass), so a permanent Dictionary cache
    // gives zero benefit inside a single call and only pays off across the ~steps calls in one
    // generation, at the cost of holding every dequantized layer in memory simultaneously forever.
    // At Q3_K_S (real checkpoint) that's roughly a 9x dequant-to-float32 expansion across 60
    // layers -- measured growing linearly to 53GB+ and still climbing (heading toward ~75GB) on
    // this 64GB machine before being killed for OOM, real run, 2026-09-18 (see docs/086). Re-reading
    // fresh per call costs at most `steps` extra dequant passes over the same 8.3GB compressed
    // checkpoint -- cheap relative to risking an OOM kill. Small per-head norm weights ([128]/
    // [3072]) are negligible either way; this only matters for the large per-layer matmul weights.
    private float[] GetWeight(string name)
    {
        string fullName = Resolve(name);
        return _weights.ReadF32(fullName);
    }

    private float[]? TryGetWeight(string name)
    {
        string fullName = Resolve(name);
        if (_weights.Contains(fullName))
        {
            return _weights.ReadF32(fullName);
        }
        return null;
    }

    private void EnsureGpuResident(int numImgTokens, int numTxtTokens, float[] ropeCos, float[] ropeSin)
    {
        _residentGpuWeights ??= new QwenImageGpuWeights(_backend!, GetWeight, TryGetWeight, _numLayers, HiddenDim, InChannels, ContextDim, HeadDim);

        if (_residentGpuWorkspace is null
            || _residentGpuWorkspace.NumImgTokens != numImgTokens
            || _residentGpuWorkspace.NumTxtTokens != numTxtTokens)
        {
            _residentGpuWorkspace?.Dispose();
            _residentGpuWorkspace = new QwenImageGpuWorkspace(_backend!, numImgTokens, numTxtTokens, HiddenDim, InChannels, ropeCos, ropeSin, HeadDim);
        }
    }

    /// <summary>
    /// GPU-resident forward pass (docs/094 Phase 2). Standard text-to-image only (no Edit
    /// reference-latent conditioning -- see <see cref="Forward"/>'s dispatch guard). Mirrors the CPU
    /// <see cref="Forward"/>/<see cref="TransformerBlock"/>/<see cref="JointAttention"/> structure
    /// block-for-block; every op maps to an existing Vulkan primitive already proven correct by
    /// FLUX.1/FLUX.2/SD3.5's own GPU ports (<see cref="QwenImageGpuWeights"/>'s doc comment has the
    /// full architectural-difference rationale for why each substitution is safe).
    /// </summary>
    private float[] ForwardGpu(
        IVisionOpsBackend visionOps,
        IImageOpsBackend imageOps,
        float[] latent, float timestep, float[] textContext, int latH, int latW)
    {
        int patchH = latH / PatchSize;
        int patchW = latW / PatchSize;
        int numImgTokens = patchH * patchW;
        int numTxtTokens = textContext.Length / ContextDim;
        int totalTokens = numTxtTokens + numImgTokens;

        var (ropeCos, ropeSin) = GetOrComputeRope(numTxtTokens, patchH, patchW);
        EnsureGpuResident(numImgTokens, numTxtTokens, ropeCos, ropeSin);
        var gw = _residentGpuWeights!;
        var ws = _residentGpuWorkspace!;

        var packedInput = PackLatents(latent, latH, latW);

        // Input projections (host-side patchify, then a real GEMM upload each -- same pattern
        // MMDiTModel.ForwardGpu uses for its own x_embedder).
        var imgTokensF32 = Linear("img_in", packedInput, InChannels, HiddenDim);
        var txtTokensF32 = Linear("txt_in", textContext, ContextDim, HiddenDim);
        var xGpu = _backend!.Upload(imgTokensF32.AsSpan(0, numImgTokens * HiddenDim), TensorShape.D2(numImgTokens, HiddenDim), exact: true);
        var cGpu = _backend.Upload(txtTokensF32.AsSpan(0, numTxtTokens * HiddenDim), TensorShape.D2(numTxtTokens, HiddenDim), exact: true);

        // Timestep embedding: tiny (256->3072->3072), kept on the existing CPU path (same choice
        // MMDiTModel.ForwardGpu makes for its own time+pooled embedding).
        var tEmb = ComputeTimestepEmbedding(timestep);
        var tVecSilu = DiffusionOpsSilu(tEmb);
        var tVecGpu = _backend.Upload(tVecSilu.AsSpan(0, HiddenDim), TensorShape.D1(HiddenDim), exact: true);

        bool batchSuccess = false;
        try
        {
            // Per-block batching (docs/094 Phase 2 debugging pass, 2026-09-19), NOT one single
            // 60-layer batch: real precedent in this codebase for a single giant batch being a
            // genuine bug/regression source (FLUX.2's Stage 5 finding -- see
            // docs/093-flux2-gpu-performance-optimization-plan.md). Being investigated as a
            // candidate for this port's own "runs fine, wrong texture" defect.
            for (int b = 0; b < _numLayers; b++)
            {
                imageOps.BeginBatch();
                RunBlockGpu(visionOps, imageOps, ws, gw.Blocks[b], xGpu, cGpu, tVecGpu, numImgTokens, numTxtTokens, totalTokens);
                imageOps.EndBatch();

                if (OnBlockOutputGpu != null)
                {
                    var imgHost = new float[numImgTokens * HiddenDim];
                    _backend.Download(xGpu, imgHost);
                    OnBlockOutputGpu(b, imgHost);
                }
            }
            batchSuccess = true;

            // Final layer: norm_out.linear (AdaLN shift/scale) + proj_out.
            imageOps.BeginBatch();
            visionOps.Sgemm(ws.FinalMod, tVecGpu, gw.FinalNormLinearWeight, 1, HiddenDim, 2 * HiddenDim);
            if (gw.FinalNormLinearBias is { } fnb) imageOps.AddRowBroadcastInPlace(ws.FinalMod, fnb, 1, 2 * HiddenDim);

            visionOps.AdaLNModulate(ws.NormedImg1, xGpu, ws.FinalMod, numImgTokens, HiddenDim, shiftOffset: HiddenDim, scaleOffset: 0, isRmsNorm: false, eps: 1e-6f);

            visionOps.Sgemm(ws.Unpatchified, ws.NormedImg1, gw.FinalProjOutWeight, numImgTokens, HiddenDim, InChannels);
            if (gw.FinalProjOutBias is { } fpb) imageOps.AddRowBroadcastInPlace(ws.Unpatchified, fpb, numImgTokens, InChannels);
            imageOps.EndBatch();

            var outPacked = new float[numImgTokens * InChannels];
            _backend.Download(ws.Unpatchified, outPacked);
            return UnpackLatents(outPacked, latH, latW);
        }
        finally
        {
            if (!batchSuccess)
            {
                try { imageOps.EndBatch(); } catch { }
            }
            _backend.Free(tVecGpu);
            _backend.Free(xGpu);
            _backend.Free(cGpu);
        }
    }

    /// <summary>
    /// One MM-DiT block's GPU compute, extracted verbatim from <see cref="ForwardGpu"/>'s per-block
    /// loop body (docs/094 Phase 2 follow-up, 2026-09-20) so the production 60-block loop and a
    /// single-block isolation test (<see cref="RunSingleBlockGpuForTest"/>) share the exact same
    /// code path -- no risk of the two drifting apart. Mutates <paramref name="xGpu"/>/
    /// <paramref name="cGpu"/> in place (the real residual streams), matching the loop's own
    /// behavior. Caller owns BeginBatch/EndBatch.
    /// </summary>
    private void RunBlockGpu(
        IVisionOpsBackend visionOps, IImageOpsBackend imageOps,
        QwenImageGpuWorkspace ws, QwenImageGpuWeights.BlockGpuWeights bw,
        CoreTensor xGpu, CoreTensor cGpu, CoreTensor tVecGpu,
        int numImgTokens, int numTxtTokens, int totalTokens)
    {
        visionOps.Sgemm(ws.ImgMod, tVecGpu, bw.ImgModWeight, 1, HiddenDim, 6 * HiddenDim);
        if (bw.ImgModBias is { } imb) imageOps.AddRowBroadcastInPlace(ws.ImgMod, imb, 1, 6 * HiddenDim);
        visionOps.Sgemm(ws.TxtMod, tVecGpu, bw.TxtModWeight, 1, HiddenDim, 6 * HiddenDim);
        if (bw.TxtModBias is { } tmb) imageOps.AddRowBroadcastInPlace(ws.TxtMod, tmb, 1, 6 * HiddenDim);

        // Affine-free LayerNorm + modulate (isRmsNorm: false -- see QwenImageModel's own
        // CPU doc comment on LayerNormNoAffine; AdaLNModulate with isRmsNorm:false applies
        // the equivalent plain-LayerNorm-then-(1+scale)*x+shift used elsewhere for SD3.5).
        visionOps.AdaLNModulate(ws.NormedImg1, xGpu, ws.ImgMod, numImgTokens, HiddenDim, shiftOffset: 0, scaleOffset: HiddenDim, isRmsNorm: false, eps: 1e-6f);
        visionOps.AdaLNModulate(ws.NormedTxt1, cGpu, ws.TxtMod, numTxtTokens, HiddenDim, shiftOffset: 0, scaleOffset: HiddenDim, isRmsNorm: false, eps: 1e-6f);

        // Separate (non-fused) Q/K/V projections.
        visionOps.Sgemm(ws.ImgQ, ws.NormedImg1, bw.ImgToQWeight, numImgTokens, HiddenDim, HiddenDim);
        visionOps.Sgemm(ws.ImgK, ws.NormedImg1, bw.ImgToKWeight, numImgTokens, HiddenDim, HiddenDim);
        visionOps.Sgemm(ws.ImgV, ws.NormedImg1, bw.ImgToVWeight, numImgTokens, HiddenDim, HiddenDim);
        visionOps.Sgemm(ws.TxtQ, ws.NormedTxt1, bw.TxtAddQWeight, numTxtTokens, HiddenDim, HiddenDim);
        visionOps.Sgemm(ws.TxtK, ws.NormedTxt1, bw.TxtAddKWeight, numTxtTokens, HiddenDim, HiddenDim);
        visionOps.Sgemm(ws.TxtV, ws.NormedTxt1, bw.TxtAddVWeight, numTxtTokens, HiddenDim, HiddenDim);

        // Per-stream QK-RMSNorm (different learned scales for img vs. txt).
        visionOps.QKNorm(ws.ImgQ, ws.ImgK, bw.ImgNormQScale, bw.ImgNormKScale, numImgTokens, NumHeads, HeadDim, eps: 1e-6f);
        visionOps.QKNorm(ws.TxtQ, ws.TxtK, bw.TxtNormQScale, bw.TxtNormKScale, numTxtTokens, NumHeads, HeadDim, eps: 1e-6f);

        // Concatenate [txt; img] (Qwen Image's own ordering -- opposite of FLUX/SD3.5's
        // [img; txt], confirmed against the CPU path's ConcatSequences call order).
        visionOps.FluxConcatTxtImg(ws.TxtQ, ws.ImgQ, ws.Q, numTxtTokens, numImgTokens, HiddenDim);
        visionOps.FluxConcatTxtImg(ws.TxtK, ws.ImgK, ws.K, numTxtTokens, numImgTokens, HiddenDim);
        visionOps.FluxConcatTxtImg(ws.TxtV, ws.ImgV, ws.V, numTxtTokens, numImgTokens, HiddenDim);

        // 3D-RoPE over the full joint sequence at once (matches the CPU path's single
        // ApplyRoPE call over the concatenated q/k arrays).
        visionOps.Flux2DRoPE(ws.Q, ws.K, ws.RopeCos, ws.RopeSin, startToken: 0, tokenCount: totalTokens, NumHeads, HeadDim);

        imageOps.MultiHeadAttentionTiled(ws.AttnOut, ws.Q, ws.K, ws.V, totalTokens, totalTokens, NumHeads, HeadDim);

        // Slice the joint attention output back into per-stream buffers ([txt; img] order:
        // txt occupies rows [0, numTxt), img occupies [numTxt, numTxt+numImg)).
        visionOps.FluxSliceImg(ws.AttnOut, ws.TxtAttnOut, nTxt: 0, nImg: numTxtTokens, dim: HiddenDim);
        visionOps.FluxSliceImg(ws.AttnOut, ws.ImgAttnOut, nTxt: numTxtTokens, nImg: numImgTokens, dim: HiddenDim);

        visionOps.Sgemm(ws.ImgOut, ws.ImgAttnOut, bw.ImgToOutWeight, numImgTokens, HiddenDim, HiddenDim);
        if (bw.ImgToOutBias is { } iob) imageOps.AddRowBroadcastInPlace(ws.ImgOut, iob, numImgTokens, HiddenDim);
        visionOps.ScaleGateAdd(xGpu, ws.ImgOut, ws.ImgMod, numImgTokens, HiddenDim, gateOffset: 2 * HiddenDim);

        visionOps.Sgemm(ws.TxtOut, ws.TxtAttnOut, bw.TxtToAddOutWeight, numTxtTokens, HiddenDim, HiddenDim);
        if (bw.TxtToAddOutBias is { } tob) imageOps.AddRowBroadcastInPlace(ws.TxtOut, tob, numTxtTokens, HiddenDim);
        visionOps.ScaleGateAdd(cGpu, ws.TxtOut, ws.TxtMod, numTxtTokens, HiddenDim, gateOffset: 2 * HiddenDim);

        // Norm2 + plain-GELU FFN (NOT gated -- see QwenImageGpuWeights' doc comment).
        visionOps.AdaLNModulate(ws.NormedImg2, xGpu, ws.ImgMod, numImgTokens, HiddenDim, shiftOffset: 3 * HiddenDim, scaleOffset: 4 * HiddenDim, isRmsNorm: false, eps: 1e-6f);
        visionOps.AdaLNModulate(ws.NormedTxt2, cGpu, ws.TxtMod, numTxtTokens, HiddenDim, shiftOffset: 3 * HiddenDim, scaleOffset: 4 * HiddenDim, isRmsNorm: false, eps: 1e-6f);

        int ffDim = HiddenDim * 4;
        visionOps.Sgemm(ws.ImgMlpBuf, ws.NormedImg2, bw.ImgMlpUpWeight, numImgTokens, HiddenDim, ffDim);
        if (bw.ImgMlpUpBias is { } imub) imageOps.AddRowBroadcastInPlace(ws.ImgMlpBuf, imub, numImgTokens, ffDim);
        visionOps.VisionGeluInPlace(ws.ImgMlpBuf);
        visionOps.Sgemm(ws.ImgOut, ws.ImgMlpBuf, bw.ImgMlpDownWeight, numImgTokens, ffDim, HiddenDim);
        if (bw.ImgMlpDownBias is { } imdb) imageOps.AddRowBroadcastInPlace(ws.ImgOut, imdb, numImgTokens, HiddenDim);
        visionOps.ScaleGateAdd(xGpu, ws.ImgOut, ws.ImgMod, numImgTokens, HiddenDim, gateOffset: 5 * HiddenDim);

        visionOps.Sgemm(ws.TxtMlpBuf, ws.NormedTxt2, bw.TxtMlpUpWeight, numTxtTokens, HiddenDim, ffDim);
        if (bw.TxtMlpUpBias is { } tmub) imageOps.AddRowBroadcastInPlace(ws.TxtMlpBuf, tmub, numTxtTokens, ffDim);
        visionOps.VisionGeluInPlace(ws.TxtMlpBuf);
        visionOps.Sgemm(ws.TxtOut, ws.TxtMlpBuf, bw.TxtMlpDownWeight, numTxtTokens, ffDim, HiddenDim);
        if (bw.TxtMlpDownBias is { } tmdb) imageOps.AddRowBroadcastInPlace(ws.TxtOut, tmdb, numTxtTokens, HiddenDim);
        visionOps.ScaleGateAdd(cGpu, ws.TxtOut, ws.TxtMod, numTxtTokens, HiddenDim, gateOffset: 5 * HiddenDim);
    }

    /// <summary>
    /// Test-only single-block GPU isolation entry point (docs/094 Phase 2 follow-up, 2026-09-20):
    /// runs exactly ONE block (via the shared <see cref="RunBlockGpu"/>) against an arbitrary
    /// injected (img, txt) state -- e.g. a real CPU trajectory's state entering block 29, captured
    /// via <see cref="OnBlockStateCpu"/> -- rather than the block's own accumulated GPU trajectory.
    /// This is the "common-input" cross-injection technique: it isolates ONE block's own numerical
    /// behavior from 28+ blocks of upstream accumulated drift.
    ///
    /// <para><b>Deliberately does NOT call <see cref="EnsureGpuResident"/></b>: that method uploads
    /// ALL 60 blocks' weights (~41GB at FP16 for this ~20.8B-param checkpoint, observed to push this
    /// machine's RAM usage to near-total during a real end-to-end run) even though this test only
    /// ever touches one block. Instead builds a fresh, standalone
    /// <see cref="QwenImageGpuWorkspace"/> and exactly ONE <see cref="QwenImageGpuWeights.BlockGpuWeights"/>
    /// per call, both disposed immediately after -- bounding this test's real footprint to roughly
    /// 680MB (FP16) or 1.36GB (<paramref name="forceFp32ForThisBlock"/>: true), not the whole
    /// model's resident set.</para>
    /// </summary>
    public (float[] imgOut, float[] txtOut) RunSingleBlockGpuForTest(
        int blockIndex, float[] imgIn, float[] txtIn, float timestep,
        int numImgTokens, int numTxtTokens, int patchH, int patchW,
        bool forceFp32ForThisBlock = false)
    {
        if (_backend is not IVisionOpsBackend visionOps || _backend is not IImageOpsBackend imageOps)
            throw new InvalidOperationException("RunSingleBlockGpuForTest requires a GPU backend implementing IVisionOpsBackend/IImageOpsBackend.");

        var (ropeCos, ropeSin) = GetOrComputeRope(numTxtTokens, patchH, patchW);
        using var ws = new QwenImageGpuWorkspace(_backend!, numImgTokens, numTxtTokens, HiddenDim, InChannels, ropeCos, ropeSin, HeadDim);
        using var bw = new QwenImageGpuWeights.BlockGpuWeights(_backend!, GetWeight, TryGetWeight, blockIndex, HiddenDim, HeadDim, forceFp32ForThisBlock);

        int totalTokens = numImgTokens + numTxtTokens;
        var tEmb = ComputeTimestepEmbedding(timestep);
        var tVecSilu = DiffusionOpsSilu(tEmb);

        var xGpu = _backend!.Upload(imgIn.AsSpan(0, numImgTokens * HiddenDim), TensorShape.D2(numImgTokens, HiddenDim), exact: true);
        var cGpu = _backend.Upload(txtIn.AsSpan(0, numTxtTokens * HiddenDim), TensorShape.D2(numTxtTokens, HiddenDim), exact: true);
        var tVecGpu = _backend.Upload(tVecSilu.AsSpan(0, HiddenDim), TensorShape.D1(HiddenDim), exact: true);

        try
        {
            imageOps.BeginBatch();
            bool ok = false;
            try
            {
                RunBlockGpu(visionOps, imageOps, ws, bw, xGpu, cGpu, tVecGpu, numImgTokens, numTxtTokens, totalTokens);
                ok = true;
            }
            finally
            {
                if (ok) imageOps.EndBatch();
                else { try { imageOps.EndBatch(); } catch { } }
            }

            var imgOut = new float[numImgTokens * HiddenDim];
            var txtOut = new float[numTxtTokens * HiddenDim];
            _backend.Download(xGpu, imgOut);
            _backend.Download(cGpu, txtOut);
            return (imgOut, txtOut);
        }
        finally
        {
            _backend.Free(xGpu);
            _backend.Free(cGpu);
            _backend.Free(tVecGpu);
        }
    }

    /// <summary>
    /// Test-only single-block CPU isolation entry point, mirroring
    /// <see cref="RunSingleBlockGpuForTest"/> -- runs exactly one <see cref="TransformerBlock"/>
    /// call against an arbitrary injected (img, txt) state. Used as the CPU-side control/reference
    /// for the same cross-injection experiment.
    ///
    /// <para><b>WARNING, real trap found and fixed 2026-09-20 (docs/094 Phase 2 follow-up)</b>:
    /// <see cref="TransformerBlock"/> mutates <paramref name="imgIn"/>/<paramref name="txtIn"/> IN
    /// PLACE (it applies the residual directly onto the passed arrays and returns those SAME
    /// references) -- calling this method turns your input arrays into the block's OUTPUT as a
    /// side effect. If you need to reuse the same input elsewhere afterward (e.g. also feeding it to
    /// <see cref="RunSingleBlockGpuForTest"/>), pass clones here, not the originals — a real bug in
    /// an earlier version of <c>QwenImageBlock29Fp32IsolationTests</c> called this on the same
    /// arrays it later fed to the GPU lanes, silently corrupting them and producing a large, entirely
    /// spurious CPU-vs-GPU divergence that had nothing to do with the GPU at all.</para>
    /// </summary>
    public (float[] imgOut, float[] txtOut) RunSingleBlockCpuForTest(
        int blockIndex, float[] imgIn, float[] txtIn, float timestep, int patchH, int patchW)
    {
        int numImg = imgIn.Length / HiddenDim;
        int numTxt = txtIn.Length / HiddenDim;
        var tEmb = ComputeTimestepEmbedding(timestep);
        var (cos, sin) = GetOrComputeRope(numTxt, patchH, patchW);
        string prefix = $"transformer_blocks.{blockIndex}";
        return TransformerBlock(prefix, imgIn, txtIn, tEmb, cos, sin, numImg, numTxt, modulateIndex: null);
    }

    /// <summary>
    /// Evaluates the Qwen Image transformer forward pass.
    /// Supports optional reference image latents for Qwen Image Edit.
    /// </summary>
    public float[] Forward(
        float[] latent,
        float timestep,
        float[] textContext,
        int latH,
        int latW,
        float[]? refLatent = null)
    {
        // GPU-resident path (docs/094 Phase 2) only covers standard text-to-image (no Edit
        // reference-latent conditioning yet, matching the scope of every other GPU port in this
        // codebase's first pass) -- fall back to CPU when refLatent is supplied.
        if (!_gpuDisabled && _backend is not null && refLatent is null && _backend is IVisionOpsBackend visionOps && _backend is IImageOpsBackend imageOps)
        {
            try
            {
                lock (this)
                {
                    return ForwardGpu(visionOps, imageOps, latent, timestep, textContext, latH, latW);
                }
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                // Fall back to CPU for the rest of this model's lifetime. Retrying every call
                // re-attempted the whole 20B-parameter upload per forward (and on a device that can't
                // hold it -- e.g. a shared-memory iGPU, ErrorOutOfHostMemory -- turned each CPU step
                // from ~34s into ~250s).
                _gpuDisabled = true;
                Console.WriteLine($"[QwenImage] GPU forward failed, using CPU from now on: {ex.GetType().Name}: {ex.Message}");
            }
        }

        int patchH = latH / PatchSize;
        int patchW = latW / PatchSize;
        int numTargetTokens = patchH * patchW;
        int numTxtTokens = textContext.Length / ContextDim;

        // 1. Pack 16-channel latents into 64-channel patches
        var packedTarget = PackLatents(latent, latH, latW);
        float[] packedInput;
        int[]? modulateIndex = null;
        int numImgTokens = numTargetTokens;

        if (refLatent is not null)
        {
            // Qwen Image Edit: concatenate reference visual tokens with target canvas tokens
            var packedRef = PackLatents(refLatent, latH, latW);
            numImgTokens = numTargetTokens + numTargetTokens;
            packedInput = new float[numImgTokens * InChannels];
            Array.Copy(packedTarget, 0, packedInput, 0, packedTarget.Length);
            Array.Copy(packedRef, 0, packedInput, packedTarget.Length, packedRef.Length);

            modulateIndex = new int[numImgTokens];
            for (int i = 0; i < numTargetTokens; i++) modulateIndex[i] = 0; // target: 0
            for (int i = numTargetTokens; i < numImgTokens; i++) modulateIndex[i] = 1; // ref: 1
        }
        else
        {
            packedInput = packedTarget;
        }

        // 2. Input projections
        var imgTokens = Linear("img_in", packedInput, InChannels, HiddenDim);
        var txtTokens = Linear("txt_in", textContext, ContextDim, HiddenDim);

        // 3. Timestep embedding (sinusoidal 256 -> linear 3072 -> silu -> linear 3072)
        var tEmb = ComputeTimestepEmbedding(timestep);

        // 4. 3D-RoPE positional encoding
        var (cos, sin) = GetOrComputeRope(numTxtTokens, patchH, patchW);

        // 5. 60 MM-DiT Transformer blocks
        for (int b = 0; b < _numLayers; b++)
        {
            string p = $"transformer_blocks.{b}";
            (imgTokens, txtTokens) = TransformerBlock(p, imgTokens, txtTokens, tEmb, cos, sin, numImgTokens, numTxtTokens, modulateIndex);
            OnBlockOutputCpu?.Invoke(b, imgTokens);
            OnBlockStateCpu?.Invoke(b, imgTokens, txtTokens);
        }

        // 6. Final layer norm and projection -- real checkpoint key names are `norm_out.linear`
        // (AdaLN shift/scale) and `proj_out` (final projection), NOT `final_layer.*` (a real bug
        // found via KeyNotFoundException on the first real-weight run, 2026-09-18 -- confirmed
        // against the checkpoint's own tensor names, not guessed).
        // Real reference chunks this into [scale, shift] (in that order) and applies
        // Flux::modulate's x*(1+scale)+shift -- the `1+scale` offset is easy to miss since
        // DiffusionOps.LayerNorm's generic weight/bias multiply doesn't add it implicitly (found
        // by direct comparison against AdaLayerNormContinuous::forward, not assumed).
        var finalNorm = Linear("norm_out.linear", DiffusionOpsSilu(tEmb), HiddenDim, HiddenDim * 2);
        var finalScale = finalNorm.AsSpan(0, HiddenDim);
        var finalShift = finalNorm.AsSpan(HiddenDim, HiddenDim);
        var finalGamma = new float[HiddenDim];
        for (int d = 0; d < HiddenDim; d++) finalGamma[d] = 1.0f + finalScale[d];

        // Extract target canvas tokens (first numTargetTokens)
        var targetTokens = imgTokens.AsSpan(0, numTargetTokens * HiddenDim).ToArray();
        var normImg = (float[])targetTokens.Clone();
        DiffusionOps.LayerNorm(normImg, finalGamma, finalShift, HiddenDim);

        var outPacked = Linear("proj_out", normImg, HiddenDim, InChannels);

        // 7. Unpack patches [numTargetTokens, 64] -> [16, latH, latW]
        return UnpackLatents(outPacked, latH, latW);
    }

    private (float[] imgOut, float[] txtOut) TransformerBlock(
        string prefix,
        float[] img,
        float[] txt,
        float[] tEmb,
        float[] cos,
        float[] sin,
        int numImg,
        int numTxt,
        int[]? modulateIndex)
    {
        // 1. Modulations
        var imgMod = Linear($"{prefix}.img_mod.1", DiffusionOpsSilu(tEmb), HiddenDim, HiddenDim * 6);
        var txtMod = Linear($"{prefix}.txt_mod.1", DiffusionOpsSilu(tEmb), HiddenDim, HiddenDim * 6);

        var imgS1 = imgMod.AsSpan(0 * HiddenDim, HiddenDim);
        var imgSc1 = imgMod.AsSpan(1 * HiddenDim, HiddenDim);
        var imgG1 = imgMod.AsSpan(2 * HiddenDim, HiddenDim);
        var imgS2 = imgMod.AsSpan(3 * HiddenDim, HiddenDim);
        var imgSc2 = imgMod.AsSpan(4 * HiddenDim, HiddenDim);
        var imgG2 = imgMod.AsSpan(5 * HiddenDim, HiddenDim);

        var txtS1 = txtMod.AsSpan(0 * HiddenDim, HiddenDim);
        var txtSc1 = txtMod.AsSpan(1 * HiddenDim, HiddenDim);
        var txtG1 = txtMod.AsSpan(2 * HiddenDim, HiddenDim);
        var txtS2 = txtMod.AsSpan(3 * HiddenDim, HiddenDim);
        var txtSc2 = txtMod.AsSpan(4 * HiddenDim, HiddenDim);
        var txtG2 = txtMod.AsSpan(5 * HiddenDim, HiddenDim);

        // 2. Modulated norm1 -- real reference (examples/stable-diffusion.cpp/src/model/diffusion/
        // qwen_image.hpp) applies an affine-free LayerNorm (img_norm1/txt_norm1, both constructed
        // with elementwise_affine=false, i.e. no learnable weight -- LayerNormNoAffine is the exact
        // match) BEFORE Flux::modulate, not a direct modulate of the raw residual stream. Missing
        // this caused HunyuanVideoModel to blow up to NaN by block 8 of 20 on a real checkpoint
        // (docs/086, 2026-09-18) -- fixed proactively here after finding the identical gap by
        // inspection, confirmed against this model's own real C++ reference before applying.
        var normedImg1 = (float[])img.Clone();
        DiffusionOps.LayerNormNoAffine(normedImg1, HiddenDim);
        normedImg1 = Modulate(normedImg1, numImg, imgS1, imgSc1, modulateIndex);
        var normedTxt1 = (float[])txt.Clone();
        DiffusionOps.LayerNormNoAffine(normedTxt1, HiddenDim);
        normedTxt1 = Modulate(normedTxt1, numTxt, txtS1, txtSc1, null);

        // 3. Joint Attention
        var (imgAttn, txtAttn) = JointAttention($"{prefix}.attn", normedImg1, normedTxt1, cos, sin, numImg, numTxt);

        // 4. Apply gate1 & residual
        ApplyGatedResidual(img, imgAttn, numImg, imgG1, modulateIndex);
        ApplyGatedResidual(txt, txtAttn, numTxt, txtG1, null);

        // 5. Modulated norm2 + MLP
        var normedImg2 = (float[])img.Clone();
        DiffusionOps.LayerNormNoAffine(normedImg2, HiddenDim);
        normedImg2 = Modulate(normedImg2, numImg, imgS2, imgSc2, modulateIndex);
        var normedTxt2 = (float[])txt.Clone();
        DiffusionOps.LayerNormNoAffine(normedTxt2, HiddenDim);
        normedTxt2 = Modulate(normedTxt2, numTxt, txtS2, txtSc2, null);

        var imgMlp = FeedForward($"{prefix}.img_mlp", normedImg2, numImg);
        var txtMlp = FeedForward($"{prefix}.txt_mlp", normedTxt2, numTxt);

        ApplyGatedResidual(img, imgMlp, numImg, imgG2, modulateIndex);
        ApplyGatedResidual(txt, txtMlp, numTxt, txtG2, null);

        return (img, txt);
    }

    private (float[] imgAttn, float[] txtAttn) JointAttention(
        string prefix,
        float[] img,
        float[] txt,
        float[] cos,
        float[] sin,
        int numImg,
        int numTxt)
    {
        int totalSeq = numTxt + numImg;

        var imgQ = Linear($"{prefix}.to_q", img, HiddenDim, HiddenDim);
        var imgK = Linear($"{prefix}.to_k", img, HiddenDim, HiddenDim);
        var imgV = Linear($"{prefix}.to_v", img, HiddenDim, HiddenDim);

        var txtQ = Linear($"{prefix}.add_q_proj", txt, HiddenDim, HiddenDim);
        var txtK = Linear($"{prefix}.add_k_proj", txt, HiddenDim, HiddenDim);
        var txtV = Linear($"{prefix}.add_v_proj", txt, HiddenDim, HiddenDim);

        RmsNormHeads(imgQ, numImg, NumHeads, HeadDim, SmallWeight($"{prefix}.norm_q.weight"));
        RmsNormHeads(imgK, numImg, NumHeads, HeadDim, SmallWeight($"{prefix}.norm_k.weight"));
        RmsNormHeads(txtQ, numTxt, NumHeads, HeadDim, SmallWeight($"{prefix}.norm_added_q.weight"));
        RmsNormHeads(txtK, numTxt, NumHeads, HeadDim, SmallWeight($"{prefix}.norm_added_k.weight"));

        var q = ConcatSequences(txtQ, imgQ, numTxt, numImg);
        var k = ConcatSequences(txtK, imgK, numTxt, numImg);
        var v = ConcatSequences(txtV, imgV, numTxt, numImg);

        // Apply 3D-RoPE (repeated if multi-token edit sequence)
        int ropeSeq = Math.Min(totalSeq, cos.Length / HeadDim);
        QwenImageRoPE.ApplyRoPE(q, cos, sin, ropeSeq, NumHeads, HeadDim);
        QwenImageRoPE.ApplyRoPE(k, cos, sin, ropeSeq, NumHeads, HeadDim);

        var attnOut = MultiHeadAttention(q, k, v, totalSeq, NumHeads, HeadDim);

        var txtAttnSlice = attnOut.AsSpan(0, numTxt * HiddenDim).ToArray();
        var imgAttnSlice = attnOut.AsSpan(numTxt * HiddenDim, numImg * HiddenDim).ToArray();

        var finalImg = Linear($"{prefix}.to_out.0", imgAttnSlice, HiddenDim, HiddenDim);
        var finalTxt = Linear($"{prefix}.to_add_out", txtAttnSlice, HiddenDim, HiddenDim);

        return (finalImg, finalTxt);
    }

    // Shared tiled/AVX2 attention (same [seq, heads*headDim] layout). This was a scalar,
    // single-threaded loop with a per-row allocation and a 3072-float stride through V in P·V:
    // ~27 G multiply-adds per forward on one core, most of Qwen Image's CPU step time.
    private static float[] MultiHeadAttention(float[] q, float[] k, float[] v, int seqLen, int numHeads, int headDim)
        => DiffusionOps.MultiHeadAttention(q, k, v, seqLen, seqLen, numHeads, headDim);

    private static void RmsNormHeads(float[] qk, int seqLen, int numHeads, int headDim, float[] gamma)
    {
        for (int s = 0; s < seqLen; s++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int off = (s * numHeads + h) * headDim;
                float sumSq = 0f;
                for (int d = 0; d < headDim; d++)
                {
                    float val = qk[off + d];
                    sumSq += val * val;
                }
                float invRms = 1.0f / MathF.Sqrt(sumSq / headDim + 1e-6f);
                for (int d = 0; d < headDim; d++)
                    qk[off + d] = qk[off + d] * invRms * gamma[d];
            }
        }
    }

    // Small per-block tensors (QK-norm gammas) cached: GetWeight is a locked disk read.
    private float[] SmallWeight(string name) => _biasCache.GetOrAdd("w:" + name, _ => GetWeight(name))!;

    private static float[] ConcatSequences(float[] a, float[] b, int lenA, int lenB)
    {
        var result = new float[(lenA + lenB) * HiddenDim];
        Array.Copy(a, 0, result, 0, lenA * HiddenDim);
        Array.Copy(b, 0, result, lenA * HiddenDim, lenB * HiddenDim);
        return result;
    }

    private float[] FeedForward(string prefix, float[] x, int seqLen)
    {
        // Real reference (examples/stable-diffusion.cpp/src/model/diffusion/qwen_image.hpp,
        // block.hpp's shared FeedForward): img_mlp/txt_mlp are BOTH constructed with
        // Activation::GELU, NOT the default GEGLU -- a plain net.0=GELU(dim,inner_dim) (Linear then
        // gelu, no gating) followed by net.2's down-projection. Confirmed directly against the real
        // checkpoint too: transformer_blocks.N.img_mlp.net.0.proj.weight is [3072,12288]
        // (dim->4*dim, single width), not [3072,24576] (which a GEGLU gate+value split would need)
        // -- a real bug found via ArgumentOutOfRangeException on the first real-weight run
        // (2026-09-18), not a guess.
        int intermediateDim = HiddenDim * 4;
        var up = Linear($"{prefix}.net.0.proj", x, HiddenDim, intermediateDim);
        DiffusionOps.GeluInPlace(up.AsSpan(0, seqLen * intermediateDim)); // parallel, same tanh GELU
        return Linear($"{prefix}.net.2", up, intermediateDim, HiddenDim);
    }

    private static float[] Modulate(float[] x, int seqLen, ReadOnlySpan<float> shift, ReadOnlySpan<float> scale, int[]? index)
    {
        var outF = new float[x.Length];
        for (int i = 0; i < seqLen; i++)
        {
            int off = i * HiddenDim;
            float factor = (index is not null && index[i] == 1) ? 0.0f : 1.0f;
            for (int d = 0; d < HiddenDim; d++)
            {
                float val = x[off + d];
                outF[off + d] = val * (1.0f + scale[d] * factor) + shift[d] * factor;
            }
        }
        return outF;
    }

    private static void ApplyGatedResidual(float[] x, float[] branch, int seqLen, ReadOnlySpan<float> gate, int[]? index)
    {
        for (int i = 0; i < seqLen; i++)
        {
            int off = i * HiddenDim;
            float factor = (index is not null && index[i] == 1) ? 1.0f : gate[0];
            for (int d = 0; d < HiddenDim; d++)
                x[off + d] += branch[off + d] * gate[d];
        }
    }

    private float[] ComputeTimestepEmbedding(float timestep)
    {
        // REAL BUG FOUND AND FIXED 2026-09-18 (docs/089's checkerboard-artifact investigation):
        // this call was missing flipSinToCos: true. The real reference's own timestep-embedding op
        // (examples/stable-diffusion.cpp/ggml/src/ggml-cpu/ops.cpp's
        // ggml_compute_forward_timestep_embedding_f32: `embed_data[j] = cosf(arg);
        // embed_data[j+half] = sinf(arg);`) fills [cos, sin] order -- the same flip_sin_to_cos=true
        // convention already found missing for Wan's and LTX-Video's own timestep embeddings this
        // project's history (same shared-helper default-false bug class, independently recurring a
        // third time here). Every AdaLN modulation in the DiT derives from this one value.
        var emb = DiffusionOps.SinusoidalTimestepEmbedding(timestep, flipSinToCos: true);
        var t0 = Linear("time_text_embed.timestep_embedder.linear_1", emb, 256, HiddenDim);
        DiffusionOps.SiluInPlace(t0);
        return Linear("time_text_embed.timestep_embedder.linear_2", t0, HiddenDim, HiddenDim);
    }

    private static float[] DiffusionOpsSilu(float[] x)
    {
        var res = (float[])x.Clone();
        DiffusionOps.SiluInPlace(res);
        return res;
    }

    private float[] Linear(string name, float[] x, int inDim, int outDim)
    {
        string wName = Resolve($"{name}.weight");
        string bName = Resolve($"{name}.bias");
        // Cached: ReadF32 is a locked disk read, and this ran for every Linear of every block and step.
        float[]? b = _biasCache.GetOrAdd(bName, n => _weights.Contains(n) ? _weights.ReadF32(n) : null);
        int rows = x.Length / inDim;
        var result = new float[rows * outDim];
        _cache.Linear(wName, x, b ?? ReadOnlySpan<float>.Empty, result, rows, inDim, outDim);
        return result;
    }

    public static float[] PackLatents(float[] latents, int latH, int latW)
    {
        int patchH = latH / PatchSize;
        int patchW = latW / PatchSize;
        int numTokens = patchH * patchW;
        var packed = new float[numTokens * InChannels];

        for (int ph = 0; ph < patchH; ph++)
        {
            for (int pw = 0; pw < patchW; pw++)
            {
                int tokenIdx = ph * patchW + pw;
                int tokenOff = tokenIdx * InChannels;
                int chanOffset = 0;

                for (int c = 0; c < OutChannels; c++)
                {
                    for (int dy = 0; dy < PatchSize; dy++)
                    {
                        for (int dx = 0; dx < PatchSize; dx++)
                            packed[tokenOff + chanOffset++] = latents[(c * latH + (ph * PatchSize + dy)) * latW + (pw * PatchSize + dx)];
                    }
                }
            }
        }
        return packed;
    }

    public static float[] UnpackLatents(float[] packed, int latH, int latW)
    {
        int patchH = latH / PatchSize;
        int patchW = latW / PatchSize;
        var unpacked = new float[OutChannels * latH * latW];

        for (int ph = 0; ph < patchH; ph++)
        {
            for (int pw = 0; pw < patchW; pw++)
            {
                int tokenIdx = ph * patchW + pw;
                int tokenOff = tokenIdx * InChannels;
                int chanOffset = 0;

                for (int c = 0; c < OutChannels; c++)
                {
                    for (int dy = 0; dy < PatchSize; dy++)
                    {
                        for (int dx = 0; dx < PatchSize; dx++)
                            unpacked[(c * latH + (ph * PatchSize + dy)) * latW + (pw * PatchSize + dx)] = packed[tokenOff + chanOffset++];
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
            _residentGpuWorkspace?.Dispose();
            _residentGpuWeights?.Dispose();
            _cache.Dispose();
            _weights.Dispose();
        }
    }
}
