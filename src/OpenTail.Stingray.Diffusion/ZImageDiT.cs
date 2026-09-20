
namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Z-Image Scalable Single-Stream DiT (S3-DiT) forward pass.
///
/// Architecture (30 layers + 2 refiners each):
///   x_embedder:       Linear(64, 3840)   — project image patches
///   cap_embedder:     [RMSNorm(2560) → Linear(2560, 3840)]  — project Qwen3 text features
///   t_embedder:       TimestepMLP → [256]   — timestep conditioning
///   context_refiner × 2:  transformer blocks (no modulation) on text tokens only
///   noise_refiner × 2:    transformer blocks (with modulation) on image tokens only
///   layers × 30:          single-stream blocks on [txt | img] concat with adaLN(256-dim)
///   final_layer:      AdaLN + Linear(3840, 64)  — project image patches back
///
/// Weight names follow diffusers naming (Tongyi-MAI/Z-Image-Turbo/transformer/).
/// Supports both single-file safetensors and multi-shard directories.
/// </summary>
public sealed class ZImageDiT : IDisposable
{
    private readonly IWeightLoader _st;
    private readonly ZImageParams _p;
    private readonly ZImageRoPE _rope;
    private readonly IComputeBackend? _backend;
    private readonly Dictionary<string, float[]> _cache = new(StringComparer.Ordinal);
    /// <summary>Raw quantized weight buffers cached on the GPU (lazy-populated by GPU dequant path).</summary>
    private readonly Dictionary<string, Core.Tensor>? _gpuWeights;
    /// <summary>bf16 dequantized weights cached on GPU — uploaded once on first use, reused every step.</summary>
    private readonly Dictionary<string, Core.Tensor>? _gpuWeightsBf16;
    /// <summary>fp16 dequantized weights cached on GPU — uploaded once on first use, reused every step.</summary>
    private readonly Dictionary<string, Core.Tensor>? _gpuWeightsFp16;
    /// <summary>fp8 E4M3 weights cached on GPU — uploaded once on first use, reused every step (sm_89+).</summary>
    private readonly Dictionary<string, Core.Tensor>? _gpuWeightsFp8;
    private bool _disposed;

    // ── RoPE freq cache (position IDs are constant across denoising steps) ──
    private int[]?   _cachedImgPosIds;
    private float[]? _cachedImgFreqs;
    private int[]?   _cachedTxtPosIds;
    private float[]? _cachedTxtFreqs;
    private float[]? _cachedCombinedFreqs;

    // ── Text Context cache (invariant across denoising steps for a given prompt) ──
    private float[]? _cachedTxtEmbeds;
    private float[]? _cachedRefinedTxtHid;

    // Cached unit-scale / unit-gate arrays (unmodulated blocks use scale=1, gate=1)
    private float[]? _onesCache;

    // GPU residency for the 30 main `layers.N` blocks (docs/094 Phase 7, 2026-09-20): the
    // resident ApplyBlockGpu/ZImageGpuWeights/ZImageGpuWorkspace trio already existed, fully
    // built and parity-tested (ZImageGpuParityTests), but nothing in this class's own Forward()
    // ever called it -- the real per-step loop still used the old per-op immediate-dispatch MatQ
    // path. Wiring it in here, not writing a new implementation.
    private ZImageGpuWeights? _residentGpuWeights;
    private ZImageGpuWorkspace? _residentGpuWorkspace;
    private int _residentGpuNTok = -1;

    /// <summary>A real structural bug WAS found and fixed 2026-09-20 (docs/094 Phase 7):
    /// `ApplyBlockGpu` passed `AdaLNModulate` a scale of (1+rawScale) where the shader's own
    /// formula is norm*(1.0+s)+sh, double-counting the "+1" AND never applying the learned RMSNorm
    /// gamma (attention_norm1.weight/ffn_norm1.weight, uploaded but referenced nowhere in the GPU
    /// path). Fixed by folding the host-side norm weight into the scale vector before upload:
    /// `s = normW1*(1+rawScale) - 1`. Real-scale bisection went from cosine=0.9885 at block 0 to
    /// cosine=0.999997 at block 0 -- confirms the fix is real and correct at the block-math level.
    /// BUT the full end-to-end image is STILL not the clean coherent apple the naive/CPU path
    /// produces (a quilted/patchwork artifact remains, plausibly this iGPU's forced FP16 weight
    /// upload compounding over 30 blocks / 4 turbo-schedule steps -- see the doc comment for full
    /// analysis). Kept `false` matching this session's own standard (SD3.5/Qwen Image): don't ship
    /// fast-but-visibly-wrong. Not a `const` so the branch stays reachable (avoids CS0162).</summary>
    private static readonly bool ZImageGpuResidencyRealScaleBugFound = false;

    // Per-block diagnostic hooks for the real-scale bug bisection (docs/094 Phase 7, 2026-09-20),
    // same convention already proven for WanModel/QwenImageModel this session.
    public Action<int, float[]>? OnMainBlockOutputCpu { get; set; }
    public Action<int, float[]>? OnMainBlockOutputGpu { get; set; }

    // Finer within-block debug hooks (docs/094 Phase 7 sub-block bisection, 2026-09-20): capture
    // the four modulation vectors, and the activation right after each sub-block's gated residual
    // add, so a divergence can be localized to modulation vs. attention vs. FFN within a single
    // block rather than only "somewhere in the whole block".
    public Action<float[], float[], float[], float[]>? OnModulationCpu { get; set; }
    public Action<float[], float[], float[], float[]>? OnModulationGpu { get; set; }
    public Action<float[]>? OnAfterAttnCpu { get; set; }
    public Action<float[]>? OnAfterAttnGpu { get; set; }

    /// <summary>Test-only entry point running the real GPU-resident 30-block loop directly
    /// (bypassing the disabled-by-default dispatch in <see cref="Forward"/>), for a real-scale
    /// bisection against <see cref="RunMainLayersCpuForTest"/>.</summary>
    public void RunMainLayersGpuForTest(IImageOpsBackend imageOps, float[] x, int nTok, float[] freqs, float[] adaln)
        => RunMainLayersGpu(imageOps, x, nTok, freqs, adaln);

    /// <summary>Test-only entry point running the real CPU 30-block loop directly, with per-block
    /// capture, for a real-scale bisection against <see cref="RunMainLayersGpuForTest"/>.</summary>
    public void RunMainLayersCpuForTest(float[] x, int nTok, float[] freqs, float[] adaln)
    {
        for (int l = 0; l < _p.NLayers; l++)
        {
            ApplyBlock($"layers.{l}", x, nTok, freqs, adaln, true);
            OnMainBlockOutputCpu?.Invoke(l, (float[])x.Clone());
        }
    }

    /// <summary>Minimum batch size to route a MatQ call through the GPU backend.</summary>
    private const int MinGpuBatch = 16;
    private readonly QuantizedWeightCache _quantizedCache;
    private readonly Dictionary<string, float[]?> _biasCache = new(StringComparer.Ordinal);

    public ZImageDiT(IWeightLoader st, ZImageParams p, IComputeBackend? backend = null)
    {
        _st      = st;
        _p       = p;
        _rope    = new ZImageRoPE(p);
        _backend = backend;
        _quantizedCache = new QuantizedWeightCache(st);
        if (backend?.SupportsGpuDequant == true)
            _gpuWeights = new Dictionary<string, Core.Tensor>(StringComparer.Ordinal);
        // Cache bf16 weights on GPU so each weight is uploaded only once across all denoising steps.
        if (backend?.BestSgemmPrecision == SgemmPrecision.Bf16)
            _gpuWeightsBf16 = new Dictionary<string, Core.Tensor>(StringComparer.Ordinal);
        // Cache fp16 weights on GPU — avoids re-dequantising + re-uploading every GEMM call.
        if (backend?.BestSgemmPrecision == SgemmPrecision.Fp16)
            _gpuWeightsFp16 = new Dictionary<string, Core.Tensor>(StringComparer.Ordinal);
        // Cache fp8 weights on GPU (sm_89+, 2× smaller than bf16, use fp8 tensor cores).
        if (backend?.BestSgemmPrecision == SgemmPrecision.Fp8E4M3)
            _gpuWeightsFp8 = new Dictionary<string, Core.Tensor>(StringComparer.Ordinal);
    }

    // ── Main forward pass ─────────────────────────────────────────────────

    /// <summary>
    /// Run the S3-DiT forward pass for one denoising step.
    /// </summary>
    /// <param name="imgPatches">Noisy image patches [nImg, 64].</param>
    /// <param name="imgPosIds">Image patch position IDs [nImg*3].</param>
    /// <param name="txtEmbeds">Qwen3 text hidden states [nTxt, 2560].</param>
    /// <param name="txtPosIds">Text position IDs [nTxt*3].</param>
    /// <param name="t">Fractional timestep in [0,1].</param>
    /// <returns>Predicted velocity (packed patches) [nImg, 64].</returns>
    public float[] Forward(
        float[] imgPatches, int[] imgPosIds,
        float[] txtEmbeds,  int[] txtPosIds,
        float t)
    {
        int nImg = imgPatches.Length / _p.PatchDim;
        int nTxt = txtEmbeds.Length / _p.CapFeatDim;
        int dim  = _p.Dim;

        // ── 1. Embed ──────────────────────────────────────────────────────
        var imgHid = MatQ(imgPatches, nImg, _p.PatchDim, "x_embedder.weight", "x_embedder.bias", dim);
        var adaln  = TimestepEmbed(t);           // [256]

        // ── 2. Build per-group RoPE freqs (cached: pos IDs are constant across steps) ──
        if (!ReferenceEquals(imgPosIds, _cachedImgPosIds))
        {
            _cachedImgFreqs  = _rope.BuildFreqs(imgPosIds, nImg);
            _cachedImgPosIds = imgPosIds;
        }
        if (!ReferenceEquals(txtPosIds, _cachedTxtPosIds))
        {
            _cachedTxtFreqs  = _rope.BuildFreqs(txtPosIds, nTxt);
            _cachedTxtPosIds = txtPosIds;
        }
        var imgFreqs = _cachedImgFreqs!;
        var txtFreqs = _cachedTxtFreqs!;

        // ── 3. Refine text context (cached across steps) ────────────────────
        float[] txtHid;
        if (ReferenceEquals(txtEmbeds, _cachedTxtEmbeds) && _cachedRefinedTxtHid is not null)
        {
            txtHid = _cachedRefinedTxtHid;
        }
        else
        {
            txtHid = EmbedCap(txtEmbeds, nTxt);
            for (int r = 0; r < _p.NRefinerLayers; r++)
                ApplyBlock($"context_refiner.{r}", txtHid, nTxt, txtFreqs, null, false);

            _cachedTxtEmbeds = txtEmbeds;
            _cachedRefinedTxtHid = (float[])txtHid.Clone();
        }

        // Noise refiner: 2 modulated blocks on image tokens with image RoPE
        for (int r = 0; r < _p.NRefinerLayers; r++)
            ApplyBlock($"noise_refiner.{r}", imgHid, nImg, imgFreqs, adaln, true);

        // ── 4. Concatenate + combined RoPE ────────────────────────────────
        // Reference order: [img, txt] — image tokens first, text tokens after.
        int nTotal = nImg + nTxt;
        var x = new float[nTotal * dim];
        Array.Copy(imgHid, 0, x, 0,         nImg * dim);
        Array.Copy(txtHid, 0, x, nImg * dim, nTxt * dim);

        // Build combined position IDs and RoPE freqs: [imgPos, txtPos]
        // Cache the combined freqs alongside img/txt — they change iff either input changes.
        if (_cachedCombinedFreqs is null ||
            !ReferenceEquals(imgPosIds, _cachedImgPosIds) ||
            !ReferenceEquals(txtPosIds, _cachedTxtPosIds))
        {
            var combinedPos = new int[(nImg + nTxt) * 3];
            Array.Copy(imgPosIds, 0, combinedPos, 0,        nImg * 3);
            Array.Copy(txtPosIds, 0, combinedPos, nImg * 3, nTxt * 3);
            _cachedCombinedFreqs = _rope.BuildFreqs(combinedPos, nTotal);
        }
        var freqs = _cachedCombinedFreqs;

        // ── 4. 30 main transformer blocks ─────────────────────────────────
        // DISABLED, 2026-09-20 (docs/094 Phase 7): real end-to-end run (256x256/4 steps) completed
        // in 64.8s (down from 183.6s -- a real 2.8x speedup on paper) but produced PURE NOISE, not
        // the known-good coherent apple this exact config verifies on the CPU/naive-GPU path. The
        // existing small-scale synthetic ZImageGpuParityTests (t=24, dim=384) still passes -- the
        // bug is real-scale-specific and not yet found. Do NOT re-enable
        // (ZImageGpuResidencyRealScaleBugFound=false) until root-caused; a fast wrong answer is not
        // a result, matching this whole project's own hard-won discipline (see docs/094's Qwen
        // Image/SD3.5 GPU investigations for the same lesson repeatedly).
        if (ZImageGpuResidencyRealScaleBugFound && _backend is IImageOpsBackend imageOpsMain)
        {
            try
            {
                RunMainLayersGpu(imageOpsMain, x, nTotal, freqs, adaln);
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                Console.WriteLine($"[ZImage GPU Exception] {ex}");
                for (int l = 0; l < _p.NLayers; l++)
                    ApplyBlock($"layers.{l}", x, nTotal, freqs, adaln, true);
            }
        }
        else
        {
            for (int l = 0; l < _p.NLayers; l++)
                ApplyBlock($"layers.{l}", x, nTotal, freqs, adaln, true);
        }

        // ── 5. Extract image portion (first nImg tokens) and apply final layer
        var imgOut = new float[nImg * dim];
        Array.Copy(x, 0, imgOut, 0, nImg * dim);

        return FinalLayer(imgOut, nImg, adaln);
    }

    // GPU-resident dispatch for the 30 main `layers.N` blocks (docs/094 Phase 7). Mutates `x` in
    // place, matching ApplyBlock's own CPU calling convention exactly so the caller's fallback
    // path is a drop-in swap. Weights upload once (first call) and are reused for this instance's
    // lifetime; the workspace is rebuilt only if the token count changes (a different
    // resolution/prompt length).
    private float[]? _cachedGpuRopeCos, _cachedGpuRopeSin;
    private float[]? _cachedGpuRopeFreqsKey;

    private void RunMainLayersGpu(IImageOpsBackend imageOps, float[] x, int nTok, float[] freqs, float[] adaln)
    {
        int dim = _p.Dim;
        int headDim = _p.HeadDim;
        int halfHead = headDim / 2;

        _residentGpuWeights ??= new ZImageGpuWeights(_backend!, W, _p.NLayers, dim, headDim, _p.FfnHidden, _p.AdalnEmbedDim);

        // De-interleave freqs (real format: (cos,sin) pairs per token) into the compact GPU
        // layout, cached by reference since freqs is the same array object across denoising steps
        // (Forward's own _cachedCombinedFreqs) -- avoids redoing this O(nTok*halfHead) split every
        // single step.
        if (!ReferenceEquals(_cachedGpuRopeFreqsKey, freqs))
        {
            var ropeCos = new float[nTok * halfHead];
            var ropeSin = new float[nTok * halfHead];
            for (int i = 0; i < nTok * halfHead; i++)
            {
                ropeCos[i] = freqs[i * 2];
                ropeSin[i] = freqs[i * 2 + 1];
            }
            _cachedGpuRopeCos = ropeCos;
            _cachedGpuRopeSin = ropeSin;
            _cachedGpuRopeFreqsKey = freqs;
        }

        if (_residentGpuWorkspace is null || _residentGpuNTok != nTok)
        {
            _residentGpuWorkspace?.Dispose();
            _residentGpuWorkspace = new ZImageGpuWorkspace(_backend!, nTok, dim, _p.FfnHidden, _cachedGpuRopeCos!, _cachedGpuRopeSin!, headDim);
            _residentGpuNTok = nTok;
        }

        var ws = _residentGpuWorkspace;
        using var xGpu = _backend!.Upload(x, TensorShape.D2(nTok, dim), exact: true);
        imageOps.ScaleInPlace(ws.X, 0f);
        _backend.AddInPlace(ws.X, xGpu);
        using var adalnGpu = _backend.Upload(adaln, TensorShape.D2(1, _p.AdalnEmbedDim), exact: true);

        // NOT wrapped in BeginBatch()/EndBatch() per block: ApplyBlockGpu does a real mid-block
        // Download() (the tanh-gate modulation round-trip, see its own doc comment) which requires
        // an immediate submit+fence-wait -- invalid while a batch recording session is open (the
        // exact same class of driver rejection already documented in this codebase for SDXL's own
        // Stage 5 attempt). The already-existing, already-verified-correct ZImageGpuParityTests
        // exercises ApplyBlockGpu completely unbatched too -- matching that, not inventing a new
        // untested combination. The real residency win here comes from ws.X staying GPU-resident
        // across blocks (no per-block activation Upload/Download), not from command-buffer
        // batching, which is a separate, independent optimization this pass doesn't attempt.
        for (int l = 0; l < _p.NLayers; l++)
        {
            ApplyBlockGpu(_residentGpuWeights.Layers[l], ws, nTok, adalnGpu, imageOps);
            if (OnMainBlockOutputGpu != null)
            {
                var hostX = new float[nTok * dim];
                _backend.Download(ws.X, hostX);
                OnMainBlockOutputGpu(l, hostX);
            }
        }

        _backend.Download(ws.X, x);
    }

    // ── Embedding helpers ─────────────────────────────────────────────────

    private float[] EmbedCap(float[] txtFeats, int nTxt)
    {
        var normed = new float[txtFeats.Length];
        var normW  = W("cap_embedder.0.weight");
        for (int i = 0; i < nTxt; i++)
        {
            int off = i * _p.CapFeatDim;
            DiffusionOps.RmsNorm(txtFeats, off, _p.CapFeatDim, normW, _p.NormEps, normed, off);
        }
        return MatQ(normed, nTxt, _p.CapFeatDim, "cap_embedder.1.weight", "cap_embedder.1.bias", _p.Dim);
    }

    private float[] TimestepEmbed(float t)
    {
        // Reference convention: t_model=0 at pure noise, t_model=1 at clean image.
        // Our scheduler sigma=t: t=1 at noise, t=0 at clean.
        // So t_model = 1 - t.
        float tScaled = (1f - t) * _p.TScale;
        int   freqDim = 256;
        var   sinEmb  = DiffusionOps.SinusoidalTimestepEmbedding(tScaled, freqDim);
        var h = MatQ(sinEmb, 1, freqDim, "t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias", 1024);
        DiffusionOps.SiLUInPlace(h);
        return MatQ(h, 1, 1024, "t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias", _p.AdalnEmbedDim);
    }

    // ── Block implementation ───────────────────────────────────────────────

    /// <summary>Test-only entry point exposing one CPU block's in-place output directly, for a
    /// real GPU-vs-CPU parity check (mirrors F5TTS's/Parler's own "ForTest" accessor pattern).</summary>
    public void ApplyBlockForTest(string prefix, float[] x, int nTok, float[]? freqs, float[]? adaln, bool modulated) =>
        ApplyBlock(prefix, x, nTok, freqs, adaln, modulated);

    private void ApplyBlock(string prefix, float[] x, int nTok,
                            float[]? freqs, float[]? adaln, bool modulated)
    {
        int dim = _p.Dim;

        float[] scaleMsa, gateMsa, scaleMlp, gateMlp;
        if (modulated)
        {
            // adaLN_modulation: Linear(256, 4*dim) + bias → 4 vectors each [dim]
            var mod = MatQ(adaln!, 1, _p.AdalnEmbedDim,
                           $"{prefix}.adaLN_modulation.0.weight",
                           $"{prefix}.adaLN_modulation.0.bias",
                           4 * dim);
            scaleMsa = mod.AsSpan(0,       dim).ToArray();
            gateMsa  = mod.AsSpan(dim,     dim).ToArray();
            scaleMlp = mod.AsSpan(2 * dim, dim).ToArray();
            gateMlp  = mod.AsSpan(3 * dim, dim).ToArray();
            TensorPrimitives.Tanh(gateMsa.AsSpan(), gateMsa.AsSpan());
            TensorPrimitives.Tanh(gateMlp.AsSpan(), gateMlp.AsSpan());
            TensorPrimitives.Add(scaleMsa.AsSpan(), 1f, scaleMsa.AsSpan());
            TensorPrimitives.Add(scaleMlp.AsSpan(), 1f, scaleMlp.AsSpan());
        }
        else
        {
            var ones = _onesCache ??= Ones(dim);
            scaleMsa = scaleMlp = ones;
            gateMsa  = gateMlp  = ones;
        }

        OnModulationCpu?.Invoke((float[])scaleMsa.Clone(), (float[])gateMsa.Clone(), (float[])scaleMlp.Clone(), (float[])gateMlp.Clone());

        // ── Attention sub-block ───────────────────────────────────────────
        var normW1  = W($"{prefix}.attention_norm1.weight");
        var attnBuf = ArrayPool<float>.Shared.Rent(nTok * dim);
        float[] attnOut;
        try
        {
            Parallel.For(0, nTok, t =>
            {
                int off = t * dim;
                DiffusionOps.RmsNorm(x, off, dim, normW1, _p.NormEps, attnBuf, off);
                TensorPrimitives.Multiply(attnBuf.AsSpan(off, dim), scaleMsa, attnBuf.AsSpan(off, dim));
            });
            attnOut = SelfAttention(prefix, attnBuf, nTok, freqs);
        }
        finally { ArrayPool<float>.Shared.Return(attnBuf); }

        var normW2 = W($"{prefix}.attention_norm2.weight");
        Parallel.For(0, nTok, t =>
        {
            int off = t * dim;
            DiffusionOps.RmsNorm(attnOut, off, dim, normW2, _p.NormEps, attnOut, off);
            TensorPrimitives.MultiplyAdd<float>(attnOut.AsSpan(off, dim), gateMsa,
                                                x.AsSpan(off, dim), x.AsSpan(off, dim));
        });
        OnAfterAttnCpu?.Invoke((float[])x.Clone());

        // ── FFN sub-block ─────────────────────────────────────────────────
        var normW3  = W($"{prefix}.ffn_norm1.weight");
        var ffnBuf  = ArrayPool<float>.Shared.Rent(nTok * dim);
        float[] gate, up, ffnOut;
        try
        {
            Parallel.For(0, nTok, t =>
            {
                int off = t * dim;
                DiffusionOps.RmsNorm(x, off, dim, normW3, _p.NormEps, ffnBuf, off);
                TensorPrimitives.Multiply(ffnBuf.AsSpan(off, dim), scaleMlp, ffnBuf.AsSpan(off, dim));
            });

            // SiLU-gated FFN: w2(silu(w1(x)) * w3(x))
            gate   = MatQ(ffnBuf, nTok, dim, $"{prefix}.feed_forward.w1.weight", _p.FfnHidden);
            up     = MatQ(ffnBuf, nTok, dim, $"{prefix}.feed_forward.w3.weight", _p.FfnHidden);
        }
        finally { ArrayPool<float>.Shared.Return(ffnBuf); }

        DiffusionOps.SiluInPlace(gate.AsSpan());
        TensorPrimitives.Multiply(gate.AsSpan(), up.AsSpan(), gate.AsSpan());
        ffnOut = MatQ(gate, nTok, _p.FfnHidden, $"{prefix}.feed_forward.w2.weight", dim);

        var normW4 = W($"{prefix}.ffn_norm2.weight");
        Parallel.For(0, nTok, t =>
        {
            int off = t * dim;
            DiffusionOps.RmsNorm(ffnOut, off, dim, normW4, _p.NormEps, ffnOut, off);
            TensorPrimitives.MultiplyAdd<float>(ffnOut.AsSpan(off, dim), gateMlp,
                                                x.AsSpan(off, dim), x.AsSpan(off, dim));
        });
    }

    /// <summary>
    /// GPU-resident equivalent of <see cref="ApplyBlock"/> for the `modulated=true` case only
    /// (the 30 main `layers.N` blocks — the dominant cost; `context_refiner`/`noise_refiner` are
    /// a real, separate follow-up, see docs/075's own scoping note). Every op dispatched on
    /// <paramref name="imageOps"/> against <paramref name="ws"/>'s preallocated device buffers and
    /// <paramref name="bw"/>'s persistent VRAM-resident weights. This is a real "sandwich norm"
    /// structure (RMSNorm both before AND after each sub-block, unlike FLUX/Wan/F5's pre-norm-only
    /// convention) — see this method's own step comments for exactly how each piece is composed
    /// from existing, already-proven GPU kernels (no new shaders needed except `SiLuMul`, already
    /// existed on `VulkanBackend` but was newly exposed on the interface for this work).
    /// Caller wraps this in <c>imageOps.BeginBatch()</c>/<c>EndBatch()</c> per block.
    /// </summary>
    public void ApplyBlockGpu(ZImageGpuWeights.BlockWeights bw, ZImageGpuWorkspace ws, int nTok, Core.Tensor adalnGpu, Core.IImageOpsBackend imageOps)
    {
        var visionOps = (Core.IVisionOpsBackend)imageOps;
        int dim = _p.Dim;
        int nHeads = _p.NHeads;
        int headDim = _p.HeadDim;
        int ffnHidden = _p.FfnHidden;

        // 0. Real per-block modulation: adaLN_modulation.0(adaln) -> [4*dim] -> chunk into
        // scaleMsa/gateMsa/scaleMlp/gateMlp, tanh(gate), 1+scale. No existing GPU op does tanh,
        // so this one small (4*dim-float) piece round-trips through the CPU once per block --
        // same accepted tradeoff F5's own ForwardGpu made for its own per-block modulation.
        using var modGpu = imageOps.Allocate(Core.TensorShape.D2(1, 4 * dim));
        imageOps.Sgemm(modGpu, adalnGpu, bw.AdaLnModW, 1, 256, 4 * dim);
        imageOps.AddRowBroadcastInPlace(modGpu, bw.AdaLnModB, 1, 4 * dim);
        var modHost = new float[4 * dim];
        imageOps.Download(modGpu, modHost);

        // CPU's real formula (ApplyBlock): DiffusionOps.RmsNorm(x, ..., normW1, ...) applies the
        // LEARNED per-channel gamma (attention_norm1.weight/ffn_norm1.weight) as part of the
        // RMSNorm itself, THEN a separate elementwise multiply by scaleMsa/scaleMlp (=1+rawScale
        // from the adaLN modulation projection). `AdaLNModulate`'s shader has no learned-gamma
        // input at all (unweighted RMSNorm) and internally computes norm*(1.0+s)+shift -- so to
        // reproduce CPU's norm*normW1*(1+rawScale) exactly, the value passed as `s` must be
        // (normW1*(1+rawScale) - 1), NOT (1+rawScale) directly. Passing (1+rawScale) as `s`
        // silently (a) drops normW1 entirely and (b) double-counts the "+1" baked into the
        // shader's own "1.0+s" -- this was the real root cause of Z-Image GPU residency's
        // real-scale block-0 divergence (docs/094 Phase 7): invisible in the small-scale parity
        // test because it used synthetic all-ones norm weights (masking (a)) with a loose enough
        // tolerance to pass despite (b).
        var scaleMsa = new float[dim];
        var gateMsa = new float[dim];
        var scaleMlp = new float[dim];
        var gateMlp = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            float rawScalePlus1Msa = modHost[i] + 1f;
            scaleMsa[i] = bw.AttnNorm1WHost[i] * rawScalePlus1Msa - 1f;
            gateMsa[i] = MathF.Tanh(modHost[dim + i]);
            float rawScalePlus1Mlp = modHost[2 * dim + i] + 1f;
            scaleMlp[i] = bw.FfnNorm1WHost[i] * rawScalePlus1Mlp - 1f;
            gateMlp[i] = MathF.Tanh(modHost[3 * dim + i]);
        }
        using var scaleMsaGpu = imageOps.Upload(scaleMsa, Core.TensorShape.D1(dim), exact: true);
        using var gateMsaGpu = imageOps.Upload(gateMsa, Core.TensorShape.D1(dim), exact: true);
        using var scaleMlpGpu = imageOps.Upload(scaleMlp, Core.TensorShape.D1(dim), exact: true);
        using var gateMlpGpu = imageOps.Upload(gateMlp, Core.TensorShape.D1(dim), exact: true);

        // 1. Attention sub-block: pre-norm+scale (RMSNorm, zero shift -- Z-Image has none) ->
        // fused QKV -> per-HEAD QK-norm (reusing RmsNormBatched by reinterpreting the [t,dim]
        // buffer as [t*nHeads,headDim] rows -- Z-Image's real QK-norm scope is per-head, NOT
        // Wan's full-row convention, confirmed via ApplyPerHeadRmsNorm's own CPU implementation)
        // -> RoPE (Flux2DRoPE, applies to ALL heads uniformly, matching Z-Image's own convention,
        // no partial-head quirk like F5) -> attention -> O-proj -> POST-norm (RMSNorm on the raw
        // branch output) -> gated residual.
        visionOps.AdaLNModulate(ws.Normed, ws.X, ws.Zeros, scaleMsaGpu, nTok, dim, isRmsNorm: true, eps: _p.NormEps);
        imageOps.Sgemm(ws.Qkv, ws.Normed, bw.QkvW, nTok, dim, 3 * dim);

        // Split fused QKV into separate Q/K/V buffers -- reuse FLUX's own existing kernel
        // (real, already-proven op, exactly this shape: [nTok,3*dim] -> three [nTok,dim]).
        visionOps.FluxUnpackQkv(ws.Qkv, ws.Q, ws.K, ws.V, nTok, dim, dstTokenOffset: 0);

        // Per-head QK-norm: reinterpret [nTok,dim] as [nTok*nHeads,headDim] rows -- the SAME
        // real gamma[headDim] is shared across all heads/tokens (matches ApplyPerHeadRmsNorm's
        // own single [headDim]-wide weight, not a per-head-distinct one).
        visionOps.RmsNormBatched(ws.Q, ws.Q, bw.QNormW, headDim, nTok * nHeads, eps: _p.NormEps);
        visionOps.RmsNormBatched(ws.K, ws.K, bw.KNormW, headDim, nTok * nHeads, eps: _p.NormEps);

        visionOps.Flux2DRoPE(ws.Q, ws.K, ws.RopeCos, ws.RopeSin, startToken: 0, tokenCount: nTok, numHeads: nHeads, headDim: headDim);

        imageOps.MultiHeadAttentionTiled(ws.AttnOut, ws.Q, ws.K, ws.V, nTok, nTok, nHeads, headDim);
        imageOps.Sgemm(ws.AttnProj, ws.AttnOut, bw.OutW, nTok, dim, dim);

        visionOps.RmsNormBatched(ws.AttnProj, ws.AttnProj, bw.AttnNorm2W, dim, nTok, eps: _p.NormEps);
        visionOps.ScaleGateAdd(ws.X, ws.AttnProj, gateMsaGpu, nTok, dim);

        // 2. FFN sub-block: same sandwich pattern. w2(silu(w1(x)) * w3(x)) -- SiLU-gated, NOT
        // GELU-gated (F5/CosyVoice3's own convention does not apply here).
        visionOps.AdaLNModulate(ws.Normed, ws.X, ws.Zeros, scaleMlpGpu, nTok, dim, isRmsNorm: true, eps: _p.NormEps);
        imageOps.Sgemm(ws.FfnGate, ws.Normed, bw.W1, nTok, dim, ffnHidden);
        imageOps.Sgemm(ws.FfnUp, ws.Normed, bw.W3, nTok, dim, ffnHidden);
        imageOps.SiLuMul(ws.FfnGate, ws.FfnUp);
        imageOps.Sgemm(ws.FfnOut, ws.FfnGate, bw.W2, nTok, ffnHidden, dim);

        visionOps.RmsNormBatched(ws.FfnOut, ws.FfnOut, bw.FfnNorm2W, dim, nTok, eps: _p.NormEps);
        visionOps.ScaleGateAdd(ws.X, ws.FfnOut, gateMlpGpu, nTok, dim);
    }

    // ── Self-attention ────────────────────────────────────────────────────

    private float[] SelfAttention(string prefix, float[] x, int nTok, float[]? freqs)
    {
        int dim     = _p.Dim;
        int nHeads  = _p.NHeads;
        int headDim = _p.HeadDim;

        // Fused QKV: [nTok, 3*dim] — split into separate q, k, v in parallel
        var qkv = MatQ(x, nTok, dim, $"{prefix}.attention.qkv.weight", 3 * dim);
        var q = new float[nTok * dim];
        var k = new float[nTok * dim];
        var v = new float[nTok * dim];
        Parallel.For(0, nTok, tok =>
        {
            int srcOff = tok * 3 * dim;
            int dstOff = tok * dim;
            qkv.AsSpan(srcOff,           dim).CopyTo(q.AsSpan(dstOff));
            qkv.AsSpan(srcOff + dim,     dim).CopyTo(k.AsSpan(dstOff));
            qkv.AsSpan(srcOff + 2 * dim, dim).CopyTo(v.AsSpan(dstOff));
        });

        // Per-head QK norm
        var qNormW = W($"{prefix}.attention.q_norm.weight");
        var kNormW = W($"{prefix}.attention.k_norm.weight");
        ApplyPerHeadRmsNorm(q, nTok, nHeads, headDim, qNormW);
        ApplyPerHeadRmsNorm(k, nTok, nHeads, headDim, kNormW);

        if (freqs != null)
        {
            _rope.Apply(q, nTok, nHeads, freqs);
            _rope.Apply(k, nTok, nHeads, freqs);
        }

        var attn = new float[nTok * dim];
        Wan.WanAttention.TiledMultiHeadAttention(q, k, v, attn.AsSpan(), nTok, nTok, nHeads, headDim);

        return MatQ(attn, nTok, dim, $"{prefix}.attention.out.weight", dim);
    }

    private static void ApplyPerHeadRmsNorm(float[] x, int nTok, int nHeads, int headDim,
                                            float[] normW)
    {
        float eps = 1e-5f;
        var   wSp = normW.AsSpan(0, headDim);
        for (int t = 0; t < nTok; t++)
        for (int h = 0; h < nHeads; h++)
        {
            var row  = x.AsSpan((t * nHeads + h) * headDim, headDim);
            float ss = TensorPrimitives.Dot<float>(row, row);
            float r  = 1f / MathF.Sqrt(ss / headDim + eps);
            TensorPrimitives.Multiply(row, wSp, row);
            TensorPrimitives.Multiply(row, r,   row);
        }
    }

    // ── Final layer ───────────────────────────────────────────────────────

    private float[] FinalLayer(float[] imgHid, int nImg, float[] adaln)
    {
        int dim    = _p.Dim;
        int outDim = _p.PatchDim;  // 64
        string fl  = "final_layer";

        // adaLN_modulation: SiLU(adaln) → Linear(256, dim) + bias → scale
        var adaln_silu = ArrayPool<float>.Shared.Rent(_p.AdalnEmbedDim);
        try
        {
            adaln.AsSpan(0, _p.AdalnEmbedDim).CopyTo(adaln_silu.AsSpan(0, _p.AdalnEmbedDim));
            DiffusionOps.SiluInPlace(adaln_silu.AsSpan(0, _p.AdalnEmbedDim));
            var scale = MatQ(adaln_silu, 1, _p.AdalnEmbedDim,
                             $"{fl}.adaLN_modulation.1.weight",
                             $"{fl}.adaLN_modulation.1.bias", dim);
            TensorPrimitives.Add(scale.AsSpan(), 1f, scale.AsSpan()); // scale = 1 + scale

            // LayerNorm (elementwise_affine=False) + scale
            var normed = new float[nImg * dim];
            for (int t = 0; t < nImg; t++)
            {
                var row = imgHid.AsSpan(t * dim, dim);
                float mean = TensorPrimitives.Sum(row) / dim;
                TensorPrimitives.Subtract(row, mean, row);
                float varr = TensorPrimitives.Dot<float>(row, row) / dim;
                float rstd = 1f / MathF.Sqrt(varr + 1e-6f);
                var dst = normed.AsSpan(t * dim, dim);
                TensorPrimitives.Multiply(row, rstd,  dst);
                TensorPrimitives.Multiply(dst, scale, dst);
            }

            return MatQ(normed, nImg, dim, $"{fl}.linear.weight", $"{fl}.linear.bias", outDim);
        }
        finally { ArrayPool<float>.Shared.Return(adaln_silu); }
    }

    // ── Weight and matmul helpers ─────────────────────────────────────────

    /// <summary>
    /// Quantized matrix multiply using zero-copy mmap pointer from GGUF.
    /// When a GPU backend is provided and the batch is large enough, dequantizes
    /// the weight to float32 and uses the backend's batched SGEMM.
    /// Falls back to DiffusionOps.Linear for float32 safetensors backends.
    /// </summary>
    private unsafe float[] MatQ(float[] x, int n, int inDim, string wName, int outDim,
                                 string? bName = null)
    {
        var result = new float[n * outDim];

        if (_st.TryGetRaw(wName, out nint ptr, out long byteLen, out DType dtype, out int rows, out int cols))
        {
            if (_backend != null && n >= MinGpuBatch)
            {
                if (_backend.BestSgemmPrecision == SgemmPrecision.Fp8E4M3)
                {
                    // fp8 E4M3 GPU path (sm_89+): weights stored as fp8 (1 byte/element, 2× smaller than fp16).
                    // Both A (activations) and B (weights) must be fp8 for cublasGemmEx.
                    // cuBLAS requires bf16 output when inputs are fp8 — fp32 output is not supported.
                    // Weights are uploaded once on first call and cached; reused on subsequent steps.
                    int wCount = rows * cols;
                    byte[] xFp8 = ArrayPool<byte>.Shared.Rent(n * cols);
                    ushort[] cBf16Buf = ArrayPool<ushort>.Shared.Rent(n * rows);
                    try
                    {
                        // Convert activations to fp8
                        int xCount = n * cols;
                        for (int i = 0; i < xCount; i++)
                            xFp8[i] = Fp8Converter.FloatToFp8E4M3(x[i]);

                        // Get or upload the fp8 weight (cached after first call)
                        Core.Tensor wGpu;
                        if (_gpuWeightsFp8 != null && (dtype == DType.Q5_K || dtype == DType.Q4_K) &&
                            _gpuWeightsFp8.TryGetValue(wName, out var cachedW))
                        {
                            wGpu = cachedW;
                        }
                        else
                        {
                            float[] wBuf32 = ArrayPool<float>.Shared.Rent(wCount);
                            byte[] wFp8 = ArrayPool<byte>.Shared.Rent(wCount);
                            try
                            {
                                Dequantize.ToFloat32(
                                    new ReadOnlySpan<byte>((byte*)ptr, (int)byteLen),
                                    wBuf32.AsSpan(0, wCount), dtype, wCount);
                                for (int i = 0; i < wCount; i++)
                                    wFp8[i] = Fp8Converter.FloatToFp8E4M3(wBuf32[i]);
                                wGpu = _backend.UploadFp8(wFp8.AsSpan(0, wCount), TensorShape.D1(wCount));
                            }
                            finally
                            {
                                ArrayPool<float>.Shared.Return(wBuf32);
                                ArrayPool<byte>.Shared.Return(wFp8);
                            }
                            if (_gpuWeightsFp8 != null && (dtype == DType.Q5_K || dtype == DType.Q4_K))
                                _gpuWeightsFp8[wName] = wGpu;
                        }

                        bool ownW = _gpuWeightsFp8 == null || !_gpuWeightsFp8.ContainsKey(wName);
                        var xGpu = _backend.UploadFp8(xFp8.AsSpan(0, xCount), TensorShape.D1(xCount));
                        // fp8 GEMM output must be bf16 (cuBLAS restriction); convert to fp32 on download
                        var cGpu = _backend.Allocate(TensorShape.D1(n * rows), DType.BFloat16);
                        try
                        {
                            _backend.Sgemm(cGpu, xGpu, wGpu, n, cols, rows);
                            _backend.DownloadBf16(cGpu, cBf16Buf.AsSpan(0, n * rows));
                            int cCount = n * rows;
                            for (int i = 0; i < cCount; i++)
                            {
                                uint bits = (uint)cBf16Buf[i] << 16;
                                result[i] = BitConverter.UInt32BitsToSingle(bits);
                            }
                        }
                        finally
                        {
                            _backend.Free(xGpu);
                            if (ownW) _backend.Free(wGpu);
                            _backend.Free(cGpu);
                        }
                    }
                    finally
                    {
                        ArrayPool<byte>.Shared.Return(xFp8);
                        ArrayPool<ushort>.Shared.Return(cBf16Buf);
                    }
                }
                else if (_backend.BestSgemmPrecision == SgemmPrecision.Bf16)
                {
                    // bf16 GPU path: dequant weight to bf16, upload once and cache on GPU.
                    // Each weight matrix is uploaded only on first call; reused every subsequent step.
                    int wCount = rows * cols;
                    ushort[] xBf16 = ArrayPool<ushort>.Shared.Rent(n * cols);
                    ushort[] cBf16 = ArrayPool<ushort>.Shared.Rent(n * rows);
                    try
                    {
                        // Convert activations to bf16
                        int xCount = n * cols;
                        for (int i = 0; i < xCount; i++)
                        {
                            uint bits = BitConverter.SingleToUInt32Bits(x[i]);
                            xBf16[i] = (ushort)(bits >> 16);
                        }

                        // Get or upload the bf16 weight (cached after first call)
                        Core.Tensor wGpu;
                        if (_gpuWeightsBf16 != null && (dtype == DType.Q5_K || dtype == DType.Q4_K) &&
                            _gpuWeightsBf16.TryGetValue(wName, out var cachedW))
                        {
                            wGpu = cachedW;
                        }
                        else
                        {
                            float[] wBuf32 = ArrayPool<float>.Shared.Rent(wCount);
                            ushort[] wBf16 = ArrayPool<ushort>.Shared.Rent(wCount);
                            try
                            {
                                Dequantize.ToFloat32(
                                    new ReadOnlySpan<byte>((byte*)ptr, (int)byteLen),
                                    wBuf32.AsSpan(0, wCount), dtype, wCount);
                                for (int i = 0; i < wCount; i++)
                                {
                                    uint bits = BitConverter.SingleToUInt32Bits(wBuf32[i]);
                                    wBf16[i] = (ushort)(bits >> 16);
                                }
                                wGpu = _backend.UploadBf16(wBf16.AsSpan(0, wCount), TensorShape.D1(wCount));
                            }
                            finally
                            {
                                ArrayPool<float>.Shared.Return(wBuf32);
                                ArrayPool<ushort>.Shared.Return(wBf16);
                            }
                            if (_gpuWeightsBf16 != null && (dtype == DType.Q5_K || dtype == DType.Q4_K))
                                _gpuWeightsBf16[wName] = wGpu;
                        }

                        bool ownW = _gpuWeightsBf16 == null || !_gpuWeightsBf16.ContainsKey(wName);
                        var xGpu = _backend.UploadBf16(xBf16.AsSpan(0, xCount), TensorShape.D1(xCount));
                        var cGpu = _backend.Allocate(TensorShape.D1(n * rows), DType.BFloat16);
                        try
                        {
                            _backend.Sgemm(cGpu, xGpu, wGpu, n, cols, rows);
                            _backend.DownloadBf16(cGpu, cBf16.AsSpan(0, n * rows));
                        }
                        finally
                        {
                            _backend.Free(xGpu);
                            if (ownW) _backend.Free(wGpu);
                            _backend.Free(cGpu);
                        }

                        int cCount = n * rows;
                        for (int i = 0; i < cCount; i++)
                        {
                            uint bits = (uint)cBf16[i] << 16;
                            result[i] = BitConverter.UInt32BitsToSingle(bits);
                        }
                    }
                    finally
                    {
                        ArrayPool<ushort>.Shared.Return(xBf16);
                        ArrayPool<ushort>.Shared.Return(cBf16);
                    }
                }
                else if (_backend.BestSgemmPrecision == SgemmPrecision.Fp16 &&
                         _gpuWeights != null &&
                         (dtype == DType.Q5_K || dtype == DType.Q4_K))
                {
                    // GPU dequant path: upload raw quantized bytes once, dequant on GPU each call.
                    // SgemmF16: A=fp32 activations, B=fp16 weights, C=fp32 output (no fp16 overflow).
                    if (!_gpuWeights.TryGetValue(wName, out var rawGpu))
                    {
                        rawGpu = _backend.UploadRaw(
                            new ReadOnlySpan<byte>((byte*)ptr, (int)byteLen),
                            TensorShape.D1((long)byteLen), dtype);
                        _gpuWeights[wName] = rawGpu;
                    }

                    int numBlocks = rows * cols / 256;
                    int xCount   = n * cols;
                    int cCount   = n * rows;
                    var wGpu = _backend.Allocate(TensorShape.D1(rows * cols), DType.Float16);
                    try
                    {
                        if (dtype == DType.Q5_K)
                            _backend.DequantQ5KM(rawGpu, wGpu, numBlocks);
                        else
                            _backend.DequantQ4KM(rawGpu, wGpu, numBlocks);

                        // Upload activations as fp32 (not fp16) — avoids overflow for large intermediate values
                        // (e.g. SiLU(w1) * w3 can exceed fp16 range of 65504).
                        // SgemmF16 reads A as fp32, B as fp16 — weights get fp16 bandwidth savings only.
                        var xGpu = _backend.Upload(x.AsSpan(0, xCount), TensorShape.D1(xCount));
                        var cGpu = _backend.Allocate(TensorShape.D1(cCount), DType.Float32);
                        try
                        {
                            _backend.Sgemm(cGpu, xGpu, wGpu, n, cols, rows);
                            _backend.Download(cGpu, result.AsSpan());
                        }
                        finally
                        {
                            _backend.Free(xGpu);
                            _backend.Free(cGpu);
                        }
                    }
                    finally
                    {
                        _backend.Free(wGpu);
                    }
                }
                else if (_backend.BestSgemmPrecision == SgemmPrecision.Fp16)
                {
                    // fp16-weight GPU path: A=fp32 activations (no overflow), B=fp16 weights (bandwidth), C=fp32.
                    // Weights are dequantised once and cached on GPU, reused every denoising step.
                    int wCount = rows * cols;

                    Core.Tensor wGpu;
                    if (_gpuWeightsFp16 != null && (dtype == DType.Q5_K || dtype == DType.Q4_K) &&
                        _gpuWeightsFp16.TryGetValue(wName, out var cachedW16))
                    {
                        wGpu = cachedW16;
                    }
                    else
                    {
                        float[] wBuf32 = ArrayPool<float>.Shared.Rent(wCount);
                        Half[]  wHalf  = ArrayPool<Half>.Shared.Rent(wCount);
                        try
                        {
                            Dequantize.ToFloat32(
                                new ReadOnlySpan<byte>((byte*)ptr, (int)byteLen),
                                wBuf32.AsSpan(0, wCount), dtype, wCount);
                            for (int i = 0; i < wCount; i++) wHalf[i] = (Half)wBuf32[i];
                            wGpu = _backend.UploadHalf(wHalf.AsSpan(0, wCount), TensorShape.D1(wCount));
                        }
                        finally
                        {
                            ArrayPool<float>.Shared.Return(wBuf32);
                            ArrayPool<Half>.Shared.Return(wHalf);
                        }
                        if (_gpuWeightsFp16 != null && (dtype == DType.Q5_K || dtype == DType.Q4_K))
                            _gpuWeightsFp16[wName] = wGpu;
                    }

                    bool ownW16 = _gpuWeightsFp16 == null || !_gpuWeightsFp16.ContainsKey(wName);
                    int xCount16 = n * cols;
                    var xGpu16 = _backend.Upload(x.AsSpan(0, xCount16), TensorShape.D1(xCount16));
                    var cGpu16 = _backend.Allocate(TensorShape.D1(n * rows), DType.Float32);
                    try
                    {
                        _backend.Sgemm(cGpu16, xGpu16, wGpu, n, cols, rows);
                        _backend.Download(cGpu16, result.AsSpan());
                    }
                    finally
                    {
                        _backend.Free(xGpu16);
                        _backend.Free(cGpu16);
                        if (ownW16) _backend.Free(wGpu);
                    }
                }
                else
                {
                    // fp32 GPU path: dequantize weight on CPU, upload A + B, run SGEMM, download C
                    int wCount = rows * cols;
                    float[] wBuf = ArrayPool<float>.Shared.Rent(wCount);
                    try
                    {
                        Dequantize.ToFloat32(
                            new ReadOnlySpan<byte>((byte*)ptr, (int)byteLen),
                            wBuf.AsSpan(0, wCount), dtype, wCount);

                        var xGpu = _backend.Upload(x.AsSpan(0, n * cols), TensorShape.D1(n * cols));
                        var wGpu = _backend.Upload(wBuf.AsSpan(0, wCount), TensorShape.D1(wCount));
                        var cGpu = _backend.Allocate(TensorShape.D1(n * rows));
                        try
                        {
                            _backend.Sgemm(cGpu, xGpu, wGpu, n, cols, rows);
                            _backend.Download(cGpu, result.AsSpan());
                        }
                        finally
                        {
                            _backend.Free(xGpu);
                            _backend.Free(wGpu);
                            _backend.Free(cGpu);
                        }
                    }
                    finally { ArrayPool<float>.Shared.Return(wBuf); }
                }
            }
            else
            {
                // CPU path: fast pre-transposed Q4Kx8 if compatible, otherwise direct raw MatMulBatched
                _quantizedCache.Linear(wName, x.AsSpan(0, n * inDim), ReadOnlySpan<float>.Empty, result.AsSpan(), n, inDim, outDim);
            }
        }
        else
        {
            // Safetensors float32 path (VAE decoder, test mocks)
            DiffusionOps.Linear(x, _st.ReadF32(wName), null, n, inDim, outDim)
                        .AsSpan().CopyTo(result);
        }

        if (bName is not null)
        {
            float[]? bias;
            lock (_biasCache)
            {
                if (!_biasCache.TryGetValue(bName, out bias))
                {
                    bias = _st.ReadF32(bName);
                    _biasCache[bName] = bias;
                }
            }
            if (bias is not null)
            {
                for (int b = 0; b < n; b++)
                    TensorPrimitives.Add(result.AsSpan(b * outDim, outDim),
                                         bias.AsSpan(), result.AsSpan(b * outDim, outDim));
            }
        }

        return result;
    }

    private float[] MatQ(float[] x, int n, int inDim, string wName, string bName, int outDim)
        => MatQ(x, n, inDim, wName, outDim, bName);

    /// <summary>Read and cache a small weight (norm weights, q/k-norm weights).</summary>
    private float[] W(string name)
    {
        if (_cache.TryGetValue(name, out var c)) return c;
        var v = _st.ReadF32(name);
        _cache[name] = v;
        return v;
    }

    private static float[] Ones(int dim)
    {
        var v = new float[dim];
        v.AsSpan().Fill(1f);
        return v;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _residentGpuWorkspace?.Dispose();
            _residentGpuWeights?.Dispose();
            if (_gpuWeights != null && _backend != null)
            {
                foreach (var t in _gpuWeights.Values)
                    _backend.Free(t);
                _gpuWeights.Clear();
            }
            if (_gpuWeightsBf16 != null && _backend != null)
            {
                foreach (var t in _gpuWeightsBf16.Values)
                    _backend.Free(t);
                _gpuWeightsBf16.Clear();
            }
            if (_gpuWeightsFp16 != null && _backend != null)
            {
                foreach (var t in _gpuWeightsFp16.Values)
                    _backend.Free(t);
                _gpuWeightsFp16.Clear();
            }
            if (_gpuWeightsFp8 != null && _backend != null)
            {
                foreach (var t in _gpuWeightsFp8.Values)
                    _backend.Free(t);
                _gpuWeightsFp8.Clear();
            }
            _cache.Clear();
            lock (_biasCache) _biasCache.Clear();
            _quantizedCache.Dispose();
            _st.Dispose();
        }
    }
}
