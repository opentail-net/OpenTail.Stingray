
namespace OpenTail.Stingray.Diffusion.Flux2;

/// <summary>
/// FLUX.2 (Klein &amp; Kontext) Multi-Reference Diffusion Transformer forward pass.
/// Supports simultaneous conditioning on text prompts and multiple reference images.
/// </summary>
public sealed class Flux2DiT : IDisposable
{
    private readonly Flux2Params _p;
    private readonly IWeightLoader? _weights;
    private readonly string _prefix;
    private readonly QuantizedWeightCache? _cache;

    // GPU double-block residency (docs/091, wired 2026-09-19). Weights upload once and are
    // reused for the lifetime of this instance; the workspace is (re)created if the token shape
    // changes (a different resolution/prompt-length). Real, measured finding on this project's own
    // dev iGPU: CPU currently wins here (see docs/091/PerformanceLeague.md) -- wired anyway per
    // explicit operator instruction, since there is nothing to optimize further without a real,
    // exercised GPU code path in the actual generation pipeline to measure and improve.
    private Flux2GpuWeights? _gpuWeights;
    private Flux2GpuWorkspace? _gpuWorkspace;
    private int _gpuWsNImg = -1, _gpuWsNTxt = -1;

    public Flux2Params Params => _p;
    public QuantizedWeightCache? WeightCache => _cache;

    /// <summary>Structural-only constructor (no real weights) -- used by conformance tests that
    /// check RoPE orthogonality, shape flow, and data-flow order without real checkpoint math.
    /// <see cref="Forward"/> in this mode only applies RoPE/norm as a structural stand-in.</summary>
    public Flux2DiT(Flux2Params @params)
    {
        _p = @params ?? throw new ArgumentNullException(nameof(@params));
        _weights = null;
        _prefix = "";
        _cache = null;
    }

    /// <summary>Real weight-loading constructor. Tensor names/shapes confirmed against the real
    /// checkpoint and BFL source (docs/087) -- all linears are bias-free, modulation is SHARED
    /// (not per-block) across all double/single blocks of a given type.</summary>
    public Flux2DiT(IWeightLoader weights, Flux2Params @params, string prefix = "")
    {
        _weights = weights ?? throw new ArgumentNullException(nameof(weights));
        _p = @params ?? throw new ArgumentNullException(nameof(@params));
        _prefix = prefix;
        _cache = new QuantizedWeightCache(weights, prefix);
    }

    public void Dispose()
    {
        _cache?.Dispose();
        _gpuWorkspace?.Dispose();
        _gpuWeights?.Dispose();
    }

    private string Resolve(string name) => _prefix + name;

    /// <summary>
    /// Perf instrumentation (docs/088's FLUX.2 speed investigation, 2026-09-19): opt-in via
    /// <c>STINGRAY_FLUX2_PERF_TRACE=1</c>, zero overhead otherwise. Accumulates wall-clock time
    /// spent dequantizing vs. spent in the actual matmul, to answer "is this dequant-bound or
    /// compute-bound" before committing to a weight-residency or quantized-matmul fix.
    /// </summary>
    public static class PerfTrace
    {
        public static bool Enabled { get; } = Environment.GetEnvironmentVariable("STINGRAY_FLUX2_PERF_TRACE") == "1";
        public static long DequantTicks;
        public static long MatmulTicks;
        public static long DequantBytes;
        public static int DequantCalls;

        public static void Reset()
        {
            DequantTicks = 0; MatmulTicks = 0; DequantBytes = 0; DequantCalls = 0;
        }

        public static void Report(string label, QuantizedWeightCache? cache = null)
        {
            if (!Enabled) return;
            double dequantS = DequantTicks / (double)System.Diagnostics.Stopwatch.Frequency;
            double matmulS = MatmulTicks / (double)System.Diagnostics.Stopwatch.Frequency;
            double total = dequantS + matmulS;
            string cacheInfo = cache != null
                ? $", repacked={cache.CachedTensorCount} tensors ({cache.UsedBytes / 1024.0 / 1024.0:F1}MB / {cache.BudgetBytes / 1024.0 / 1024.0:F1}MB)"
                : "";
            Console.Error.WriteLine(
                $"[Flux2PerfTrace] {label}: dequant={dequantS:F2}s ({(total > 0 ? 100 * dequantS / total : 0):F1}%), " +
                $"matmul={matmulS:F2}s ({(total > 0 ? 100 * matmulS / total : 0):F1}%), " +
                $"dequantCalls={DequantCalls}, dequantBytes={DequantBytes / 1024.0 / 1024.0:F1}MB{cacheInfo}");
        }
    }

    /// <summary>
    /// Reads small auxiliary tensors (norm scales, 1D vectors) from the loader. Small tensors
    /// (&lt;=1M elements) are automatically cached in <see cref="GgufWeightLoader"/>.
    /// </summary>
    private float[] GetWeight(string name)
    {
        if (!PerfTrace.Enabled) return _weights!.ReadF32(Resolve(name));

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var w = _weights!.ReadF32(Resolve(name));
        sw.Stop();
        PerfTrace.DequantTicks += sw.ElapsedTicks;
        PerfTrace.DequantCalls++;
        PerfTrace.DequantBytes += (long)w.Length * sizeof(float);
        return w;
    }

    /// <summary>Bias-free linear projection -- confirmed empirically (docs/087): no `.bias` tensor
    /// exists anywhere in the real FLUX.2 checkpoint for any linear layer.
    /// Routes through <see cref="QuantizedWeightCache"/> when available, using native repacked Q4_K_X8
    /// SIMD matmuls without FP32 allocations.</summary>
    private float[] LinearNoBias(string weightName, ReadOnlySpan<float> x, int n, int inDim, int outDim)
    {
        var result = new float[n * outDim];
        if (_cache != null)
        {
            if (!PerfTrace.Enabled)
            {
                _cache.Linear(weightName, x, ReadOnlySpan<float>.Empty, result, n, inDim, outDim);
                return result;
            }

            var sw = System.Diagnostics.Stopwatch.StartNew();
            _cache.Linear(weightName, x, ReadOnlySpan<float>.Empty, result, n, inDim, outDim);
            sw.Stop();
            PerfTrace.MatmulTicks += sw.ElapsedTicks;
            return result;
        }

        var w = GetWeight(weightName);
        DiffusionOps.Linear(x, w, ReadOnlySpan<float>.Empty, result, n, inDim, outDim);
        return result;
    }

    /// <summary>
    /// Forward evaluation step predicting velocity for the target image latent, conditioned on
    /// text and (for now) implemented only for the no-reference-image case -- matches BFL's real
    /// base <c>Flux2.forward()</c> (examples/flux2/src/flux2/model.py), NOT the reference-image
    /// KV-cache path (<c>forward_kv_extract</c>/<c>causal_attn_fn</c>'s token-isolation attention).
    /// </summary>
    /// <exception cref="NotSupportedException">
    /// Thrown when reference images are supplied -- the real reference-token-isolation attention
    /// (<c>causal_attn_fn</c>) is a documented, not-yet-implemented gap (docs/087), and silently
    /// running the no-ref data flow on reference-image input would produce a plausible-looking but
    /// numerically wrong result rather than a clear error.
    /// </exception>
    public float[] Forward(
        float[] targetLatent, int[] targetPositions,
        IReadOnlyList<float[]>? refLatents, IReadOnlyList<int[]>? refPositions,
        float[] textEmbeds, int[] textPositions,
        float[] pooledEmbed,
        float timestep,
        float guidance = 3.5f,
        IComputeBackend? gpuBackend = null)
    {
        if (refLatents != null && refLatents.Count > 0)
        {
            throw new NotSupportedException(
                "Flux2DiT.Forward: reference-image conditioning is not yet implemented -- the real " +
                "reference-token-isolation attention (causal_attn_fn) is a documented gap, see " +
                "docs/087-flux2-implementation-plan.md. Running the plain no-ref data flow on " +
                "reference-image input would silently produce a wrong result instead of failing loudly.");
        }

        int nTarget = targetPositions.Length / 4;
        int nTxt = textPositions.Length / 4;
        int d = _p.HiddenSize;

        float[] vec = ComputeModulationVec(timestep, pooledEmbed, guidance);

        float[] img = _weights != null
            ? LinearNoBias("img_in.weight", targetLatent, nTarget, _p.InChannels, d)
            : ProjectTokens(targetLatent, nTarget, _p.InChannels, d);
        float[] txt = _weights != null
            ? LinearNoBias("txt_in.weight", textEmbeds, nTxt, _p.ContextInDim, d)
            : ProjectTokens(textEmbeds, nTxt, _p.ContextInDim, d);

        var (imgCos, imgSin) = Flux2RoPE.BuildContextFreqs(targetPositions, nTarget, _p.AxesDim, _p.Theta);
        var (txtCos, txtSin) = Flux2RoPE.BuildContextFreqs(textPositions, nTxt, _p.AxesDim, _p.Theta);

        // Shared modulation -- computed ONCE from vec, reused identically by every double/single
        // block of that type (real finding, docs/087 -- NOT per-block AdaLN like FLUX.1).
        float[][]? modImg = null, modTxt = null, modSingle = null;
        if (_weights != null)
        {
            modImg = ComputeModulation("double_stream_modulation_img.lin.weight", vec, d, 6);
            modTxt = ComputeModulation("double_stream_modulation_txt.lin.weight", vec, d, 6);
            modSingle = ComputeModulation("single_stream_modulation.lin.weight", vec, d, 3);
        }

        bool useGpuDoubleBlocks = _weights != null && gpuBackend is not null
            && gpuBackend is not CpuBackend
            && gpuBackend is IVisionOpsBackend && gpuBackend is IImageOpsBackend;

        if (useGpuDoubleBlocks)
        {
            RunDoubleBlocksGpu(gpuBackend!, img, txt, vec, targetPositions, textPositions, nTarget, nTxt);
        }
        else
        {
            for (int layer = 0; layer < _p.DepthDoubleBlocks; layer++)
            {
                if (_weights != null)
                    ApplyDoubleBlockReal(layer, img, txt, modImg!, modTxt!, imgCos, imgSin, txtCos, txtSin, nTarget, nTxt);
                else
                    ApplyDoubleBlock(layer, img, txt, vec, imgCos, imgSin, txtCos, txtSin, nTarget, nTxt);
            }
        }

        int nSeq = nTxt + nTarget;
        var unified = new float[nSeq * d];
        txt.AsSpan().CopyTo(unified.AsSpan(0, nTxt * d));
        img.AsSpan().CopyTo(unified.AsSpan(nTxt * d, nTarget * d));

        int headDim = _p.HeadDim;
        var unifiedCos = new float[nSeq * headDim];
        var unifiedSin = new float[nSeq * headDim];
        txtCos.AsSpan().CopyTo(unifiedCos.AsSpan(0, nTxt * headDim));
        txtSin.AsSpan().CopyTo(unifiedSin.AsSpan(0, nTxt * headDim));
        imgCos.AsSpan().CopyTo(unifiedCos.AsSpan(nTxt * headDim, nTarget * headDim));
        imgSin.AsSpan().CopyTo(unifiedSin.AsSpan(nTxt * headDim, nTarget * headDim));

        for (int layer = 0; layer < _p.DepthSingleBlocks; layer++)
        {
            if (_weights != null)
                ApplySingleBlockReal(layer, unified, modSingle!, unifiedCos, unifiedSin, nSeq);
            else
                ApplySingleBlock(layer, unified, vec, unifiedCos, unifiedSin, nSeq);
        }

        var velocity = new float[nTarget * _p.OutChannels];
        if (_weights != null)
        {
            var modFinal = ComputeModulation("final_layer.adaLN_modulation.1.weight", vec, d, 2);
            var (finalShift, finalScale) = (modFinal[0], modFinal[1]);
            var imgMod = ModulateCopy(unified.AsSpan(nTxt * d, nTarget * d).ToArray(), nTarget, d, finalShift, finalScale);
            var finalOut = LinearNoBias("final_layer.linear.weight", imgMod, nTarget, d, _p.OutChannels);
            finalOut.AsSpan().CopyTo(velocity);
        }
        else
        {
            ProjectOutput(unified.AsSpan(nTxt * d, nTarget * d), velocity, nTarget, d, _p.OutChannels);
        }

        return velocity;
    }

    /// <summary>Real timestep+guidance modulation vector -- confirmed against `model.py`'s
    /// `forward()`: `vec = time_in(timestep_embedding(t,256)); if guidance: vec += guidance_in(...)`.
    /// FLUX.2 has NO pooled CLIP conditioning (VecInDim=0) -- pooledEmbed is unused when real
    /// weights are present.</summary>
    internal float[] ComputeModulationVec(float timestep, float[] pooledEmbed, float guidance)
    {
        int d = _p.HiddenSize;
        if (_weights == null)
        {
            var vecStub = new float[d];
            for (int i = 0; i < Math.Min(pooledEmbed.Length, d); i++)
                vecStub[i] = pooledEmbed[i] + MathF.Sin(timestep * (i + 1)) * 0.1f;
            if (_p.GuidanceEmbed)
            {
                float gScale = (guidance - 1.0f) * 0.05f;
                for (int i = 0; i < d; i++) vecStub[i] += gScale;
            }
            return vecStub;
        }

        var tEmb = DiffusionOps.SinusoidalTimestepEmbedding(timestep, 256, 10000f, flipSinToCos: true);
        var vec = MlpEmbedder("time_in", tEmb, d);

        if (_p.GuidanceEmbed)
        {
            var gEmb = DiffusionOps.SinusoidalTimestepEmbedding(guidance, 256, 10000f, flipSinToCos: true);
            var gVec = MlpEmbedder("guidance_in", gEmb, d);
            for (int i = 0; i < d; i++) vec[i] += gVec[i];
        }

        return vec;
    }

    /// <summary>Real `MLPEmbedder`: Linear(in,hidden) -> SiLU -> Linear(hidden,hidden), bias-free.</summary>
    private float[] MlpEmbedder(string prefix, float[] x, int hidden)
    {
        var h = LinearNoBias($"{prefix}.in_layer.weight", x, 1, x.Length, hidden);
        DiffusionOps.SiluInPlace(h);
        return LinearNoBias($"{prefix}.out_layer.weight", h, 1, hidden, hidden);
    }

    /// <summary>Real `Modulation.forward`: SiLU(vec) -> Linear(dim, multiplier*dim) -> chunk into
    /// `multiplier` pieces of width `dim` each.</summary>
    internal float[][] ComputeModulation(string weightName, float[] vec, int dim, int multiplier)
    {
        var silu = (float[])vec.Clone();
        DiffusionOps.SiluInPlace(silu);
        var full = LinearNoBias(weightName, silu, 1, dim, multiplier * dim);
        var chunks = new float[multiplier][];
        for (int i = 0; i < multiplier; i++)
            chunks[i] = full.AsSpan(i * dim, dim).ToArray();
        return chunks;
    }

    private static float[] ProjectTokens(float[] input, int nTokens, int inDim, int outDim)
    {
        var output = new float[nTokens * outDim];
        int copyDim = Math.Min(inDim, outDim);

        for (int i = 0; i < nTokens; i++)
        {
            var src = input.AsSpan(i * inDim, copyDim);
            var dst = output.AsSpan(i * outDim, copyDim);
            src.CopyTo(dst);
        }
        return output;
    }

    /// <summary>
    /// Real DoubleStreamBlock (`model.py`'s `_prepare_qkv`/`_apply_residuals`): affine-free
    /// LayerNorm + AdaLN-modulate each stream separately, project SEPARATE img/txt QKV (own
    /// weights per stream), per-head RMSNorm Q/K, joint attention over the concatenated [txt,img]
    /// sequence (RoPE applied to Q/K AFTER the norm, per real `apply_rope(q,k,pe_full)` call order),
    /// split the attention output back per-stream, then a separate gated-residual + separate
    /// gated-FFN (SiLU-gated, mlp_ratio=3.0) per stream.
    /// </summary>
    /// <summary>Internal (not private) so <c>Flux2DoubleBlockGpuParityTests</c> can call it
    /// directly per-block, matching <see cref="DoubleBlockGpu"/>'s test structure exactly.</summary>
    internal void ApplyDoubleBlockReal(
        int layerIdx,
        float[] img, float[] txt,
        float[][] modImg, float[][] modTxt,
        float[] imgCos, float[] imgSin,
        float[] txtCos, float[] txtSin,
        int nTarget, int nTxt)
    {
        int d = _p.HiddenSize;
        int headDim = _p.HeadDim;
        int numHeads = _p.NumHeads;
        int mlpHidden = (int)(d * _p.MlpRatio);
        string bp = $"double_blocks.{layerIdx}.";

        var (imgShift1, imgScale1, imgGate1) = (modImg[0], modImg[1], modImg[2]);
        var (imgShift2, imgScale2, imgGate2) = (modImg[3], modImg[4], modImg[5]);
        var (txtShift1, txtScale1, txtGate1) = (modTxt[0], modTxt[1], modTxt[2]);
        var (txtShift2, txtScale2, txtGate2) = (modTxt[3], modTxt[4], modTxt[5]);

        var imgMod1 = ModulateCopy(img, nTarget, d, imgShift1, imgScale1);
        var txtMod1 = ModulateCopy(txt, nTxt, d, txtShift1, txtScale1);

        var imgQkv = LinearNoBias($"{bp}img_attn.qkv.weight", imgMod1, nTarget, d, 3 * d);
        var txtQkv = LinearNoBias($"{bp}txt_attn.qkv.weight", txtMod1, nTxt, d, 3 * d);

        var imgQ = new float[nTarget * d];
        var imgK = new float[nTarget * d];
        var imgV = new float[nTarget * d];
        for (int i = 0; i < nTarget; i++)
        {
            imgQkv.AsSpan(i * 3 * d, d).CopyTo(imgQ.AsSpan(i * d, d));
            imgQkv.AsSpan(i * 3 * d + d, d).CopyTo(imgK.AsSpan(i * d, d));
            imgQkv.AsSpan(i * 3 * d + 2 * d, d).CopyTo(imgV.AsSpan(i * d, d));
        }

        var txtQ = new float[nTxt * d];
        var txtK = new float[nTxt * d];
        var txtV = new float[nTxt * d];
        for (int i = 0; i < nTxt; i++)
        {
            txtQkv.AsSpan(i * 3 * d, d).CopyTo(txtQ.AsSpan(i * d, d));
            txtQkv.AsSpan(i * 3 * d + d, d).CopyTo(txtK.AsSpan(i * d, d));
            txtQkv.AsSpan(i * 3 * d + 2 * d, d).CopyTo(txtV.AsSpan(i * d, d));
        }

        var imgQNorm = GetWeight($"{bp}img_attn.norm.query_norm.scale");
        var imgKNorm = GetWeight($"{bp}img_attn.norm.key_norm.scale");
        var txtQNorm = GetWeight($"{bp}txt_attn.norm.query_norm.scale");
        var txtKNorm = GetWeight($"{bp}txt_attn.norm.key_norm.scale");
        DiffusionOps.RmsNorm(imgQ, imgQNorm, headDim);
        DiffusionOps.RmsNorm(imgK, imgKNorm, headDim);
        DiffusionOps.RmsNorm(txtQ, txtQNorm, headDim);
        DiffusionOps.RmsNorm(txtK, txtKNorm, headDim);

        int nSeq = nTxt + nTarget;
        var q = new float[nSeq * d];
        var k = new float[nSeq * d];
        var v = new float[nSeq * d];
        txtQ.AsSpan().CopyTo(q.AsSpan(0, nTxt * d));
        imgQ.AsSpan().CopyTo(q.AsSpan(nTxt * d, nTarget * d));
        txtK.AsSpan().CopyTo(k.AsSpan(0, nTxt * d));
        imgK.AsSpan().CopyTo(k.AsSpan(nTxt * d, nTarget * d));
        txtV.AsSpan().CopyTo(v.AsSpan(0, nTxt * d));
        imgV.AsSpan().CopyTo(v.AsSpan(nTxt * d, nTarget * d));

        var peCos = new float[nSeq * headDim];
        var peSin = new float[nSeq * headDim];
        txtCos.AsSpan().CopyTo(peCos.AsSpan(0, nTxt * headDim));
        txtSin.AsSpan().CopyTo(peSin.AsSpan(0, nTxt * headDim));
        imgCos.AsSpan().CopyTo(peCos.AsSpan(nTxt * headDim, nTarget * headDim));
        imgSin.AsSpan().CopyTo(peSin.AsSpan(nTxt * headDim, nTarget * headDim));

        Flux2RoPE.ApplyRoPE(q, peCos, peSin, nSeq, numHeads, headDim);
        Flux2RoPE.ApplyRoPE(k, peCos, peSin, nSeq, numHeads, headDim);

        var attn = DiffusionOps.MultiHeadAttention(q, k, v, nSeq, nSeq, numHeads, headDim);

        var txtAttn = attn.AsSpan(0, nTxt * d).ToArray();
        var imgAttn = attn.AsSpan(nTxt * d, nTarget * d).ToArray();

        var imgAttnOut = LinearNoBias($"{bp}img_attn.proj.weight", imgAttn, nTarget, d, d);
        var txtAttnOut = LinearNoBias($"{bp}txt_attn.proj.weight", txtAttn, nTxt, d, d);
        for (int i = 0; i < nTarget; i++)
            for (int c = 0; c < d; c++)
                img[i * d + c] += imgGate1[c] * imgAttnOut[i * d + c];
        for (int i = 0; i < nTxt; i++)
            for (int c = 0; c < d; c++)
                txt[i * d + c] += txtGate1[c] * txtAttnOut[i * d + c];

        var imgMod2 = ModulateCopy(img, nTarget, d, imgShift2, imgScale2);
        var txtMod2 = ModulateCopy(txt, nTxt, d, txtShift2, txtScale2);
        var imgFfn = GatedFfn($"{bp}img_mlp", imgMod2, nTarget, d, mlpHidden);
        var txtFfn = GatedFfn($"{bp}txt_mlp", txtMod2, nTxt, d, mlpHidden);
        for (int i = 0; i < nTarget; i++)
            for (int c = 0; c < d; c++)
                img[i * d + c] += imgGate2[c] * imgFfn[i * d + c];
        for (int i = 0; i < nTxt; i++)
            for (int c = 0; c < d; c++)
                txt[i * d + c] += txtGate2[c] * txtFfn[i * d + c];
    }

    /// <summary>
    /// Real SingleStreamBlock (`model.py`'s `SingleStreamBlock`): fused QKV+gated-MLP-up
    /// `linear1`, per-head RMSNorm Q/K, RoPE applied after norm, attention (real `causal_attn_fn`
    /// with `num_ref_tokens=0` degenerates to plain joint attention over the whole sequence for
    /// the no-ref case this method implements), `concat(attn, SiLU-gated-mlp)` through `linear2`,
    /// gated residual.
    /// </summary>
    private void ApplySingleBlockReal(int layerIdx, float[] unified, float[][] mod, float[] cos, float[] sin, int nSeq)
    {
        int d = _p.HiddenSize;
        int headDim = _p.HeadDim;
        int numHeads = _p.NumHeads;
        int mlpHidden = (int)(d * _p.MlpRatio);
        string bp = $"single_blocks.{layerIdx}.";
        var (shift, scale, gate) = (mod[0], mod[1], mod[2]);

        var xMod = ModulateCopy(unified, nSeq, d, shift, scale);
        var qkvMlp = LinearNoBias($"{bp}linear1.weight", xMod, nSeq, d, 3 * d + 2 * mlpHidden);

        var q = new float[nSeq * d];
        var k = new float[nSeq * d];
        var v = new float[nSeq * d];
        var mlp = new float[nSeq * 2 * mlpHidden];
        int rowWidth = 3 * d + 2 * mlpHidden;
        for (int i = 0; i < nSeq; i++)
        {
            qkvMlp.AsSpan(i * rowWidth, d).CopyTo(q.AsSpan(i * d, d));
            qkvMlp.AsSpan(i * rowWidth + d, d).CopyTo(k.AsSpan(i * d, d));
            qkvMlp.AsSpan(i * rowWidth + 2 * d, d).CopyTo(v.AsSpan(i * d, d));
            qkvMlp.AsSpan(i * rowWidth + 3 * d, 2 * mlpHidden).CopyTo(mlp.AsSpan(i * 2 * mlpHidden, 2 * mlpHidden));
        }

        var qNorm = GetWeight($"{bp}norm.query_norm.scale");
        var kNorm = GetWeight($"{bp}norm.key_norm.scale");
        DiffusionOps.RmsNorm(q, qNorm, headDim);
        DiffusionOps.RmsNorm(k, kNorm, headDim);

        Flux2RoPE.ApplyRoPE(q, cos, sin, nSeq, numHeads, headDim);
        Flux2RoPE.ApplyRoPE(k, cos, sin, nSeq, numHeads, headDim);

        var attn = DiffusionOps.MultiHeadAttention(q, k, v, nSeq, nSeq, numHeads, headDim);

        var mlpAct = new float[nSeq * mlpHidden];
        for (int i = 0; i < nSeq; i++)
        {
            var u1 = mlp.AsSpan(i * 2 * mlpHidden, mlpHidden);
            var u2 = mlp.AsSpan(i * 2 * mlpHidden + mlpHidden, mlpHidden);
            for (int c = 0; c < mlpHidden; c++)
                mlpAct[i * mlpHidden + c] = DiffusionOps.Silu(u1[c]) * u2[c];
        }

        var combined = new float[nSeq * (d + mlpHidden)];
        for (int i = 0; i < nSeq; i++)
        {
            attn.AsSpan(i * d, d).CopyTo(combined.AsSpan(i * (d + mlpHidden), d));
            mlpAct.AsSpan(i * mlpHidden, mlpHidden).CopyTo(combined.AsSpan(i * (d + mlpHidden) + d, mlpHidden));
        }

        var output = LinearNoBias($"{bp}linear2.weight", combined, nSeq, d + mlpHidden, d);
        for (int i = 0; i < nSeq; i++)
            for (int c = 0; c < d; c++)
                unified[i * d + c] += gate[c] * output[i * d + c];
    }

    /// <summary>Real SiLU-gated FFN (`img_mlp`/`txt_mlp`): up-proj to 2*mlpHidden, split, gate,
    /// down-proj. NOT a plain GELU MLP like FLUX.1.</summary>
    private float[] GatedFfn(string prefix, float[] x, int n, int d, int mlpHidden)
    {
        var up = LinearNoBias($"{prefix}.0.weight", x, n, d, 2 * mlpHidden);
        var gated = new float[n * mlpHidden];
        for (int i = 0; i < n; i++)
        {
            var u1 = up.AsSpan(i * 2 * mlpHidden, mlpHidden);
            var u2 = up.AsSpan(i * 2 * mlpHidden + mlpHidden, mlpHidden);
            for (int c = 0; c < mlpHidden; c++)
                gated[i * mlpHidden + c] = DiffusionOps.Silu(u1[c]) * u2[c];
        }
        return LinearNoBias($"{prefix}.2.weight", gated, n, mlpHidden, d);
    }

    /// <summary>Affine-free LayerNorm (eps=1e-6) then AdaLN-modulate: `(1+scale)*norm(x) + shift`,
    /// returned as a new array (does not mutate <paramref name="x"/>).</summary>
    private static float[] ModulateCopy(float[] x, int n, int d, float[] shift, float[] scale)
    {
        var normed = new float[n * d];
        DiffusionOps.LayerNormNoAffine(x, normed, d);
        var output = new float[n * d];
        DiffusionOps.ModulateRows(normed, output, n, d, shift, scale);
        return output;
    }

    // --- Structural (no-weights) placeholder path, unchanged for existing conformance tests ---

    private void ApplyDoubleBlock(
        int layerIdx,
        float[] img, float[] txt,
        float[] vec,
        float[] imgCos, float[] imgSin,
        float[] txtCos, float[] txtSin,
        int nTarget, int nTxt)
    {
        int d = _p.HiddenSize;
        RmsNorm(img, d);
        RmsNorm(txt, d);
        Flux2RoPE.ApplyRoPE(img, imgCos, imgSin, nTarget, _p.NumHeads, _p.HeadDim);
        Flux2RoPE.ApplyRoPE(txt, txtCos, txtSin, nTxt, _p.NumHeads, _p.HeadDim);
    }

    private void ApplySingleBlock(int layerIdx, float[] unified, float[] vec, float[] cos, float[] sin, int nSeq)
    {
        int d = _p.HiddenSize;
        RmsNorm(unified, d);
        Flux2RoPE.ApplyRoPE(unified, cos, sin, nSeq, _p.NumHeads, _p.HeadDim);
    }

    private static void ProjectOutput(ReadOnlySpan<float> hidden, Span<float> output, int nTokens, int hiddenDim, int outDim)
    {
        int copyDim = Math.Min(hiddenDim, outDim);
        for (int i = 0; i < nTokens; i++)
        {
            var src = hidden.Slice(i * hiddenDim, copyDim);
            var dst = output.Slice(i * outDim, copyDim);
            src.CopyTo(dst);
        }
    }

    private static void RmsNorm(Span<float> tensor, int dim) => DiffusionOps.RmsNormNoAffinePostSqrtEps(tensor, dim);

    // ── GPU double-block-only residency (docs/091, 2026-09-19) ─────────────────────────────

    /// <summary>
    /// Computes the SHARED double-stream modulation vectors (real finding, docs/087: ONE set,
    /// reused by every double block -- NOT per-block AdaLN like FLUX.1). Must be called once per
    /// step, before the double-block GPU loop. <paramref name="siluVecGpu"/> is `SiLU(vec)`
    /// pre-computed and uploaded by the caller (a `[1,d]` vector -- cheap enough on the CPU side
    /// that a dedicated GPU SiLU kernel isn't warranted for this single small vector, unlike the
    /// per-token gated-FFN activation which does need one, see <c>SiluGateMul</c>).
    /// </summary>
    internal void ComputeSharedDoubleModulationGpu(
        IVisionOpsBackend visionOps,
        Flux2GpuWorkspace ws,
        Flux2GpuWeights gw,
        OpenTail.Stingray.Core.Tensor siluVecGpu)
    {
        int d = _p.HiddenSize;
        visionOps.Sgemm(ws.ImgMod, siluVecGpu, gw.DoubleModImgWeight, 1, d, d * 6);
        visionOps.Sgemm(ws.TxtMod, siluVecGpu, gw.DoubleModTxtWeight, 1, d, d * 6);
    }

    /// <summary>
    /// Real DoubleStreamBlock GPU forward (mirrors <see cref="ApplyDoubleBlockReal"/>'s CPU math
    /// exactly): modulate img/txt separately using the SHARED <paramref name="ws"/>.ImgMod/TxtMod
    /// (computed once by <see cref="ComputeSharedDoubleModulationGpu"/>, not per-block) -> QKV
    /// projections -> per-stream QK-RMSNorm -> concat q/k/v as [txt, img] -> 4-axis RoPE on the
    /// FULL concatenated sequence (both streams are rotated in FLUX.2, unlike HunyuanVideo,
    /// confirmed 2026-09-19 against `model.py`'s `_prepare_qkv`) -> joint attention -> split back
    /// -> proj + gated residual -> re-modulate -> SiLU-gated FFN (<c>SiluGateMul</c>, not GEGLU) ->
    /// proj + gated residual. No bias terms anywhere (FLUX.2's linears are bias-free).
    /// </summary>
    /// <summary>
    /// Runs the 8 double-stream blocks on GPU (docs/091, wired 2026-09-19), mutating
    /// <paramref name="img"/>/<paramref name="txt"/> in place with the result -- everything before
    /// and after this call (img_in/txt_in projection, the 48 single-stream blocks, final layer)
    /// stays on the existing CPU path unchanged. GPU weights upload once and are cached for this
    /// instance's lifetime (<see cref="_gpuWeights"/>); the workspace is recreated only if the
    /// token shape changes. Real, measured finding on this project's own dev iGPU: CPU currently
    /// wins here (docs/091) -- this path exists so GPU is a real, selectable, exercised option to
    /// optimize further, not a dead code path nobody can measure or improve.
    /// </summary>
    private void RunDoubleBlocksGpu(
        IComputeBackend backend,
        float[] img, float[] txt,
        float[] vec,
        int[] targetPositions, int[] textPositions,
        int nImg, int nTxt)
    {
        var visionOps = (IVisionOpsBackend)backend;
        int d = _p.HiddenSize;
        int mlpHidden = (int)(d * _p.MlpRatio);

        _gpuWeights ??= new Flux2GpuWeights(backend, GetWeight, _p, includeSingleBlocks: false);

        if (_gpuWorkspace is null || _gpuWsNImg != nImg || _gpuWsNTxt != nTxt)
        {
            _gpuWorkspace?.Dispose();

            var combinedPositions = new int[(nTxt + nImg) * 4];
            Array.Copy(textPositions, 0, combinedPositions, 0, textPositions.Length);
            Array.Copy(targetPositions, 0, combinedPositions, textPositions.Length, targetPositions.Length);
            var (ropeCosCompact, ropeSinCompact) = Flux2RoPE.BuildContextFreqsCompact(combinedPositions, nTxt + nImg, _p.AxesDim, _p.Theta);

            _gpuWorkspace = new Flux2GpuWorkspace(backend, nImg, nTxt, d, mlpHidden, _p.HeadDim, ropeCosCompact, ropeSinCompact,
                initialImgHidden: img, initialTxtHidden: txt);
            _gpuWsNImg = nImg;
            _gpuWsNTxt = nTxt;
        }
        else
        {
            // Same shape as last step (fixed resolution/prompt length across a denoising loop) --
            // just re-upload this step's hidden state, keeping the RoPE table and scratch buffers.
            _gpuWorkspace.RefreshHidden(backend, img, txt);
        }

        var ws = _gpuWorkspace;
        var siluVec = (float[])vec.Clone();
        DiffusionOps.SiluInPlace(siluVec);
        var siluVecGpu = backend.Upload(siluVec, TensorShape.D1(d));

        try
        {
            var imageOps = (IImageOpsBackend)backend;
            ComputeSharedDoubleModulationGpu(visionOps, ws, _gpuWeights, siluVecGpu);

            // Batch-size sweep experiment (docs/093, Phase 0): STINGRAY_FLUX2_GPU_BATCH_SIZE
            // groups this many blocks per BeginBatch/EndBatch (one command buffer + one
            // submit+fence-wait per group) instead of the default 1 (every dispatch inside
            // DoubleBlockGpu submits+waits individually -- the current, un-optimized baseline).
            // This project's own history has BOTH a large real win (FLUX.1, 2-block chunking) and
            // a near-null result (Wan, ~2.5%) from batching -- measure for this workload, don't
            // assume either outcome (docs/093).
            int batchSize = int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_FLUX2_GPU_BATCH_SIZE"), out var bs) && bs > 0
                ? bs : 1;

            for (int i = 0; i < _p.DepthDoubleBlocks; i += batchSize)
            {
                int end = Math.Min(i + batchSize, _p.DepthDoubleBlocks);
                if (batchSize > 1) imageOps.BeginBatch();
                for (int layer = i; layer < end; layer++)
                    DoubleBlockGpu(ws, _gpuWeights.DoubleBlocks[layer], visionOps, imageOps);
                if (batchSize > 1) imageOps.EndBatch();
            }
            backend.Synchronize();

            backend.Download(ws.ImgHidden, img);
            backend.Download(ws.TxtHidden, txt);
        }
        finally
        {
            backend.Free(siluVecGpu);
        }
    }

    internal void DoubleBlockGpu(
        Flux2GpuWorkspace ws,
        Flux2GpuWeights.DoubleBlockGpuWeights bw,
        IVisionOpsBackend visionOps,
        IImageOpsBackend imageOps)
    {
        int d = _p.HiddenSize;
        int mlpHidden = (int)(d * _p.MlpRatio);
        int nImg = ws.NumImg, nTxt = ws.NumTxt, nSeq = ws.NumSeq;
        int nh = _p.NumHeads, hd = _p.HeadDim;

        // 1. Modulate (affine-free LayerNorm, isRmsNorm: false -- verified 2026-09-19,
        //    Flux2AdaLNModulateLayerNormGpuTests) using the shared shift1/scale1 (offsets 0/d).
        //    Dual-dispatch (docs/094 FLUX.2 GPU optimization wave): txt is stream A (rows
        //    [0, nTxt)), img is stream B (rows [nTxt, nTxt+nImg)) -- matches AttnOut's own
        //    [txt;img] row-major layout used later in this method.
        visionOps.AdaLNModulateDual(
            ws.NormedTxt, ws.TxtHidden, ws.TxtMod, shiftOffsetA: 0, scaleOffsetA: d,
            ws.NormedImg, ws.ImgHidden, ws.ImgMod, shiftOffsetB: 0, scaleOffsetB: d,
            nTokens: nSeq, streamSplit: nTxt, dim: d, isRmsNorm: false, eps: 1e-6f);

        // 2. QKV projections (separate weights per stream, bias-free).
        visionOps.Sgemm(ws.QkvImg, ws.NormedImg, bw.ImgAttnQkv, nImg, d, d * 3);
        visionOps.Sgemm(ws.QkvTxt, ws.NormedTxt, bw.TxtAttnQkv, nTxt, d, d * 3);

        // 3-5. Fused QKV unpack + per-stream QK-RMSNorm + full 4-axis RoPE (Flux2QkvNormRope).
        //      Replaces 5 separate dispatches (2 unpacks + 2 norms + 1 RoPE) with 2 fused dispatches,
        //      eliminating unnormalized and unrotated Q/K/V round-trips through GPU memory.
        visionOps.Flux2QkvNormRope(ws.QkvTxt, ws.Q, ws.K, ws.V, ws.RopeCos, ws.RopeSin,
            bw.TxtQkNormQScale, bw.TxtQkNormKScale, nTxt, nh, hd, dstTokenOffset: 0, eps: 1e-6f);
        visionOps.Flux2QkvNormRope(ws.QkvImg, ws.Q, ws.K, ws.V, ws.RopeCos, ws.RopeSin,
            bw.ImgQkNormQScale, bw.ImgQkNormKScale, nImg, nh, hd, dstTokenOffset: nTxt, eps: 1e-6f);

        // 6. Joint attention over the full sequence.
        imageOps.MultiHeadAttentionTiled(ws.AttnOut, ws.Q, ws.K, ws.V, nSeq, nSeq, nh, hd);

        // 7. Split back: txt occupies AttnOut's first nTxt rows directly (row-major contiguous);
        //    img starts at row nTxt, projected directly via row-offset Sgemm without FluxSliceImg copy.
        visionOps.Sgemm(ws.OutTxt, ws.AttnOut, bw.TxtAttnProjWeight, nTxt, d, d);
        visionOps.Sgemm(ws.OutImg, ws.AttnOut, bw.ImgAttnProjWeight, nImg, d, d, inputRowOffsetElements: nTxt * d);

        // 8. Gated residual (gate1, offset 2*d). Dual-dispatch: txt=A, img=B.
        visionOps.ScaleGateAddDual(
            ws.TxtHidden, ws.OutTxt, ws.TxtMod, gateOffsetA: 2 * d,
            ws.ImgHidden, ws.OutImg, ws.ImgMod, gateOffsetB: 2 * d,
            nTokens: nSeq, streamSplit: nTxt, dim: d);

        // 9. Re-modulate for the FFN (shift2/scale2, offsets 3*d/4*d). Dual-dispatch.
        visionOps.AdaLNModulateDual(
            ws.NormedTxt, ws.TxtHidden, ws.TxtMod, shiftOffsetA: 3 * d, scaleOffsetA: 4 * d,
            ws.NormedImg, ws.ImgHidden, ws.ImgMod, shiftOffsetB: 3 * d, scaleOffsetB: 4 * d,
            nTokens: nSeq, streamSplit: nTxt, dim: d, isRmsNorm: false, eps: 1e-6f);

        // 10. SiLU-gated FFN: up-project to 2*mlpHidden, then fused down-GEMM with on-the-fly SiLU activation.
        //     Eliminates 2 SiluGateMul dispatches and the intermediate ws.MlpGatedBuf round-trip.
        visionOps.Sgemm(ws.MlpUpBuf, ws.NormedTxt, bw.TxtMlp0Weight, nTxt, d, 2 * mlpHidden);
        visionOps.SgemmSiluGate(ws.OutTxt, ws.MlpUpBuf, bw.TxtMlp2Weight, nTxt, mlpHidden, d);

        visionOps.Sgemm(ws.MlpUpBuf, ws.NormedImg, bw.ImgMlp0Weight, nImg, d, 2 * mlpHidden);
        visionOps.SgemmSiluGate(ws.OutImg, ws.MlpUpBuf, bw.ImgMlp2Weight, nImg, mlpHidden, d);

        // 11. Gated residual (gate2, offset 5*d). Dual-dispatch.
        visionOps.ScaleGateAddDual(
            ws.TxtHidden, ws.OutTxt, ws.TxtMod, gateOffsetA: 5 * d,
            ws.ImgHidden, ws.OutImg, ws.ImgMod, gateOffsetB: 5 * d,
            nTokens: nSeq, streamSplit: nTxt, dim: d);
    }
}
