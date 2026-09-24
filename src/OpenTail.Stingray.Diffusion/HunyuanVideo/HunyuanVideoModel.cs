using System.Collections.Concurrent;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.HunyuanVideo;

/// <summary>
/// Native C# HunyuanVideo Dual-Stream and Single-Stream Diffusion Transformer (DiT).
/// Reference: stable-diffusion.cpp:src/model/diffusion/hunyuan.hpp:HunyuanVideoModel
/// </summary>
public sealed class HunyuanVideoModel : IDisposable
{
    private readonly IWeightLoader _weights;
    private readonly QuantizedWeightCache _quantizedCache;
    private readonly ConcurrentDictionary<string, float[]?> _biasCache = new(StringComparer.Ordinal);
    private readonly string _prefix;
    private readonly IComputeBackend? _backend;
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    private readonly Dictionary<string, CoreTensor>? _gpuWeights;
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _depthDouble;
    private readonly int _depthSingle;
    private bool _disposed;

    // CPU-path RoPE cache (docs/094 FLUX.2 GPU optimization wave's cross-model caching survey,
    // 2026-09-20): Compute3DRoPE depends only on (numFrames, patchH, patchW, headDim) -- fixed for
    // the whole generation -- but Forward() recomputed it via fresh trig evaluation on every single
    // denoising step. Same bug class already found and fixed in Flux2DiT (text-conditioning) and
    // WanModel (this exact same RoPE-table case) this same pass. NOTE: unlike Flux2DiT's
    // text-conditioning, HunyuanVideo's own `TokenRefiner` output is NOT cached here even though it
    // looks superficially similar -- it takes `timestep` as an input and genuinely varies per step
    // (real architectural difference, not an oversight).
    private readonly object _ropeCacheLock = new();
    private (int numFrames, int patchH, int patchW, int headDim)? _cachedRopeKey;
    private (float[] cos, float[] sin)? _cachedRope;

    public const int InChannels = 64;   // 16 * 2 * 2
    public const int OutChannels = 16;
    public const int TextDim = 4096;    // LLaMA-3 / Qwen2.5-VL text dimension

    public int Dim => _dim;
    public int NumHeads => _numHeads;
    public int DepthDouble => _depthDouble;
    public int DepthSingle => _depthSingle;

    /// <summary>True for guidance-distilled checkpoints (e.g. <c>hunyuan_video_720_cfgdistill</c>):
    /// guidance goes in through <c>guidance_in</c>, one forward per step, no true CFG.</summary>
    public bool HasGuidanceEmbedding => _weights.Contains(Resolve("guidance_in.mlp.0.weight"));

    public HunyuanVideoModel(
        IWeightLoader weights,
        string prefix = "",
        int dim = 3072,
        int numHeads = 24,
        int depthDouble = 20,
        int depthSingle = 0,
        IComputeBackend? backend = null)
    {
        _weights = weights;
        _quantizedCache = new QuantizedWeightCache(weights);
        _prefix = prefix;
        _backend = backend;
        (_dim, _numHeads, _depthDouble, _depthSingle) = DetectConfig(weights, prefix, dim, numHeads, depthDouble, depthSingle);
        _headDim = _dim / _numHeads;
        if (backend is not null)
            _gpuWeights = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
    }

    private static (int dim, int numHeads, int depthDouble, int depthSingle) DetectConfig(
        IWeightLoader weights,
        string prefix,
        int defDim,
        int defHeads,
        int defDouble,
        int defSingle)
    {
        int detectedDouble = defDouble;
        int detectedSingle = defSingle;

        for (int i = 60; i >= 0; i--)
        {
            string key1 = $"{prefix}double_blocks.{i}.img_attn.qkv.weight";
            string key2 = $"{prefix}double_blocks.{i}.img_attn_qkv.weight";
            if (weights.Contains(key1) || weights.Contains(key2) ||
                weights.Contains("model.diffusion_model." + key1) || weights.Contains("model.diffusion_model." + key2))
            {
                detectedDouble = i + 1;
                break;
            }
        }

        for (int i = 60; i >= 0; i--)
        {
            string key1 = $"{prefix}single_blocks.{i}.linear1.weight";
            if (weights.Contains(key1) || weights.Contains("model.diffusion_model." + key1))
            {
                detectedSingle = i + 1;
                break;
            }
        }

        int detectedDim = defDim;
        string inProjKey = $"{prefix}img_in.proj.weight";
        if (weights.Contains(inProjKey))
        {
            var w = weights.ReadF32(inProjKey);
            if (w.Length >= InChannels && w.Length % InChannels == 0)
                detectedDim = w.Length / InChannels;
        }

        int detectedHeads = detectedDim == 2048 ? 16 : defHeads;
        return (detectedDim, detectedHeads, detectedDouble, detectedSingle);
    }

    private string Resolve(string name)
    {
        string direct = _prefix + name;
        if (_weights.Contains(direct)) return direct;
        if (_weights.Contains("model.diffusion_model." + direct)) return "model.diffusion_model." + direct;
        if (_weights.Contains("diffusion_model." + direct)) return "diffusion_model." + direct;

        // Alternate naming variants between diffusers / ComfyUI / official Tencent formats
        string[] candidateReplacements =
        {
            direct.Replace("time_in.in_layer", "time_in.mlp.0"),
            direct.Replace("time_in.out_layer", "time_in.mlp.2"),
            direct.Replace("txt_in.in_layer", "txt_in.input_embedder"),
            direct.Replace("img_attn.qkv", "img_attn_qkv"),
            direct.Replace("img_attn.proj", "img_attn_proj"),
            direct.Replace("txt_attn.qkv", "txt_attn_qkv"),
            direct.Replace("txt_attn.proj", "txt_attn_proj"),
            direct.Replace("img_attn.norm.key_norm", "img_attn_k_norm"),
            direct.Replace("img_attn.norm.query_norm", "img_attn_q_norm"),
            direct.Replace("txt_attn.norm.key_norm", "txt_attn_k_norm"),
            direct.Replace("txt_attn.norm.query_norm", "txt_attn_q_norm"),
        };

        foreach (var cand in candidateReplacements)
        {
            if (_weights.Contains(cand)) return cand;
            if (_weights.Contains("model.diffusion_model." + cand)) return "model.diffusion_model." + cand;
            if (_weights.Contains("diffusion_model." + cand)) return "diffusion_model." + cand;
        }

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
    /// Evaluates the HunyuanVideo forward pass.
    /// </summary>
    /// <param name="pooledClip">CLIP-L <c>pooler_output</c> [768] for <c>vector_in</c>; null = omitted.</param>
    /// <param name="guidance">Distilled guidance, already x1000 (diffusers: <c>guidance_scale * 1000</c>),
    /// for the <c>guidance_in</c> embedder of guidance-distilled checkpoints; null = omitted.</param>
    public float[] Forward(
        float[] latent,
        float timestep,
        float[] textContext,
        int numFrames,
        int latH,
        int latW,
        float[]? pooledClip = null,
        float? guidance = null)
    {
        int patchH = latH / 2;
        int patchW = latW / 2;
        int numImgTokens = numFrames * patchH * patchW;
        int numTxtTokens = textContext.Length / TextDim;

        // 1. Pack 16-channel video latent into 64-channel patches
        var packed = PackLatents(latent, numFrames, latH, latW);

        // 2. Input projections
        var imgTokens = Linear("img_in.proj", packed, InChannels, _dim);
        var txtTokens = TokenRefiner(textContext, timestep, numTxtTokens);

        // 3. Modulation vector: vec = time_in(t) + vector_in(CLIP-L pooled) + guidance_in(g).
        // Reference: stable-diffusion.cpp hunyuan.hpp forward (and diffusers
        // HunyuanVideoConditionEmbedding). Until 2026-09-24 only time_in was used, so every
        // block's AdaLN modulation was missing the text and guidance terms.
        var tEmb = ComputeTimestepEmbedding(timestep);
        if (pooledClip is not null && TryGetWeight("vector_in.in_layer.weight") is not null)
        {
            var v0 = Linear("vector_in.in_layer", pooledClip, pooledClip.Length, _dim);
            DiffusionOps.SiluInPlace(v0);
            var v1 = Linear("vector_in.out_layer", v0, _dim, _dim);
            for (int d = 0; d < _dim; d++) tEmb[d] += v1[d];
        }
        if (guidance is float g && HasGuidanceEmbedding)
        {
            var gEmb = DiffusionOps.SinusoidalTimestepEmbedding(g, flipSinToCos: true);
            var g0 = Linear("guidance_in.mlp.0", gEmb, 256, _dim);
            DiffusionOps.SiluInPlace(g0);
            var g1 = Linear("guidance_in.mlp.2", g0, _dim, _dim);
            for (int d = 0; d < _dim; d++) tEmb[d] += g1[d];
        }

        // 4. 3D-RoPE positional frequencies (cached across steps -- see _cachedRope's own doc
        //    comment; cos/sin are read-only ApplyRoPE inputs, safe to share across calls).
        float[] cos, sin;
        var ropeKey = (numFrames, patchH, patchW, _headDim);
        lock (_ropeCacheLock)
        {
            if (_cachedRopeKey == ropeKey && _cachedRope is not null)
            {
                (cos, sin) = _cachedRope.Value;
            }
            else
            {
                (cos, sin) = HunyuanVideoRoPE.Compute3DRoPE(numFrames, patchH, patchW, _headDim);
                _cachedRopeKey = ropeKey;
                _cachedRope = (cos, sin);
            }
        }

        bool dumpDiag = Environment.GetEnvironmentVariable("STINGRAY_HUNYUAN_DUMP_LATENT") == "1";
        if (dumpDiag)
        {
            Console.Error.WriteLine($"[HunyuanVideo.Forward] imgTokens nan={imgTokens.Count(v => !float.IsFinite(v))}/{imgTokens.Length} txtTokens nan={txtTokens.Count(v => !float.IsFinite(v))}/{txtTokens.Length} tEmb nan={tEmb.Count(v => !float.IsFinite(v))}/{tEmb.Length} cos nan={cos.Count(v => !float.IsFinite(v))} sin nan={sin.Count(v => !float.IsFinite(v))}");
        }

        // 5. Dual-Stream Blocks (double_blocks)
        for (int b = 0; b < _depthDouble; b++)
        {
            string p = $"double_blocks.{b}";
            (imgTokens, txtTokens) = DoubleBlock(p, imgTokens, txtTokens, tEmb, cos, sin, numImgTokens, numTxtTokens);
            if (dumpDiag)
            {
                int imgNan = imgTokens.Count(v => !float.IsFinite(v));
                int txtNan = txtTokens.Count(v => !float.IsFinite(v));
                if (imgNan > 0 || txtNan > 0)
                    Console.Error.WriteLine($"[HunyuanVideo.Forward] after double_blocks.{b}: imgTokens nan={imgNan}/{imgTokens.Length} txtTokens nan={txtNan}/{txtTokens.Length}");
            }
        }

        // 6. Single-Stream Blocks (single_blocks) if present
        if (_depthSingle > 0)
        {
            // Real reference (HunyuanVideoAttnProcessor2_0, `add_q_proj is None` branch) concatenates
            // [hidden_states(img), encoder_hidden_states(txt)] -- IMAGE FIRST, TEXT SECOND -- and
            // applies RoPE only to the image prefix (`query[:, :-txt_len]`), leaving text tokens
            // unrotated. This previously concatenated txt-first, which put RoPE's cos/sin table
            // (sized for numImgTokens rows only) over the wrong prefix -- rotating (mostly/entirely)
            // TEXT tokens with image positional frequencies while the real image tokens got NO RoPE
            // at all, discarding all spatial/temporal structure. Fixed 2026-09-19 (docs/088).
            var singleTokens = ConcatSequences(imgTokens, txtTokens, numImgTokens, numTxtTokens);
            int totalSeq = numTxtTokens + numImgTokens;
            for (int b = 0; b < _depthSingle; b++)
            {
                string p = $"single_blocks.{b}";
                singleTokens = SingleBlock(p, singleTokens, tEmb, cos, sin, numImgTokens, totalSeq);
            }
            imgTokens = singleTokens.AsSpan(0, numImgTokens * _dim).ToArray();
        }

        // 7. Final Layer: chunks are [shift, scale] and the modulation is LN(x)*(1+scale)+shift with
        // an affine-free LayerNorm (reference: Flux::LastLayer, used by hunyuan.hpp; the original
        // checkpoint's order -- diffusers' converter swaps it). This previously used the shift
        // half as the multiplier (no +1) and the scale half as the offset.
        var headMod = Linear("final_layer.adaLN_modulation.1", DiffusionOpsSilu(tEmb), _dim, _dim * 2);
        var headShift = headMod.AsSpan(0, _dim);
        var headScale = headMod.AsSpan(_dim, _dim);

        var normed = (float[])imgTokens.Clone();
        DiffusionOps.LayerNormNoAffine(normed, _dim);
        normed = Modulate(normed, numImgTokens, headShift, headScale);

        var outPacked = Linear("final_layer.linear", normed, _dim, InChannels);

        // 8. Unpack patches [numTokens, 64] -> [16, numFrames, latH, latW]
        return UnpackLatents(outPacked, numFrames, latH, latW);
    }

    private (float[] imgOut, float[] txtOut) DoubleBlock(
        string prefix,
        float[] img,
        float[] txt,
        float[] tEmb,
        float[] cos,
        float[] sin,
        int numImg,
        int numTxt)
    {
        var imgMod = Linear($"{prefix}.img_mod.linear", DiffusionOpsSilu(tEmb), _dim, _dim * 6);
        var txtMod = Linear($"{prefix}.txt_mod.linear", DiffusionOpsSilu(tEmb), _dim, _dim * 6);

        var imgS1 = imgMod.AsSpan(0 * _dim, _dim);
        var imgSc1 = imgMod.AsSpan(1 * _dim, _dim);
        var imgG1 = imgMod.AsSpan(2 * _dim, _dim);
        var imgS2 = imgMod.AsSpan(3 * _dim, _dim);
        var imgSc2 = imgMod.AsSpan(4 * _dim, _dim);
        var imgG2 = imgMod.AsSpan(5 * _dim, _dim);

        var txtS1 = txtMod.AsSpan(0 * _dim, _dim);
        var txtSc1 = txtMod.AsSpan(1 * _dim, _dim);
        var txtG1 = txtMod.AsSpan(2 * _dim, _dim);
        var txtS2 = txtMod.AsSpan(3 * _dim, _dim);
        var txtSc2 = txtMod.AsSpan(4 * _dim, _dim);
        var txtG2 = txtMod.AsSpan(5 * _dim, _dim);

        // Real AdaLN-Zero requires an affine-free LayerNorm BEFORE the scale/shift modulation --
        // same "LayerNormNoAffine -> Modulate" pattern WanModel's CPU path uses (see its
        // ApplyGatedResidualRows doc comment: this Modulate/gating logic is a byte-identical
        // extraction shared with Wan). Modulating the raw, unnormalized residual stream directly
        // (as this previously did) leaves img/txt growing unboundedly across blocks via the gated
        // residual adds below, with no renormalization -- confirmed to overflow into NaN by
        // double_blocks.8-9 of 20 on the real fp8 checkpoint (2026-09-18 diagnostic).
        var normedImg1 = (float[])img.Clone();
        DiffusionOps.LayerNormNoAffine(normedImg1, _dim);
        normedImg1 = Modulate(normedImg1, numImg, imgS1, imgSc1);
        var normedTxt1 = (float[])txt.Clone();
        DiffusionOps.LayerNormNoAffine(normedTxt1, _dim);
        normedTxt1 = Modulate(normedTxt1, numTxt, txtS1, txtSc1);

        var (imgAttn, txtAttn) = JointAttention($"{prefix}", normedImg1, normedTxt1, cos, sin, numImg, numTxt);

        ApplyGatedResidual(img, imgAttn, numImg, imgG1);
        ApplyGatedResidual(txt, txtAttn, numTxt, txtG1);

        var normedImg2 = (float[])img.Clone();
        DiffusionOps.LayerNormNoAffine(normedImg2, _dim);
        normedImg2 = Modulate(normedImg2, numImg, imgS2, imgSc2);
        var normedTxt2 = (float[])txt.Clone();
        DiffusionOps.LayerNormNoAffine(normedTxt2, _dim);
        normedTxt2 = Modulate(normedTxt2, numTxt, txtS2, txtSc2);

        var imgMlp = FeedForward($"{prefix}.img_mlp", normedImg2, numImg);
        var txtMlp = FeedForward($"{prefix}.txt_mlp", normedTxt2, numTxt);

        ApplyGatedResidual(img, imgMlp, numImg, imgG2);
        ApplyGatedResidual(txt, txtMlp, numTxt, txtG2);

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

        var imgQkv = Linear($"{prefix}.img_attn.qkv", img, _dim, _dim * 3);
        var txtQkv = Linear($"{prefix}.txt_attn.qkv", txt, _dim, _dim * 3);

        var (imgQ, imgK, imgV) = SplitQkv(imgQkv, numImg, _dim);
        var (txtQ, txtK, txtV) = SplitQkv(txtQkv, numTxt, _dim);

        // Real checkpoint carries BOTH q_norm and k_norm per attention (img_attn_q_norm/k_norm,
        // txt_attn_q_norm/k_norm) -- QK-RMSNorm, same convention as Qwen3/SD3.5. Applying only
        // K-norm (as this previously did) leaves Q magnitude unbounded across blocks, which was
        // found to overflow attention scores into NaN by block 8 of 20 double_blocks on the real
        // fp8 checkpoint (STINGRAY_HUNYUAN_DUMP_LATENT=1 diagnostic, 2026-09-18).
        var normImgQ = TryGetWeight($"{prefix}.img_attn.norm.query_norm.weight") ?? TryGetWeight($"{prefix}.img_attn.norm.query_norm.scale");
        if (normImgQ is not null) RmsNormHeads(imgQ, numImg, _numHeads, _headDim, normImgQ);
        var normImgK = TryGetWeight($"{prefix}.img_attn.norm.key_norm.weight") ?? TryGetWeight($"{prefix}.img_attn.norm.key_norm.scale");
        if (normImgK is not null) RmsNormHeads(imgK, numImg, _numHeads, _headDim, normImgK);
        var normTxtQ = TryGetWeight($"{prefix}.txt_attn.norm.query_norm.weight") ?? TryGetWeight($"{prefix}.txt_attn.norm.query_norm.scale");
        if (normTxtQ is not null) RmsNormHeads(txtQ, numTxt, _numHeads, _headDim, normTxtQ);
        var normTxtK = TryGetWeight($"{prefix}.txt_attn.norm.key_norm.weight") ?? TryGetWeight($"{prefix}.txt_attn.norm.key_norm.scale");
        if (normTxtK is not null) RmsNormHeads(txtK, numTxt, _numHeads, _headDim, normTxtK);

        // Real reference applies RoPE to the img q/k BEFORE concatenating with txt (see
        // HunyuanVideoAttnProcessor2_0's `else` branch, taken when `add_q_proj is not None` --
        // the double-block case): text tokens never receive RoPE at all. Rotate img's own q/k
        // first (img is a standalone numImg-length buffer here), then concatenate img-first,
        // txt-second to match the reference's `torch.cat([query, encoder_query], dim=1)` and
        // `hidden_states[:, :-txt_len]` / `hidden_states[:, -txt_len:]` output split. Previously
        // this concatenated txt-first and RoPE'd the resulting prefix -- rotating text tokens
        // with image positional frequencies while leaving the real image tokens unrotated. Fixed
        // 2026-09-19 (docs/088).
        int ropeSeq = Math.Min(numImg, cos.Length / _headDim);
        HunyuanVideoRoPE.ApplyRoPE(imgQ, cos, sin, ropeSeq, _numHeads, _headDim);
        HunyuanVideoRoPE.ApplyRoPE(imgK, cos, sin, ropeSeq, _numHeads, _headDim);

        var q = ConcatSequences(imgQ, txtQ, numImg, numTxt);
        var k = ConcatSequences(imgK, txtK, numImg, numTxt);
        var v = ConcatSequences(imgV, txtV, numImg, numTxt);

        var attnOut = MultiHeadAttention(q, k, v, totalSeq, _numHeads, _headDim);

        var imgAttnSlice = attnOut.AsSpan(0, numImg * _dim).ToArray();
        var txtAttnSlice = attnOut.AsSpan(numImg * _dim, numTxt * _dim).ToArray();

        var finalImg = Linear($"{prefix}.img_attn.proj", imgAttnSlice, _dim, _dim);
        var finalTxt = Linear($"{prefix}.txt_attn.proj", txtAttnSlice, _dim, _dim);

        return (finalImg, finalTxt);
    }

    private float[] SingleBlock(
        string prefix,
        float[] x,
        float[] tEmb,
        float[] cos,
        float[] sin,
        int numImg,
        int totalSeq)
    {
        var mod = Linear($"{prefix}.modulation.linear", DiffusionOpsSilu(tEmb), _dim, _dim * 3);
        var s = mod.AsSpan(0 * _dim, _dim);
        var sc = mod.AsSpan(1 * _dim, _dim);
        var g = mod.AsSpan(2 * _dim, _dim);

        // Same missing-pre-norm bug as DoubleBlock (see its comment) -- affine-free LayerNorm
        // before the AdaLN modulation, not a direct modulate of the raw residual stream.
        var normed = (float[])x.Clone();
        DiffusionOps.LayerNormNoAffine(normed, _dim);
        normed = Modulate(normed, totalSeq, s, sc);

        // QKV + MLP in single linear1
        int mlpHidden = _dim * 4;
        int linear1Out = _dim * 3 + mlpHidden;
        var preAttnMlp = Linear($"{prefix}.linear1", normed, _dim, linear1Out);

        // Split along the LAST dim, per token (reference: torch.split(linear1(x), [3*hidden,
        // mlp_hidden], dim=-1)). This used to take the first totalSeq*3*dim floats of the whole
        // row-major buffer as "qkv", scrambling q/k/v and the MLP input across tokens in all 40
        // single blocks (found 2026-09-24).
        var qkv = new float[totalSeq * _dim * 3];
        var mlpIn = new float[totalSeq * mlpHidden];
        for (int tok = 0; tok < totalSeq; tok++)
        {
            Array.Copy(preAttnMlp, tok * linear1Out, qkv, tok * _dim * 3, _dim * 3);
            Array.Copy(preAttnMlp, tok * linear1Out + _dim * 3, mlpIn, tok * mlpHidden, mlpHidden);
        }

        var (q, k, v) = SplitQkv(qkv, totalSeq, _dim);

        // Same missing-Q-norm bug as JointAttention (see its comment) -- single_blocks.N.q_norm.weight
        // is a real tensor in the checkpoint, confirmed via direct tensor-name enumeration.
        var normQ = TryGetWeight($"{prefix}.q_norm.weight") ?? TryGetWeight($"{prefix}.q_norm.scale");
        if (normQ is not null) RmsNormHeads(q, totalSeq, _numHeads, _headDim, normQ);
        var normK = TryGetWeight($"{prefix}.k_norm.weight") ?? TryGetWeight($"{prefix}.k_norm.scale");
        if (normK is not null) RmsNormHeads(k, totalSeq, _numHeads, _headDim, normK);

        // Real reference (line 82-96's `add_q_proj is None` branch, the single-block case): RoPE
        // is applied only to the image-token prefix of the concatenated [img, txt] sequence; text
        // tokens (the tail) are left unrotated. x/q/k here are already ordered img-first (see the
        // Forward() call site), so restricting to the numImg prefix matches the reference exactly.
        int ropeSeq = Math.Min(numImg, cos.Length / _headDim);
        HunyuanVideoRoPE.ApplyRoPE(q, cos, sin, ropeSeq, _numHeads, _headDim);
        HunyuanVideoRoPE.ApplyRoPE(k, cos, sin, ropeSeq, _numHeads, _headDim);

        var attnOut = MultiHeadAttention(q, k, v, totalSeq, _numHeads, _headDim);

        // MLP activation
        DiffusionOps.GeluInPlace(mlpIn);

        // Linear2 projects [attnOut (dim) + mlpIn (4*dim)] -> dim
        var combined = ConcatFeatures(attnOut, mlpIn, totalSeq, _dim, mlpHidden);
        var blockOut = Linear($"{prefix}.linear2", combined, _dim + mlpHidden, _dim);

        ApplyGatedResidual(x, blockOut, totalSeq, g);
        return x;
    }

    private static float[] ConcatFeatures(float[] a, float[] b, int seqLen, int dimA, int dimB)
    {
        int outDim = dimA + dimB;
        var res = new float[seqLen * outDim];
        for (int i = 0; i < seqLen; i++)
        {
            Array.Copy(a, i * dimA, res, i * outDim, dimA);
            Array.Copy(b, i * dimB, res, i * outDim + dimA, dimB);
        }
        return res;
    }

    private static (float[] q, float[] k, float[] v) SplitQkv(float[] qkv, int seqLen, int dim)
    {
        var q = new float[seqLen * dim];
        var k = new float[seqLen * dim];
        var v = new float[seqLen * dim];

        for (int i = 0; i < seqLen; i++)
        {
            int qkvOff = i * dim * 3;
            Array.Copy(qkv, qkvOff, q, i * dim, dim);
            Array.Copy(qkv, qkvOff + dim, k, i * dim, dim);
            Array.Copy(qkv, qkvOff + dim * 2, v, i * dim, dim);
        }
        return (q, k, v);
    }

    private static void RmsNormHeads(float[] x, int seqLen, int numHeads, int headDim, ReadOnlySpan<float> weight)
    {
        for (int s = 0; s < seqLen; s++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int off = (s * numHeads + h) * headDim;
                float sumSq = 0f;
                for (int d = 0; d < headDim; d++)
                {
                    float val = x[off + d];
                    sumSq += val * val;
                }
                float rms = 1.0f / MathF.Sqrt(sumSq / headDim + 1e-6f);
                for (int d = 0; d < headDim; d++)
                    x[off + d] = x[off + d] * rms * (weight.Length > d ? weight[d] : 1.0f);
            }
        }
    }

    private static float[] MultiHeadAttention(float[] q, float[] k, float[] v, int seqLen, int numHeads, int headDim)
    {
        int dim = numHeads * headDim;
        var outF = new float[seqLen * dim];
        Wan.WanAttention.TiledMultiHeadAttention(q, k, v, outF.AsSpan(), seqLen, seqLen, numHeads, headDim);
        return outF;
    }

    private float[] FeedForward(string prefix, float[] x, int seqLen)
    {
        int mlpHidden = _dim * 4;
        var fc1 = Linear($"{prefix}.fc1", x, _dim, mlpHidden);
        DiffusionOps.GeluInPlace(fc1);
        return Linear($"{prefix}.fc2", fc1, mlpHidden, _dim);
    }

    private static float[] ConcatSequences(float[] a, float[] b, int lenA, int lenB)
    {
        var res = new float[a.Length + b.Length];
        Array.Copy(a, 0, res, 0, a.Length);
        Array.Copy(b, 0, res, a.Length, b.Length);
        return res;
    }

    private float[] Modulate(float[] x, int seqLen, ReadOnlySpan<float> shift, ReadOnlySpan<float> scale)
        => DiffusionOps.ModulateRows(x, seqLen, _dim, shift, scale);

    private void ApplyGatedResidual(float[] x, float[] branch, int seqLen, ReadOnlySpan<float> gate)
        => DiffusionOps.ApplyGatedResidualRows(x, branch, seqLen, _dim, gate);

    private float[] ComputeTimestepEmbedding(float timestep)
    {
        // REAL BUG FOUND AND FIXED 2026-09-18 (docs/088's HunyuanVideo noise investigation):
        // confirmed against the real reference (transformer_hunyuan_video.py's
        // `self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, ...)`) -- missing
        // `flipSinToCos: true` produced the wrong `[sin,cos]` order instead of real `[cos,sin]`.
        // Same bug class already found and fixed for Wan, LTX-Video, and Qwen Image this session.
        var emb = DiffusionOps.SinusoidalTimestepEmbedding(timestep, flipSinToCos: true);
        var t0 = Linear("time_in.in_layer", emb, 256, _dim);
        DiffusionOps.SiluInPlace(t0);
        return Linear("time_in.out_layer", t0, _dim, _dim);
    }

    private int? _refinerDepth;

    /// <summary>
    /// Real `HunyuanVideoTokenRefiner`: pools the raw text hidden states, combines with a
    /// timestep embedding to form a per-block AdaLN gate signal, projects the raw text tokens into
    /// model dim, then refines them through `HunyuanVideoIndividualTokenRefiner`'s self-attention +
    /// FFN blocks (gated, no shift/scale modulation on the norms themselves -- only the residual
    /// gates are conditioned). Reference: `examples/diffusers/.../transformer_hunyuan_video.py`
    /// `HunyuanVideoTokenRefiner`/`HunyuanVideoIndividualTokenRefinerBlock`. Real checkpoint keys
    /// (`txt_in.t_embedder`/`txt_in.c_embedder`/`txt_in.input_embedder`/
    /// `txt_in.individual_token_refiner.blocks.N.*`) confirmed directly against the real
    /// safetensors header -- 2 refiner blocks present in this checkpoint.
    /// </summary>
    private float[] TokenRefiner(float[] textContext, float timestep, int seqLen)
    {
        // Pooled projection: mean over sequence of the RAW (pre-projection) text hidden states.
        var pooled = new float[TextDim];
        for (int s = 0; s < seqLen; s++)
            for (int d = 0; d < TextDim; d++)
                pooled[d] += textContext[s * TextDim + d];
        for (int d = 0; d < TextDim; d++) pooled[d] /= seqLen;

        // Same flipSinToCos fix as ComputeTimestepEmbedding above -- this is a separate call site
        // (token refiner's own timestep embedder) using the same real Timesteps(flip_sin_to_cos=True).
        var tEmb = DiffusionOps.SinusoidalTimestepEmbedding(timestep, flipSinToCos: true);
        var t0 = Linear("txt_in.t_embedder.mlp.0", tEmb, 256, _dim);
        DiffusionOps.SiluInPlace(t0);
        var tOut = Linear("txt_in.t_embedder.mlp.2", t0, _dim, _dim);

        var c0 = Linear("txt_in.c_embedder.linear_1", pooled, TextDim, _dim);
        DiffusionOps.SiluInPlace(c0);
        var cOut = Linear("txt_in.c_embedder.linear_2", c0, _dim, _dim);

        var temb = new float[_dim];
        for (int d = 0; d < _dim; d++) temb[d] = tOut[d] + cOut[d];

        var x = Linear("txt_in.input_embedder", textContext, TextDim, _dim);

        _refinerDepth ??= DetectRefinerDepth();
        for (int b = 0; b < _refinerDepth; b++)
            x = TokenRefinerBlock($"txt_in.individual_token_refiner.blocks.{b}", x, temb, seqLen);

        return x;
    }

    private int DetectRefinerDepth()
    {
        for (int i = 30; i >= 0; i--)
        {
            if (_weights.Contains(Resolve($"txt_in.individual_token_refiner.blocks.{i}.self_attn_qkv.weight")))
                return i + 1;
        }
        return 0;
    }

    private float[] TokenRefinerBlock(string prefix, float[] x, float[] temb, int seqLen)
    {
        var normed1 = (float[])x.Clone();
        var norm1W = GetWeight($"{prefix}.norm1.weight");
        var norm1B = GetWeight($"{prefix}.norm1.bias");
        DiffusionOps.LayerNorm(normed1, norm1W, norm1B, _dim, eps: 1e-6f);

        var qkv = Linear($"{prefix}.self_attn_qkv", normed1, _dim, _dim * 3);
        var (q, k, v) = SplitQkv(qkv, seqLen, _dim);
        var attnOut = MultiHeadAttention(q, k, v, seqLen, _numHeads, _headDim);
        attnOut = Linear($"{prefix}.self_attn_proj", attnOut, _dim, _dim);

        var gate = Linear($"{prefix}.adaLN_modulation.1", DiffusionOpsSilu(temb), _dim, _dim * 2);
        var gateMsa = gate.AsSpan(0, _dim);
        var gateMlp = gate.AsSpan(_dim, _dim);

        for (int s = 0; s < seqLen; s++)
            for (int d = 0; d < _dim; d++)
                x[s * _dim + d] += attnOut[s * _dim + d] * gateMsa[d];

        var normed2 = (float[])x.Clone();
        var norm2W = GetWeight($"{prefix}.norm2.weight");
        var norm2B = GetWeight($"{prefix}.norm2.bias");
        DiffusionOps.LayerNorm(normed2, norm2W, norm2B, _dim, eps: 1e-6f);

        int mlpHidden = _dim * 4;
        var fc1 = Linear($"{prefix}.mlp.fc1", normed2, _dim, mlpHidden);
        DiffusionOps.SiluInPlace(fc1);
        var fc2 = Linear($"{prefix}.mlp.fc2", fc1, mlpHidden, _dim);

        for (int s = 0; s < seqLen; s++)
            for (int d = 0; d < _dim; d++)
                x[s * _dim + d] += fc2[s * _dim + d] * gateMlp[d];

        return x;
    }

    private static float[] DiffusionOpsSilu(float[] x)
    {
        var res = (float[])x.Clone();
        DiffusionOps.SiluInPlace(res);
        return res;
    }

    private float[] Linear(string name, float[] x, int inDim, int outDim)
    {
        int rows = x.Length / inDim;
        var outF = new float[rows * outDim];
        Linear(name, x.AsSpan(), outF.AsSpan(), inDim, outDim);
        return outF;
    }

    private void Linear(string name, ReadOnlySpan<float> x, Span<float> output, int inDim, int outDim)
    {
        string wName = Resolve($"{name}.weight");
        string bName = Resolve($"{name}.bias");
        var b = _biasCache.GetOrAdd(bName, k => TryGetWeightDirect(k));
        int rows = x.Length / inDim;
        _quantizedCache.Linear(wName, x, b ?? ReadOnlySpan<float>.Empty, output, rows, inDim, outDim);
    }

    private float[]? TryGetWeightDirect(string fullName)
    {
        if (_weights.Contains(fullName))
        {
            return _weights.ReadF32(fullName);
        }
        return null;
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

                    for (int c = 0; c < 16; c++)
                    {
                        for (int dy = 0; dy < 2; dy++)
                        {
                            for (int dx = 0; dx < 2; dx++)
                            {
                                int y = ph * 2 + dy;
                                int x = pw * 2 + dx;
                                int latIdx = ((c * numFrames + f) * latH + y) * latW + x;
                                packed[tokenOff + chanOffset++] = latents[latIdx];
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
        int totalLatents = 16 * numFrames * latH * latW;
        var latents = new float[totalLatents];

        for (int f = 0; f < numFrames; f++)
        {
            for (int ph = 0; ph < patchH; ph++)
            {
                for (int pw = 0; pw < patchW; pw++)
                {
                    int tokenIdx = (f * patchH + ph) * patchW + pw;
                    int tokenOff = tokenIdx * InChannels;
                    int chanOffset = 0;

                    for (int c = 0; c < 16; c++)
                    {
                        for (int dy = 0; dy < 2; dy++)
                        {
                            for (int dx = 0; dx < 2; dx++)
                            {
                                int y = ph * 2 + dy;
                                int x = pw * 2 + dx;
                                int latIdx = ((c * numFrames + f) * latH + y) * latW + x;
                                latents[latIdx] = packed[tokenOff + chanOffset++];
                            }
                        }
                    }
                }
            }
        }
        return latents;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _quantizedCache.Dispose();
        if (_gpuWeights is not null)
        {
            foreach (var t in _gpuWeights.Values) t.Dispose();
            _gpuWeights.Clear();
        }
    }
}
