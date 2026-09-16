using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.StableDiffusion;

/// <summary>
/// Stable Diffusion 1.5 UNet (2D Condition Model).
/// Supports both CPU (SIMD AVX2/AVX-512) and GPU (Vulkan/CUDA SGEMM via IComputeBackend).
/// </summary>
public sealed class UNet2DConditionModel : IDisposable
{
    private readonly CachedWeightReader _weightReader;
    private readonly IComputeBackend? _backend;
    private readonly Dictionary<string, CoreTensor>? _gpuWeights;

    // Separate cache for the native implicit-GEMM Conv2d shader path (see ConvNative below) --
    // its weight buffer must be plain Float32, unlike _gpuWeights above. Same rationale as the
    // identical fields in SdxlUNet2DConditionModel/VaeDecoder.
    private readonly Dictionary<string, CoreTensor>? _gpuWeightsNative;
    private readonly IImageOpsBackend? _imageOps;
    private readonly Dictionary<float[], CoreTensor> _cachedContextGpu = new(ReferenceEqualityComparer.Instance);

    private const int ModelChannels = 320;
    private const int TimeEmbedDim = 1280;
    private const int ContextDim = 768;
    private const int NumHeads = 8;
    private const int MaxColChunkFloats = 8 * 1024 * 1024; // 32MB im2col buffer chunk

    public void ClearContextCache()
    {
        lock (_cachedContextGpu)
        {
            if (_imageOps is not null)
            {
                foreach (var t in _cachedContextGpu.Values) _imageOps.Free(t);
            }
            _cachedContextGpu.Clear();
        }
    }

    public UNet2DConditionModel(IWeightLoader weights, string prefix = "model.diffusion_model.", IComputeBackend? backend = null)
    {
        _weightReader = new CachedWeightReader(weights, prefix);
        _backend = backend;
        if (_backend is not null)
        {
            _gpuWeights = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
            _gpuWeightsNative = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
            _imageOps = backend as IImageOpsBackend;
        }
    }

    public IImageOpsBackend? ImageOps => _imageOps;

    private float[] GetWeight(string name) => _weightReader.Get(name);

    private float[]? TryGetWeight(string name) => _weightReader.TryGet(name);

    private CoreTensor GetGpuWeight(string name, float[] cpuWeight)
    {
        string fullName = _weightReader.Prefix + name;
        if (_gpuWeights!.TryGetValue(fullName, out var wGpu)) return wGpu;

        // Perf (2026-09-11): same fix as SdxlUNet2DConditionModel.GetGpuWeight -- upload weights
        // as Half when the backend prefers fp16 Sgemm, instead of always forcing the slowest
        // full-fp32 path. This SD1.5 UNet had gotten the later implicit-GEMM Conv() fix but had
        // been missed for this earlier one; applying it now for consistency and the same real win.
        if (_backend!.BestSgemmPrecision == SgemmPrecision.Fp16)
        {
            var half = new Half[cpuWeight.Length];
            TensorPrimitives.ConvertToHalf(cpuWeight, half);
            wGpu = _backend.UploadHalf(half, TensorShape.D1(cpuWeight.Length));
        }
        else
        {
            wGpu = _backend.Upload(cpuWeight.AsSpan(), TensorShape.D1(cpuWeight.Length));
        }
        _gpuWeights[fullName] = wGpu;
        return wGpu;
    }

    /// <summary>Conv2D via the native GPU implicit-GEMM shader -- see the identical method in
    /// SdxlUNet2DConditionModel for the full rationale. Requires stride=1 (guarded by caller).</summary>
    private float[] ConvNative(IImageOpsBackend imageOps, string name, float[] wF, float[]? bF, float[] x, int inCh, int h, int w, int outCh, int k, int padding = -1)
    {
        string wKey = $"{name}.weight";
        if (!_gpuWeightsNative!.TryGetValue(wKey, out var wGpu))
        {
            wGpu = imageOps.Upload(wF.AsSpan(), TensorShape.D1(wF.Length));
            _gpuWeightsNative[wKey] = wGpu;
        }

        string bKey = $"{name}.bias";
        if (!_gpuWeightsNative.TryGetValue(bKey, out var bGpu))
        {
            var bf = bF ?? new float[outCh];
            bGpu = imageOps.Upload(bf.AsSpan(), TensorShape.D1(bf.Length));
            _gpuWeightsNative[bKey] = bGpu;
        }

        var xGpu = imageOps.Upload(x.AsSpan(0, inCh * h * w), TensorShape.D1(inCh * h * w));
        var yGpu = imageOps.Conv2dImplicitGemm(xGpu, wGpu, bGpu, inCh, outCh, h, w, k, padding);
        var result = new float[outCh * h * w];
        try
        {
            imageOps.Download(yGpu, result);
        }
        finally
        {
            imageOps.Free(xGpu);
            imageOps.Free(yGpu);
        }
        return result;
    }

    private readonly Dictionary<string, CoreTensor> _gpuBiasCache = new(StringComparer.Ordinal);
    private bool? _unetForwardResidencySupported;
    private bool? _residentGpuAttentionSupported;
    private bool _gpuWeightsWarm;

    /// <summary>
    /// Pre-uploads all UNet conv weights, linear weights, groupnorm/layernorm biases, and conv biases
    /// to GPU device memory so that forward passes can be recorded into a single Vulkan command buffer.
    /// </summary>
    public void EnsureGpuResident()
    {
        if (_imageOps is null || _gpuWeightsWarm) return;
        EnsureGpuResidentCore(_imageOps);
        _gpuWeightsWarm = true;
    }

    private void EnsureConvGpuResident(IImageOpsBackend imageOps, string name, int outCh)
    {
        var wF = GetWeight($"{name}.weight");
        var bF = TryGetWeight($"{name}.bias");

        string wKey = $"{name}.weight";
        if (!_gpuWeightsNative!.ContainsKey(wKey))
        {
            var wGpu = imageOps.Upload(wF.AsSpan(), TensorShape.D1(wF.Length));
            _gpuWeightsNative[wKey] = wGpu;
        }

        string bKey = $"{name}.bias";
        if (!_gpuWeightsNative.ContainsKey(bKey))
        {
            var bf = bF ?? new float[outCh];
            var bGpu = imageOps.Upload(bf.AsSpan(), TensorShape.D1(bf.Length));
            _gpuWeightsNative[bKey] = bGpu;
        }
    }

    private void EnsureLinGpuResident(string name)
    {
        var wF = GetWeight($"{name}.weight");
        var bF = TryGetWeight($"{name}.bias");
        GetGpuWeight($"{name}.weight", wF);
        if (bF is not null)
        {
            GetGpuBias(_backend!, $"{name}.bias", bF);
        }
    }

    private void EnsureGroupNormGpuResident(IImageOpsBackend imageOps, string prefix)
    {
        GetGpuBias(imageOps, $"{prefix}.weight", GetWeight($"{prefix}.weight"));
        GetGpuBias(imageOps, $"{prefix}.bias", GetWeight($"{prefix}.bias"));
    }

    private void EnsureResBlockGpuResident(IImageOpsBackend imageOps, string prefix, int outCh)
    {
        EnsureGroupNormGpuResident(imageOps, $"{prefix}.in_layers.0");
        EnsureConvGpuResident(imageOps, $"{prefix}.in_layers.2", outCh);
        EnsureLinGpuResident($"{prefix}.emb_layers.1");
        EnsureGroupNormGpuResident(imageOps, $"{prefix}.out_layers.0");
        EnsureConvGpuResident(imageOps, $"{prefix}.out_layers.3", outCh);
        if (TryGetWeight($"{prefix}.skip_connection.weight") is not null)
        {
            EnsureConvGpuResident(imageOps, $"{prefix}.skip_connection", outCh);
        }
    }

    private void EnsureSpatialTransformerGpuResident(IImageOpsBackend imageOps, string prefix, int c)
    {
        EnsureGroupNormGpuResident(imageOps, $"{prefix}.norm");
        EnsureLinGpuResident($"{prefix}.proj_in");

        string tb = $"{prefix}.transformer_blocks.0";
        EnsureGroupNormGpuResident(imageOps, $"{tb}.norm1");
        EnsureLinGpuResident($"{tb}.attn1.to_q");
        EnsureLinGpuResident($"{tb}.attn1.to_k");
        EnsureLinGpuResident($"{tb}.attn1.to_v");
        EnsureLinGpuResident($"{tb}.attn1.to_out.0");

        EnsureGroupNormGpuResident(imageOps, $"{tb}.norm2");
        EnsureLinGpuResident($"{tb}.attn2.to_q");
        EnsureLinGpuResident($"{tb}.attn2.to_k");
        EnsureLinGpuResident($"{tb}.attn2.to_v");
        EnsureLinGpuResident($"{tb}.attn2.to_out.0");

        EnsureGroupNormGpuResident(imageOps, $"{tb}.norm3");
        EnsureLinGpuResident($"{tb}.ff.net.0.proj");
        EnsureLinGpuResident($"{tb}.ff.net.2");

        EnsureLinGpuResident($"{prefix}.proj_out");
    }

    private void EnsureGpuResidentCore(IImageOpsBackend imageOps)
    {
        // Input blocks
        EnsureConvGpuResident(imageOps, "input_blocks.0.0", 320);
        EnsureResBlockGpuResident(imageOps, "input_blocks.1.0", 320);
        EnsureSpatialTransformerGpuResident(imageOps, "input_blocks.1.1", 320);
        EnsureResBlockGpuResident(imageOps, "input_blocks.2.0", 320);
        EnsureSpatialTransformerGpuResident(imageOps, "input_blocks.2.1", 320);
        EnsureConvGpuResident(imageOps, "input_blocks.3.0.op", 320);
        EnsureResBlockGpuResident(imageOps, "input_blocks.4.0", 640);
        EnsureSpatialTransformerGpuResident(imageOps, "input_blocks.4.1", 640);
        EnsureResBlockGpuResident(imageOps, "input_blocks.5.0", 640);
        EnsureSpatialTransformerGpuResident(imageOps, "input_blocks.5.1", 640);
        EnsureConvGpuResident(imageOps, "input_blocks.6.0.op", 640);
        EnsureResBlockGpuResident(imageOps, "input_blocks.7.0", 1280);
        EnsureSpatialTransformerGpuResident(imageOps, "input_blocks.7.1", 1280);
        EnsureResBlockGpuResident(imageOps, "input_blocks.8.0", 1280);
        EnsureSpatialTransformerGpuResident(imageOps, "input_blocks.8.1", 1280);
        EnsureConvGpuResident(imageOps, "input_blocks.9.0.op", 1280);
        EnsureResBlockGpuResident(imageOps, "input_blocks.10.0", 1280);
        EnsureResBlockGpuResident(imageOps, "input_blocks.11.0", 1280);

        // Middle block
        EnsureResBlockGpuResident(imageOps, "middle_block.0", 1280);
        EnsureSpatialTransformerGpuResident(imageOps, "middle_block.1", 1280);
        EnsureResBlockGpuResident(imageOps, "middle_block.2", 1280);

        // Output blocks
        EnsureResBlockGpuResident(imageOps, "output_blocks.0.0", 1280);
        EnsureResBlockGpuResident(imageOps, "output_blocks.1.0", 1280);
        EnsureResBlockGpuResident(imageOps, "output_blocks.2.0", 1280);
        EnsureConvGpuResident(imageOps, "output_blocks.2.1.conv", 1280);
        EnsureResBlockGpuResident(imageOps, "output_blocks.3.0", 1280);
        EnsureSpatialTransformerGpuResident(imageOps, "output_blocks.3.1", 1280);
        EnsureResBlockGpuResident(imageOps, "output_blocks.4.0", 1280);
        EnsureSpatialTransformerGpuResident(imageOps, "output_blocks.4.1", 1280);
        EnsureResBlockGpuResident(imageOps, "output_blocks.5.0", 1280);
        EnsureSpatialTransformerGpuResident(imageOps, "output_blocks.5.1", 1280);
        EnsureConvGpuResident(imageOps, "output_blocks.5.2.conv", 1280);
        EnsureResBlockGpuResident(imageOps, "output_blocks.6.0", 640);
        EnsureSpatialTransformerGpuResident(imageOps, "output_blocks.6.1", 640);
        EnsureResBlockGpuResident(imageOps, "output_blocks.7.0", 640);
        EnsureSpatialTransformerGpuResident(imageOps, "output_blocks.7.1", 640);
        EnsureResBlockGpuResident(imageOps, "output_blocks.8.0", 640);
        EnsureSpatialTransformerGpuResident(imageOps, "output_blocks.8.1", 640);
        EnsureConvGpuResident(imageOps, "output_blocks.8.2.conv", 640);
        EnsureResBlockGpuResident(imageOps, "output_blocks.9.0", 320);
        EnsureSpatialTransformerGpuResident(imageOps, "output_blocks.9.1", 320);
        EnsureResBlockGpuResident(imageOps, "output_blocks.10.0", 320);
        EnsureSpatialTransformerGpuResident(imageOps, "output_blocks.10.1", 320);
        EnsureResBlockGpuResident(imageOps, "output_blocks.11.0", 320);
        EnsureSpatialTransformerGpuResident(imageOps, "output_blocks.11.1", 320);

        // Final out
        EnsureGroupNormGpuResident(imageOps, "out.0");
        EnsureConvGpuResident(imageOps, "out.2", 4);
    }

    private CoreTensor GetGpuBias(IComputeBackend backend, string name, float[] bF)
    {
        string fullName = _weightReader.Prefix + name;
        if (_gpuBiasCache.TryGetValue(fullName, out var bGpu)) return bGpu;
        bGpu = backend.Upload(bF.AsSpan(), TensorShape.D1(bF.Length));
        _gpuBiasCache[fullName] = bGpu;
        return bGpu;
    }

    /// <summary>Tensor-in/Tensor-out counterpart of <see cref="ConvNative"/>.</summary>
    private CoreTensor ConvGpuTensor(IImageOpsBackend imageOps, string name, CoreTensor xGpu, int inCh, int h, int w, int outCh, int k, int padding = -1, int stride = 1)
    {
        var wF = GetWeight($"{name}.weight");
        var bF = TryGetWeight($"{name}.bias");

        string wKey = $"{name}.weight";
        if (!_gpuWeightsNative!.TryGetValue(wKey, out var wGpu))
        {
            wGpu = imageOps.Upload(wF.AsSpan(), TensorShape.D1(wF.Length));
            _gpuWeightsNative[wKey] = wGpu;
        }
        string bKey = $"{name}.bias";
        if (!_gpuWeightsNative.TryGetValue(bKey, out var bGpu))
        {
            var bf = bF ?? new float[outCh];
            bGpu = imageOps.Upload(bf.AsSpan(), TensorShape.D1(bf.Length));
            _gpuWeightsNative[bKey] = bGpu;
        }

        return imageOps.Conv2dImplicitGemm(xGpu, wGpu, bGpu, inCh, outCh, h, w, k, padding, stride);
    }

    /// <summary>Tensor-in/Tensor-out counterpart of <see cref="Lin"/> -- Sgemm + row-broadcast bias.</summary>
    private CoreTensor LinGpuTensor(string name, CoreTensor xGpu, int n, int inDim, int outDim)
    {
        var wF = GetWeight($"{name}.weight");
        var bF = TryGetWeight($"{name}.bias");
        var wGpu = GetGpuWeight($"{name}.weight", wF);
        var cGpu = _backend!.Allocate(TensorShape.D1(n * outDim));
        _backend.Sgemm(cGpu, xGpu, wGpu, n, inDim, outDim);
        if (bF is not null)
        {
            var bGpu = GetGpuBias(_backend, $"{name}.bias", bF);
            ((IImageOpsBackend)_backend).AddRowBroadcastInPlace(cGpu, bGpu, n, outDim);
        }
        return cGpu;
    }

    /// <summary>Tensor-in/Tensor-out GroupNorm+SiLU with cached weights.</summary>
    private CoreTensor GroupNormSiluGpuTensor(IImageOpsBackend imageOps, string prefix, CoreTensor xGpu, int c, int hw)
    {
        var gnW = GetGpuBias(imageOps, $"{prefix}.weight", GetWeight($"{prefix}.weight"));
        var gnB = GetGpuBias(imageOps, $"{prefix}.bias", GetWeight($"{prefix}.bias"));
        return imageOps.GroupNormSilu(xGpu, gnW, gnB, c, hw, groups: 32);
    }

    private CoreTensor UploadSiluTEmb(IImageOpsBackend imageOps, float[] tEmb)
    {
        var tEmbAct = (float[])tEmb.Clone();
        DiffusionOps.SiluInPlace(tEmbAct);
        return imageOps.Upload(tEmbAct.AsSpan(), TensorShape.D1(tEmbAct.Length));
    }

    private CoreTensor ResBlockGpu(IImageOpsBackend imageOps, string prefix, CoreTensor xGpu, CoreTensor tEmbGpu, int inCh, int h, int w, int outCh)
    {
        int hw = h * w;
        var t1 = GroupNormSiluGpuTensor(imageOps, $"{prefix}.in_layers.0", xGpu, inCh, hw);
        var t2 = ConvGpuTensor(imageOps, $"{prefix}.in_layers.2", t1, inCh, h, w, outCh, 3);
        imageOps.Free(t1);

        var tProjGpu = LinGpuTensor($"{prefix}.emb_layers.1", tEmbGpu, 1, TimeEmbedDim, outCh);
        imageOps.AddChannelBroadcastInPlace(t2, tProjGpu, outCh, hw);
        imageOps.Free(tProjGpu);

        var t3 = GroupNormSiluGpuTensor(imageOps, $"{prefix}.out_layers.0", t2, outCh, hw);
        imageOps.Free(t2);
        var t4 = ConvGpuTensor(imageOps, $"{prefix}.out_layers.3", t3, outCh, h, w, outCh, 3);
        imageOps.Free(t3);

        CoreTensor skip = xGpu;
        bool freeSkip = false;
        if (TryGetWeight($"{prefix}.skip_connection.weight") is not null)
        {
            skip = ConvGpuTensor(imageOps, $"{prefix}.skip_connection", xGpu, inCh, h, w, outCh, 1, padding: 0);
            freeSkip = true;
        }
        else if (inCh != outCh)
        {
            throw new InvalidOperationException($"ResBlock {prefix} has inC ({inCh}) != outC ({outCh}) without skip_connection.");
        }

        imageOps.AddInPlace(t4, skip);
        if (freeSkip) imageOps.Free(skip);
        return t4;
    }

    private CoreTensor CpuAttentionIsland(IImageOpsBackend imageOps, CoreTensor qGpu, CoreTensor kGpu, CoreTensor vGpu, int qSeq, int kvSeq, int c, int nHeads, int headDim)
    {
        var q = new float[qSeq * c];
        var k = new float[kvSeq * c];
        var v = new float[kvSeq * c];
        imageOps.Download(qGpu, q);
        imageOps.Download(kGpu, k);
        imageOps.Download(vGpu, v);
        var attnOut = DiffusionOps.MultiHeadAttention(q, k, v, qSeq, kvSeq, nHeads, headDim);
        return imageOps.Upload(attnOut.AsSpan(), TensorShape.D1(attnOut.Length));
    }

    private CoreTensor AttentionIsland(IImageOpsBackend imageOps, CoreTensor qGpu, CoreTensor kGpu, CoreTensor vGpu, int qSeq, int kvSeq, int c, int nHeads)
    {
        int headDim = c / nHeads;
        if (headDim <= 256 && _residentGpuAttentionSupported != false)
        {
            try
            {
                var result = imageOps.MultiHeadAttentionTiled(qGpu, kGpu, vGpu, qSeq, kvSeq, nHeads, headDim);
                _residentGpuAttentionSupported = true;
                return result;
            }
            catch (NotSupportedException)
            {
                _residentGpuAttentionSupported = false;
            }
        }
        return CpuAttentionIsland(imageOps, qGpu, kGpu, vGpu, qSeq, kvSeq, c, nHeads, headDim);
    }

    private CoreTensor SpatialTransformerGpu(IImageOpsBackend imageOps, string prefix, CoreTensor xGpu, CoreTensor contextGpu, int c, int h, int w)
    {
        int hw = h * w;

        var normW = GetGpuBias(imageOps, $"{prefix}.norm.weight", GetWeight($"{prefix}.norm.weight"));
        var normB = GetGpuBias(imageOps, $"{prefix}.norm.bias", GetWeight($"{prefix}.norm.bias"));
        var xNorm = imageOps.GroupNormGpu(xGpu, normW, normB, c, hw);

        // Perf: 1x1 conv is a pure linear projection. Permuting [C, HW] -> [HW, C] first allows
        // running proj_in via SgemmF16 (128-bit vector loads, fp16 weights) instead of the scalar Conv2dImplicitGemm.
        var xNormSeq = imageOps.PermuteChwToHwc(xNorm, c, hw);
        imageOps.Free(xNorm);

        var xSeq = LinGpuTensor($"{prefix}.proj_in", xNormSeq, hw, c, c);
        imageOps.Free(xNormSeq);

        string tb = $"{prefix}.transformer_blocks.0";

        // 1. Self-attention
        var saNormW = GetGpuBias(imageOps, $"{tb}.norm1.weight", GetWeight($"{tb}.norm1.weight"));
        var saNormB = GetGpuBias(imageOps, $"{tb}.norm1.bias", GetWeight($"{tb}.norm1.bias"));
        var saNorm = imageOps.LayerNormGpu(xSeq, saNormW, saNormB, hw, c);

        var saQ = LinGpuTensor($"{tb}.attn1.to_q", saNorm, hw, c, c);
        var saK = LinGpuTensor($"{tb}.attn1.to_k", saNorm, hw, c, c);
        var saV = LinGpuTensor($"{tb}.attn1.to_v", saNorm, hw, c, c);
        imageOps.Free(saNorm);

        var saAttnOut = AttentionIsland(imageOps, saQ, saK, saV, hw, hw, c, NumHeads);
        imageOps.Free(saQ); imageOps.Free(saK); imageOps.Free(saV);

        var saProjOut = LinGpuTensor($"{tb}.attn1.to_out.0", saAttnOut, hw, c, c);
        imageOps.Free(saAttnOut);
        imageOps.AddInPlace(xSeq, saProjOut);
        imageOps.Free(saProjOut);

        // 2. Cross-attention
        var caNormW = GetGpuBias(imageOps, $"{tb}.norm2.weight", GetWeight($"{tb}.norm2.weight"));
        var caNormB = GetGpuBias(imageOps, $"{tb}.norm2.bias", GetWeight($"{tb}.norm2.bias"));
        var caNorm = imageOps.LayerNormGpu(xSeq, caNormW, caNormB, hw, c);

        var caQ = LinGpuTensor($"{tb}.attn2.to_q", caNorm, hw, c, c);
        imageOps.Free(caNorm);
        var caK = LinGpuTensor($"{tb}.attn2.to_k", contextGpu, 77, ContextDim, c);
        var caV = LinGpuTensor($"{tb}.attn2.to_v", contextGpu, 77, ContextDim, c);

        var caAttnOut = AttentionIsland(imageOps, caQ, caK, caV, hw, 77, c, NumHeads);
        imageOps.Free(caQ); imageOps.Free(caK); imageOps.Free(caV);

        var caProjOut = LinGpuTensor($"{tb}.attn2.to_out.0", caAttnOut, hw, c, c);
        imageOps.Free(caAttnOut);
        imageOps.AddInPlace(xSeq, caProjOut);
        imageOps.Free(caProjOut);

        // 3. GEGLU FeedForward
        var ffNormW = GetGpuBias(imageOps, $"{tb}.norm3.weight", GetWeight($"{tb}.norm3.weight"));
        var ffNormB = GetGpuBias(imageOps, $"{tb}.norm3.bias", GetWeight($"{tb}.norm3.bias"));
        var ffNorm = imageOps.LayerNormGpu(xSeq, ffNormW, ffNormB, hw, c);

        int mlpDim = c * 4;
        var ffH = LinGpuTensor($"{tb}.ff.net.0.proj", ffNorm, hw, c, mlpDim * 2);
        imageOps.Free(ffNorm);
        var ffGated = imageOps.GeGlu(ffH, hw, mlpDim);
        imageOps.Free(ffH);

        var ffOut = LinGpuTensor($"{tb}.ff.net.2", ffGated, hw, mlpDim, c);
        imageOps.Free(ffGated);
        imageOps.AddInPlace(xSeq, ffOut);
        imageOps.Free(ffOut);

        // Perf: run proj_out directly on xSeq [HW, C] via SgemmF16 before permuting back to [C, HW]
        var projOutSeq = LinGpuTensor($"{prefix}.proj_out", xSeq, hw, c, c);
        imageOps.Free(xSeq);

        var projOut = imageOps.PermuteHwcToChw(projOutSeq, c, hw);
        imageOps.Free(projOutSeq);

        imageOps.AddInPlace(projOut, xGpu);
        return projOut;
    }


    private CoreTensor ForwardGpu(
        IImageOpsBackend imageOps,
        float[] x,
        float[] tEmb,
        float[] context,
        int latH,
        int latW,
        IReadOnlyList<float[]>? controlDownResiduals = null,
        float[]? controlMidResidual = null,
        IReadOnlyList<CoreTensor>? controlDownGpuParam = null,
        CoreTensor? controlMidGpuParam = null)
    {
        EnsureGpuResident();

        int h = latH, w = latW;

        // Pre-upload all host inputs and residuals BEFORE BeginBatch() so that no transfer commands interrupt recording
        CoreTensor contextGpu;
        lock (_cachedContextGpu)
        {
            if (!_cachedContextGpu.TryGetValue(context, out contextGpu!))
            {
                contextGpu = imageOps.Upload(context.AsSpan(0, 77 * ContextDim), TensorShape.D1(77 * ContextDim));
                _cachedContextGpu[context] = contextGpu;
            }
        }
        var tEmbGpu = UploadSiluTEmb(imageOps, tEmb);
        var xGpu = imageOps.Upload(x.AsSpan(0, 4 * h * w), TensorShape.D1(4 * h * w));

        bool ownsControlTensors = false;
        IReadOnlyList<CoreTensor>? controlDownGpu = controlDownGpuParam;
        if (controlDownGpu is null && controlDownResiduals is not null)
        {
            var uploadedDown = new CoreTensor[controlDownResiduals.Count];
            for (int i = 0; i < controlDownResiduals.Count; i++)
            {
                var res = controlDownResiduals[i];
                if (res is not null && res.Length > 0)
                    uploadedDown[i] = imageOps.Upload(res.AsSpan(), TensorShape.D1(res.Length));
            }
            controlDownGpu = uploadedDown;
            ownsControlTensors = true;
        }

        CoreTensor? controlMidGpu = controlMidGpuParam;
        if (controlMidGpu is null && controlMidResidual is not null && controlMidResidual.Length > 0)
        {
            controlMidGpu = imageOps.Upload(controlMidResidual.AsSpan(), TensorShape.D1(controlMidResidual.Length));
            ownsControlTensors = true;
        }

        var swRecord = System.Diagnostics.Stopwatch.StartNew();
        imageOps.BeginBatch();
        bool batchSuccess = false;
        try
        {
            var savedInputs = new List<CoreTensor>(12);

            var cur = ConvGpuTensor(imageOps, "input_blocks.0.0", xGpu, 4, h, w, 320, 3);
            imageOps.Free(xGpu);
            savedInputs.Add(cur);

            cur = ResBlockGpu(imageOps, "input_blocks.1.0", cur, tEmbGpu, 320, h, w, 320);
            cur = SpatialTransformerGpu(imageOps, "input_blocks.1.1", cur, contextGpu, 320, h, w);
            savedInputs.Add(cur);

            cur = ResBlockGpu(imageOps, "input_blocks.2.0", cur, tEmbGpu, 320, h, w, 320);
            cur = SpatialTransformerGpu(imageOps, "input_blocks.2.1", cur, contextGpu, 320, h, w);
            savedInputs.Add(cur);

            // Block 3: Downsample (320 -> 320, stride 2)
            cur = ConvGpuTensor(imageOps, "input_blocks.3.0.op", cur, 320, h, w, 320, 3, padding: 1, stride: 2);
            h /= 2; w /= 2;
            savedInputs.Add(cur);

            cur = ResBlockGpu(imageOps, "input_blocks.4.0", cur, tEmbGpu, 320, h, w, 640);
            cur = SpatialTransformerGpu(imageOps, "input_blocks.4.1", cur, contextGpu, 640, h, w);
            savedInputs.Add(cur);

            cur = ResBlockGpu(imageOps, "input_blocks.5.0", cur, tEmbGpu, 640, h, w, 640);
            cur = SpatialTransformerGpu(imageOps, "input_blocks.5.1", cur, contextGpu, 640, h, w);
            savedInputs.Add(cur);

            // Block 6: Downsample (640 -> 640, stride 2)
            cur = ConvGpuTensor(imageOps, "input_blocks.6.0.op", cur, 640, h, w, 640, 3, padding: 1, stride: 2);
            h /= 2; w /= 2;
            savedInputs.Add(cur);

            cur = ResBlockGpu(imageOps, "input_blocks.7.0", cur, tEmbGpu, 640, h, w, 1280);
            cur = SpatialTransformerGpu(imageOps, "input_blocks.7.1", cur, contextGpu, 1280, h, w);
            savedInputs.Add(cur);

            cur = ResBlockGpu(imageOps, "input_blocks.8.0", cur, tEmbGpu, 1280, h, w, 1280);
            cur = SpatialTransformerGpu(imageOps, "input_blocks.8.1", cur, contextGpu, 1280, h, w);
            savedInputs.Add(cur);

            // Block 9: Downsample (1280 -> 1280, stride 2)
            cur = ConvGpuTensor(imageOps, "input_blocks.9.0.op", cur, 1280, h, w, 1280, 3, padding: 1, stride: 2);
            h /= 2; w /= 2;
            savedInputs.Add(cur);

            cur = ResBlockGpu(imageOps, "input_blocks.10.0", cur, tEmbGpu, 1280, h, w, 1280);
            savedInputs.Add(cur);

            cur = ResBlockGpu(imageOps, "input_blocks.11.0", cur, tEmbGpu, 1280, h, w, 1280);
            savedInputs.Add(cur);

            // Apply ControlNet down residuals
            if (controlDownGpu is not null)
            {
                for (int i = 0; i < Math.Min(savedInputs.Count, controlDownGpu.Count); i++)
                {
                    if (controlDownGpu[i] is not null)
                    {
                        imageOps.AddInPlace(savedInputs[i], controlDownGpu[i]);
                        if (ownsControlTensors)
                            imageOps.Free(controlDownGpu[i]);
                    }
                }
            }

            // ── Middle Block ────────────────────────────────────────────────────────
            cur = ResBlockGpu(imageOps, "middle_block.0", cur, tEmbGpu, 1280, h, w, 1280);
            cur = SpatialTransformerGpu(imageOps, "middle_block.1", cur, contextGpu, 1280, h, w);
            cur = ResBlockGpu(imageOps, "middle_block.2", cur, tEmbGpu, 1280, h, w, 1280);

            if (controlMidGpu is not null)
            {
                imageOps.AddInPlace(cur, controlMidGpu);
                if (ownsControlTensors)
                    imageOps.Free(controlMidGpu);
            }

            // ── Output Blocks ───────────────────────────────────────────────────────
            CoreTensor CatAndFreeSkip(CoreTensor current, int idx, int curC, int skipC)
            {
                var skip = savedInputs[idx];
                var result = imageOps.CatChannels(current, curC, skip, skipC, h * w);
                imageOps.Free(current);
                imageOps.Free(skip);
                return result;
            }

            // Block 0: ResBlock(1280 + 1280 -> 1280)
            cur = CatAndFreeSkip(cur, 11, 1280, 1280);
            cur = ResBlockGpu(imageOps, "output_blocks.0.0", cur, tEmbGpu, 2560, h, w, 1280);

            // Block 1: ResBlock(1280 + 1280 -> 1280)
            cur = CatAndFreeSkip(cur, 10, 1280, 1280);
            cur = ResBlockGpu(imageOps, "output_blocks.1.0", cur, tEmbGpu, 2560, h, w, 1280);

            // Block 2: ResBlock(1280 + 1280 -> 1280) + Upsample(1280 -> 1280) + Conv
            cur = CatAndFreeSkip(cur, 9, 1280, 1280);
            cur = ResBlockGpu(imageOps, "output_blocks.2.0", cur, tEmbGpu, 2560, h, w, 1280);
            var upsampled2 = imageOps.Upsample2xGpu(cur, 1280, h, w);
            imageOps.Free(cur);
            h *= 2; w *= 2;
            cur = ConvGpuTensor(imageOps, "output_blocks.2.1.conv", upsampled2, 1280, h, w, 1280, 3);
            imageOps.Free(upsampled2);

            // Block 3: ResBlock(1280 + 1280 -> 1280) + SpatialTransformer(1280)
            cur = CatAndFreeSkip(cur, 8, 1280, 1280);
            cur = ResBlockGpu(imageOps, "output_blocks.3.0", cur, tEmbGpu, 2560, h, w, 1280);
            cur = SpatialTransformerGpu(imageOps, "output_blocks.3.1", cur, contextGpu, 1280, h, w);

            // Block 4: ResBlock(1280 + 1280 -> 1280) + SpatialTransformer(1280)
            cur = CatAndFreeSkip(cur, 7, 1280, 1280);
            cur = ResBlockGpu(imageOps, "output_blocks.4.0", cur, tEmbGpu, 2560, h, w, 1280);
            cur = SpatialTransformerGpu(imageOps, "output_blocks.4.1", cur, contextGpu, 1280, h, w);

            // Block 5: ResBlock(1280 + 640 -> 1280) + SpatialTransformer(1280) + Upsample + Conv
            cur = CatAndFreeSkip(cur, 6, 1280, 640);
            cur = ResBlockGpu(imageOps, "output_blocks.5.0", cur, tEmbGpu, 1920, h, w, 1280);
            cur = SpatialTransformerGpu(imageOps, "output_blocks.5.1", cur, contextGpu, 1280, h, w);
            var upsampled5 = imageOps.Upsample2xGpu(cur, 1280, h, w);
            imageOps.Free(cur);
            h *= 2; w *= 2;
            cur = ConvGpuTensor(imageOps, "output_blocks.5.2.conv", upsampled5, 1280, h, w, 1280, 3);
            imageOps.Free(upsampled5);

            // Block 6: ResBlock(1280 + 640 -> 640) + SpatialTransformer(640)
            cur = CatAndFreeSkip(cur, 5, 1280, 640);
            cur = ResBlockGpu(imageOps, "output_blocks.6.0", cur, tEmbGpu, 1920, h, w, 640);
            cur = SpatialTransformerGpu(imageOps, "output_blocks.6.1", cur, contextGpu, 640, h, w);

            // Block 7: ResBlock(640 + 640 -> 640) + SpatialTransformer(640)
            cur = CatAndFreeSkip(cur, 4, 640, 640);
            cur = ResBlockGpu(imageOps, "output_blocks.7.0", cur, tEmbGpu, 1280, h, w, 640);
            cur = SpatialTransformerGpu(imageOps, "output_blocks.7.1", cur, contextGpu, 640, h, w);

            // Block 8: ResBlock(640 + 320 -> 640) + SpatialTransformer(640) + Upsample + Conv
            cur = CatAndFreeSkip(cur, 3, 640, 320);
            cur = ResBlockGpu(imageOps, "output_blocks.8.0", cur, tEmbGpu, 960, h, w, 640);
            cur = SpatialTransformerGpu(imageOps, "output_blocks.8.1", cur, contextGpu, 640, h, w);
            var upsampled8 = imageOps.Upsample2xGpu(cur, 640, h, w);
            imageOps.Free(cur);
            h *= 2; w *= 2;
            cur = ConvGpuTensor(imageOps, "output_blocks.8.2.conv", upsampled8, 640, h, w, 640, 3);
            imageOps.Free(upsampled8);

            // Block 9: ResBlock(640 + 320 -> 320) + SpatialTransformer(320)
            cur = CatAndFreeSkip(cur, 2, 640, 320);
            cur = ResBlockGpu(imageOps, "output_blocks.9.0", cur, tEmbGpu, 960, h, w, 320);
            cur = SpatialTransformerGpu(imageOps, "output_blocks.9.1", cur, contextGpu, 320, h, w);

            // Block 10: ResBlock(320 + 320 -> 320) + SpatialTransformer(320)
            cur = CatAndFreeSkip(cur, 1, 320, 320);
            cur = ResBlockGpu(imageOps, "output_blocks.10.0", cur, tEmbGpu, 640, h, w, 320);
            cur = SpatialTransformerGpu(imageOps, "output_blocks.10.1", cur, contextGpu, 320, h, w);

            // Block 11: ResBlock(320 + 320 -> 320) + SpatialTransformer(320)
            cur = CatAndFreeSkip(cur, 0, 320, 320);
            cur = ResBlockGpu(imageOps, "output_blocks.11.0", cur, tEmbGpu, 640, h, w, 320);
            cur = SpatialTransformerGpu(imageOps, "output_blocks.11.1", cur, contextGpu, 320, h, w);

            // ── Final Output ────────────────────────────────────────────────────────
            var finalNorm = GroupNormSiluGpuTensor(imageOps, "out.0", cur, 320, h * w);
            imageOps.Free(cur);
            var finalOut = ConvGpuTensor(imageOps, "out.2", finalNorm, 320, h, w, 4, 3);
            imageOps.Free(finalNorm);

            imageOps.Free(tEmbGpu);

            var recordMs = swRecord.ElapsedMilliseconds;
            var swSubmit = System.Diagnostics.Stopwatch.StartNew();
            imageOps.EndBatch();
            var submitMs = swSubmit.ElapsedMilliseconds;
            Console.WriteLine($"[UNet.ForwardGpu] record={recordMs}ms, submitWait={submitMs}ms");
            batchSuccess = true;
            return finalOut;
        }
        finally
        {
            if (!batchSuccess)
            {
                try { imageOps.EndBatch(); } catch { }
                imageOps.Free(xGpu);
                imageOps.Free(tEmbGpu);
                if (controlDownGpu is not null)
                {
                    foreach (var ct in controlDownGpu)
                        if (ct is not null) imageOps.Free(ct);
                }
                if (controlMidGpu is not null) imageOps.Free(controlMidGpu);
            }
        }
    }

    public float[] Conv(string name, float[] x, int inC, int h, int w, int outC, int ksize, int stride = 1, int padding = -1)
    {
        var wF = GetWeight($"{name}.weight");
        var bF = TryGetWeight($"{name}.bias");

        if (_backend is null)
        {
            return DiffusionOps.Conv2D(x, wF, bF, 1, inC, h, w, outC, ksize, ksize, stride, padding);
        }

        // Perf: see the identical comment in SdxlUNet2DConditionModel.Conv -- the implicit-GEMM
        // shader only supports stride=1; this UNet's 3 downsample convs (input_blocks.{3,6,9}.0.op)
        // fall through to the CPU-im2col+Sgemm path below unchanged.
        if (stride == 1 && _imageOps is not null)
            return ConvNative(_imageOps, name, wF, bF, x, inC, h, w, outC, ksize, padding);

        if (padding < 0) padding = (ksize - 1) / 2;
        int outH = (h + 2 * padding - ksize) / stride + 1;
        int outW = (w + 2 * padding - ksize) / stride + 1;
        int hw = outH * outW;
        int kPts = inC * ksize * ksize;

        var wGpu = GetGpuWeight($"{name}.weight", wF);

        int chunkRows = kPts > 0 ? Math.Max(1, Math.Min(outH, MaxColChunkFloats / (outW * kPts))) : outH;
        var output = new float[outC * hw];

        var colBuf = ArrayPool<float>.Shared.Rent(chunkRows * outW * kPts);
        var resBuf = ArrayPool<float>.Shared.Rent(chunkRows * outW * outC);

        try
        {
            for (int rowStart = 0; rowStart < outH; rowStart += chunkRows)
            {
                int rowEnd = Math.Min(rowStart + chunkRows, outH);
                int chunkH = rowEnd - rowStart;
                int chunkHW = chunkH * outW;
                int colSize = chunkHW * kPts;

                // Build im2col for this chunk (row-parallel -- see the identical fix + rationale
                // in VaeDecoder.Im2ColChunk / SdxlUNet2DConditionModel.Conv).
                Parallel.For(rowStart, rowEnd, oh =>
                {
                    int rowBase = (oh - rowStart) * outW * kPts;
                    int ih0 = oh * stride - padding;
                    for (int ow = 0; ow < outW; ow++)
                    {
                        int idx = rowBase + ow * kPts;
                        int iw0 = ow * stride - padding;
                        for (int ic = 0; ic < inC; ic++)
                        {
                            int inChannelBase = ic * h * w;
                            for (int kh = 0; kh < ksize; kh++)
                            {
                                int ih = ih0 + kh;
                                if ((uint)ih < (uint)h)
                                {
                                    int inRow = inChannelBase + ih * w;
                                    for (int kw = 0; kw < ksize; kw++)
                                    {
                                        int iw = iw0 + kw;
                                        colBuf[idx++] = ((uint)iw < (uint)w) ? x[inRow + iw] : 0f;
                                    }
                                }
                                else
                                {
                                    for (int kw = 0; kw < ksize; kw++)
                                        colBuf[idx++] = 0f;
                                }
                            }
                        }
                    }
                });

                // GPU SGEMM
                var colGpu = _backend.Upload(colBuf.AsSpan(0, colSize), TensorShape.D1(colSize));
                var cGpu = _backend.Allocate(TensorShape.D1(chunkHW * outC));
                try
                {
                    // Perf: Sgemm's Dispatch() already fence-waits for this specific dispatch
                    // before returning; the explicit Synchronize() (full vkDeviceWaitIdle()) was
                    // pure redundant overhead (see the same finding in SdxlUNet2DConditionModel).
                    _backend.Sgemm(cGpu, colGpu, wGpu, chunkHW, kPts, outC);
                    _backend.Download(cGpu, resBuf.AsSpan(0, chunkHW * outC));
                }
                finally
                {
                    _backend.Free(colGpu);
                    _backend.Free(cGpu);
                }

                // Perf: same class of fix as the im2col gather above -- parallelize this
                // transpose+bias-add write-back across `pos` (disjoint output locations per pos).
                int basePos = rowStart * outW;
                Parallel.For(0, chunkHW, pos =>
                {
                    int absPos = basePos + pos;
                    for (int oc = 0; oc < outC; oc++)
                        output[oc * hw + absPos] = resBuf[pos * outC + oc] + (bF is not null ? bF[oc] : 0f);
                });
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(colBuf);
            ArrayPool<float>.Shared.Return(resBuf);
        }

        return output;
    }

    public float[] Lin(string name, float[] x, int n, int inDim, int outDim)
    {
        var wF = GetWeight($"{name}.weight");
        var bF = TryGetWeight($"{name}.bias");

        if (_backend is null)
        {
            return DiffusionOps.Linear(x, wF, bF, n, inDim, outDim);
        }

        var wGpu = GetGpuWeight($"{name}.weight", wF);
        var xGpu = _backend.Upload(x.AsSpan(0, n * inDim), TensorShape.D1(n * inDim));
        var cGpu = _backend.Allocate(TensorShape.D1(n * outDim));
        var result = new float[n * outDim];

        try
        {
            // Perf: see the identical comment in Conv() above.
            _backend.Sgemm(cGpu, xGpu, wGpu, n, inDim, outDim);
            _backend.Download(cGpu, result);
        }
        finally
        {
            _backend.Free(xGpu);
            _backend.Free(cGpu);
        }

        if (bF is not null)
        {
            // Perf: vectorized (SIMD) per-row bias add instead of a scalar inner loop.
            Parallel.For(0, n, i =>
            {
                var row = result.AsSpan(i * outDim, outDim);
                TensorPrimitives.Add(row, bF, row);
            });
        }

        return result;
    }

    /// <summary>
    /// Computes sinusoidal timestep embedding and passes through 2-layer MLP.
    /// Evaluated on CPU with AVX2/AVX-512 SIMD to eliminate host-GPU round-trips for the 320-element vector.
    /// </summary>
    public float[] ComputeTimeEmbedding(float timestep)
    {
        int dim = ModelChannels; // 320
        var sinEmb = new float[dim];
        int half = dim / 2;
        float maxPeriod = 10000.0f;
        float logMaxPeriod = MathF.Log(maxPeriod);

        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-logMaxPeriod * i / half);
            float arg = timestep * freq;
            sinEmb[i]        = MathF.Cos(arg);
            sinEmb[half + i] = MathF.Sin(arg);
        }

        var w0 = GetWeight("time_embed.0.weight");
        var b0 = TryGetWeight("time_embed.0.bias");
        var emb = DiffusionOps.Linear(sinEmb, w0, b0, 1, dim, TimeEmbedDim);
        DiffusionOps.SiluInPlace(emb);

        var w2 = GetWeight("time_embed.2.weight");
        var b2 = TryGetWeight("time_embed.2.bias");
        return DiffusionOps.Linear(emb, w2, b2, 1, TimeEmbedDim, TimeEmbedDim);
    }

    /// <summary>
    /// ResBlock with GroupNorm, SiLU, Conv2D, timestep embedding projection, and residual.
    /// </summary>
    public float[] ResBlock(string prefix, float[] x, float[] tEmb, int inC, int outC, int h, int w)
    {
        // 1. in_layers: GroupNorm(32, inC) + SiLU + Conv2D(inC -> outC, 3x3)
        var gn1W = GetWeight($"{prefix}.in_layers.0.weight");
        var gn1B = GetWeight($"{prefix}.in_layers.0.bias");
        var hNorm = (float[])x.Clone();
        DiffusionOps.GroupNorm(hNorm, gn1W, gn1B, 1, inC, h, w, groups: 32);
        DiffusionOps.SiluInPlace(hNorm);

        var hOut = Conv($"{prefix}.in_layers.2", hNorm, inC, h, w, outC, 3);

        // 2. emb_layers: SiLU(tEmb) -> Linear(1280 -> outC) added spatially
        var tEmbAct = (float[])tEmb.Clone();
        DiffusionOps.SiluInPlace(tEmbAct);
        var tProj = Lin($"{prefix}.emb_layers.1", tEmbAct, 1, TimeEmbedDim, outC);

        // Perf: vectorized (SIMD) scalar-broadcast add per channel instead of a scalar loop.
        int spatial = h * w;
        for (int c = 0; c < outC; c++)
        {
            var slice = hOut.AsSpan(c * spatial, spatial);
            TensorPrimitives.Add(slice, tProj[c], slice);
        }

        // 3. out_layers: GroupNorm(32, outC) + SiLU + Conv2D(outC -> outC, 3x3)
        var gn2W = GetWeight($"{prefix}.out_layers.0.weight");
        var gn2B = GetWeight($"{prefix}.out_layers.0.bias");
        DiffusionOps.GroupNorm(hOut, gn2W, gn2B, 1, outC, h, w, groups: 32);
        DiffusionOps.SiluInPlace(hOut);

        hOut = Conv($"{prefix}.out_layers.3", hOut, outC, h, w, outC, 3);

        // 4. Skip connection (nin_shortcut if inC != outC)
        float[] xRes;
        if (TryGetWeight($"{prefix}.skip_connection.weight") is not null)
        {
            xRes = Conv($"{prefix}.skip_connection", x, inC, h, w, outC, 1, stride: 1, padding: 0);
        }
        else if (inC != outC)
        {
            throw new InvalidOperationException($"ResBlock {prefix} has inChannels ({inC}) != outChannels ({outC}) but no skip_connection weight.");
        }
        else
        {
            xRes = x;
        }

        // Residual add (vectorized)
        TensorPrimitives.Add(hOut, xRes, hOut);

        return hOut;
    }

    /// <summary>
    /// SpatialTransformer block (Self-Attention + Cross-Attention + GEGLU FeedForward).
    /// </summary>
    public float[] SpatialTransformer(string prefix, float[] x, float[] context, int c, int h, int w)
    {
        int hw = h * w;

        // 1. norm + proj_in (Conv2D 1x1)
        var normW = GetWeight($"{prefix}.norm.weight");
        var normB = GetWeight($"{prefix}.norm.bias");
        var xNorm = (float[])x.Clone();
        DiffusionOps.GroupNorm(xNorm, normW, normB, 1, c, h, w, groups: 32);

        var xProj = Conv($"{prefix}.proj_in", xNorm, c, h, w, c, 1, stride: 1, padding: 0);

        // Permute [1, C, H, W] -> [H*W, C] sequence
        var xSeq = new float[hw * c];
        for (int ch = 0; ch < c; ch++)
        {
            int chOff = ch * hw;
            for (int s = 0; s < hw; s++)
                xSeq[s * c + ch] = xProj[chOff + s];
        }

        // 2. Transformer Block (depth = 1 in SD 1.5)
        string tb = $"{prefix}.transformer_blocks.0";

        // A. Self-Attention:
        var saNormW = GetWeight($"{tb}.norm1.weight");
        var saNormB = GetWeight($"{tb}.norm1.bias");
        var saNorm = (float[])xSeq.Clone();
        DiffusionOps.LayerNorm(saNorm, saNormW, saNormB, c);

        var saQ = Lin($"{tb}.attn1.to_q", saNorm, hw, c, c);
        var saK = Lin($"{tb}.attn1.to_k", saNorm, hw, c, c);
        var saV = Lin($"{tb}.attn1.to_v", saNorm, hw, c, c);

        var saAttnOut = MultiHeadAttention(saQ, saK, saV, hw, hw, c, NumHeads);
        var saProjOut = Lin($"{tb}.attn1.to_out.0", saAttnOut, hw, c, c);

        TensorPrimitives.Add(xSeq, saProjOut, xSeq);

        // B. Cross-Attention (to CLIP text context: 77 tokens, 768 dim):
        var caNormW = GetWeight($"{tb}.norm2.weight");
        var caNormB = GetWeight($"{tb}.norm2.bias");
        var caNorm = (float[])xSeq.Clone();
        DiffusionOps.LayerNorm(caNorm, caNormW, caNormB, c);

        var caQ = Lin($"{tb}.attn2.to_q", caNorm, hw, c, c);
        var caK = Lin($"{tb}.attn2.to_k", context, 77, ContextDim, c);
        var caV = Lin($"{tb}.attn2.to_v", context, 77, ContextDim, c);

        var caAttnOut = MultiHeadAttention(caQ, caK, caV, hw, 77, c, NumHeads);
        var caProjOut = Lin($"{tb}.attn2.to_out.0", caAttnOut, hw, c, c);

        TensorPrimitives.Add(xSeq, caProjOut, xSeq);

        // C. Feed-Forward with GEGLU:
        var ffNormW = GetWeight($"{tb}.norm3.weight");
        var ffNormB = GetWeight($"{tb}.norm3.bias");
        var ffNorm = (float[])xSeq.Clone();
        DiffusionOps.LayerNorm(ffNorm, ffNormW, ffNormB, c);

        int mlpDim = c * 4;
        var ffH = Lin($"{tb}.ff.net.0.proj", ffNorm, hw, c, mlpDim * 2);
        var ffGated = new float[hw * mlpDim];
        Parallel.For(0, hw, s =>
        {
            int srcOff = s * mlpDim * 2;
            int dstOff = s * mlpDim;
            for (int d = 0; d < mlpDim; d++)
            {
                float val = ffH[srcOff + d];
                float gate = ffH[srcOff + mlpDim + d];
                float geluGate = 0.5f * gate * (1.0f + MathF.Tanh(0.79788456f * (gate + 0.044715f * gate * gate * gate)));
                ffGated[dstOff + d] = val * geluGate;
            }
        });

        var ffOut = Lin($"{tb}.ff.net.2", ffGated, hw, mlpDim, c);

        TensorPrimitives.Add(xSeq, ffOut, xSeq);

        // Permute [H*W, C] back to [1, C, H, W]
        var xSpatial = new float[hw * c];
        for (int ch = 0; ch < c; ch++)
        {
            int chOff = ch * hw;
            for (int s = 0; s < hw; s++)
                xSpatial[chOff + s] = xSeq[s * c + ch];
        }

        // proj_out (Conv2D 1x1) + residual with input x
        var projOut = Conv($"{prefix}.proj_out", xSpatial, c, h, w, c, 1, stride: 1, padding: 0);

        TensorPrimitives.Add(projOut, x, projOut);

        return projOut;
    }

    private static float[] MultiHeadAttention(float[] q, float[] k, float[] v, int qLen, int kvLen, int c, int nHeads)
        => DiffusionOps.MultiHeadAttention(q, k, v, qLen, kvLen, nHeads, c / nHeads);

    public float[] Forward(
        float[] x,
        float timestep,
        float[] context,
        int latH,
        int latW,
        IReadOnlyList<float[]>? controlDownResiduals = null,
        float[]? controlMidResidual = null,
        IReadOnlyList<CoreTensor>? controlDownGpu = null,
        CoreTensor? controlMidGpu = null)
    {
        var tEmb = ComputeTimeEmbedding(timestep);

        if (_imageOps is not null && _unetForwardResidencySupported != false)
        {
            try
            {
                var resultGpu = ForwardGpu(_imageOps, x, tEmb, context, latH, latW,
                    controlDownResiduals, controlMidResidual, controlDownGpu, controlMidGpu);
                var result = new float[4 * latH * latW];
                try
                {
                    _imageOps.Download(resultGpu, result);
                }
                finally
                {
                    _imageOps.Free(resultGpu);
                }
                _unetForwardResidencySupported = true;
                return result;
            }
            catch (NotSupportedException)
            {
                _unetForwardResidencySupported = false;
            }
        }

        if (controlDownResiduals is null && controlDownGpu is not null && _imageOps is not null)
        {
            var downloadedDown = new List<float[]>(controlDownGpu.Count);
            for (int i = 0; i < controlDownGpu.Count; i++)
            {
                var g = controlDownGpu[i];
                var buf = new float[(int)g.ElementCount];
                _imageOps.Download(g, buf);
                downloadedDown.Add(buf);
            }
            controlDownResiduals = downloadedDown;
        }

        if (controlMidResidual is null && controlMidGpu is not null && _imageOps is not null)
        {
            var buf = new float[(int)controlMidGpu.ElementCount];
            _imageOps.Download(controlMidGpu, buf);
            controlMidResidual = buf;
        }

        var savedInputs = new List<float[]>(12);

        // ── Input Blocks ────────────────────────────────────────────────────────
        int h = latH, w = latW;
        var cur = Conv("input_blocks.0.0", x, 4, h, w, 320, 3);
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.1.0", cur, tEmb, 320, 320, h, w);
        cur = SpatialTransformer("input_blocks.1.1", cur, context, 320, h, w);
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.2.0", cur, tEmb, 320, 320, h, w);
        cur = SpatialTransformer("input_blocks.2.1", cur, context, 320, h, w);
        savedInputs.Add(cur);

        // Block 3: Downsample (Conv2D 320 -> 320, stride 2)
        cur = Conv("input_blocks.3.0.op", cur, 320, h, w, 320, 3, stride: 2);
        h /= 2; w /= 2;
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.4.0", cur, tEmb, 320, 640, h, w);
        cur = SpatialTransformer("input_blocks.4.1", cur, context, 640, h, w);
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.5.0", cur, tEmb, 640, 640, h, w);
        cur = SpatialTransformer("input_blocks.5.1", cur, context, 640, h, w);
        savedInputs.Add(cur);

        // Block 6: Downsample (Conv2D 640 -> 640, stride 2)
        cur = Conv("input_blocks.6.0.op", cur, 640, h, w, 640, 3, stride: 2);
        h /= 2; w /= 2;
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.7.0", cur, tEmb, 640, 1280, h, w);
        cur = SpatialTransformer("input_blocks.7.1", cur, context, 1280, h, w);
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.8.0", cur, tEmb, 1280, 1280, h, w);
        cur = SpatialTransformer("input_blocks.8.1", cur, context, 1280, h, w);
        savedInputs.Add(cur);

        // Block 9: Downsample (Conv2D 1280 -> 1280, stride 2)
        cur = Conv("input_blocks.9.0.op", cur, 1280, h, w, 1280, 3, stride: 2);
        h /= 2; w /= 2;
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.10.0", cur, tEmb, 1280, 1280, h, w);
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.11.0", cur, tEmb, 1280, 1280, h, w);
        savedInputs.Add(cur);

        // Apply ControlNet down residuals to skip connections
        if (controlDownResiduals is not null)
        {
            for (int i = 0; i < Math.Min(savedInputs.Count, controlDownResiduals.Count); i++)
            {
                var res = controlDownResiduals[i];
                var inp = savedInputs[i];
                for (int j = 0; j < Math.Min(inp.Length, res.Length); j++)
                    inp[j] += res[j];
            }
        }

        // ── Middle Block ────────────────────────────────────────────────────────
        cur = ResBlock("middle_block.0", cur, tEmb, 1280, 1280, h, w);
        cur = SpatialTransformer("middle_block.1", cur, context, 1280, h, w);
        cur = ResBlock("middle_block.2", cur, tEmb, 1280, 1280, h, w);

        if (controlMidResidual is not null)
        {
            for (int j = 0; j < Math.Min(cur.Length, controlMidResidual.Length); j++)
                cur[j] += controlMidResidual[j];
        }

        // ── Output Blocks ───────────────────────────────────────────────────────
        static float[] CatSkip(float[] current, float[] skip, int curC, int skipC, int curH, int curW)
        {
            int hw = curH * curW;
            var cat = new float[(curC + skipC) * hw];
            Array.Copy(current, 0, cat, 0, curC * hw);
            Array.Copy(skip, 0, cat, curC * hw, skipC * hw);
            return cat;
        }

        // Block 0: ResBlock(1280 + 1280 -> 1280)
        cur = CatSkip(cur, savedInputs[11], 1280, 1280, h, w);
        cur = ResBlock("output_blocks.0.0", cur, tEmb, 2560, 1280, h, w);

        // Block 1: ResBlock(1280 + 1280 -> 1280)
        cur = CatSkip(cur, savedInputs[10], 1280, 1280, h, w);
        cur = ResBlock("output_blocks.1.0", cur, tEmb, 2560, 1280, h, w);

        // Block 2: ResBlock(1280 + 1280 -> 1280) + Upsample(1280 -> 1280)
        cur = CatSkip(cur, savedInputs[9], 1280, 1280, h, w);
        cur = ResBlock("output_blocks.2.0", cur, tEmb, 2560, 1280, h, w);
        cur = DiffusionOps.Upsample2x(cur, 1, 1280, h, w);
        h *= 2; w *= 2;
        cur = Conv("output_blocks.2.1.conv", cur, 1280, h, w, 1280, 3);

        // Block 3: ResBlock(1280 + 1280 -> 1280) + SpatialTransformer(1280)
        cur = CatSkip(cur, savedInputs[8], 1280, 1280, h, w);
        cur = ResBlock("output_blocks.3.0", cur, tEmb, 2560, 1280, h, w);
        cur = SpatialTransformer("output_blocks.3.1", cur, context, 1280, h, w);

        // Block 4: ResBlock(1280 + 1280 -> 1280) + SpatialTransformer(1280)
        cur = CatSkip(cur, savedInputs[7], 1280, 1280, h, w);
        cur = ResBlock("output_blocks.4.0", cur, tEmb, 2560, 1280, h, w);
        cur = SpatialTransformer("output_blocks.4.1", cur, context, 1280, h, w);

        // Block 5: ResBlock(1280 + 640 -> 1280) + SpatialTransformer(1280) + Upsample(1280 -> 1280)
        cur = CatSkip(cur, savedInputs[6], 1280, 640, h, w);
        cur = ResBlock("output_blocks.5.0", cur, tEmb, 1920, 1280, h, w);
        cur = SpatialTransformer("output_blocks.5.1", cur, context, 1280, h, w);
        cur = DiffusionOps.Upsample2x(cur, 1, 1280, h, w);
        h *= 2; w *= 2;
        cur = Conv("output_blocks.5.2.conv", cur, 1280, h, w, 1280, 3);

        // Block 6: ResBlock(1280 + 640 -> 640) + SpatialTransformer(640)
        cur = CatSkip(cur, savedInputs[5], 1280, 640, h, w);
        cur = ResBlock("output_blocks.6.0", cur, tEmb, 1920, 640, h, w);
        cur = SpatialTransformer("output_blocks.6.1", cur, context, 640, h, w);

        // Block 7: ResBlock(640 + 640 -> 640) + SpatialTransformer(640)
        cur = CatSkip(cur, savedInputs[4], 640, 640, h, w);
        cur = ResBlock("output_blocks.7.0", cur, tEmb, 1280, 640, h, w);
        cur = SpatialTransformer("output_blocks.7.1", cur, context, 640, h, w);

        // Block 8: ResBlock(640 + 320 -> 640) + SpatialTransformer(640) + Upsample(640 -> 640)
        cur = CatSkip(cur, savedInputs[3], 640, 320, h, w);
        cur = ResBlock("output_blocks.8.0", cur, tEmb, 960, 640, h, w);
        cur = SpatialTransformer("output_blocks.8.1", cur, context, 640, h, w);
        cur = DiffusionOps.Upsample2x(cur, 1, 640, h, w);
        h *= 2; w *= 2;
        cur = Conv("output_blocks.8.2.conv", cur, 640, h, w, 640, 3);

        // Block 9: ResBlock(640 + 320 -> 320) + SpatialTransformer(320)
        cur = CatSkip(cur, savedInputs[2], 640, 320, h, w);
        cur = ResBlock("output_blocks.9.0", cur, tEmb, 960, 320, h, w);
        cur = SpatialTransformer("output_blocks.9.1", cur, context, 320, h, w);

        // Block 10: ResBlock(320 + 320 -> 320) + SpatialTransformer(320)
        cur = CatSkip(cur, savedInputs[1], 320, 320, h, w);
        cur = ResBlock("output_blocks.10.0", cur, tEmb, 640, 320, h, w);
        cur = SpatialTransformer("output_blocks.10.1", cur, context, 320, h, w);

        // Block 11: ResBlock(320 + 320 -> 320) + SpatialTransformer(320)
        cur = CatSkip(cur, savedInputs[0], 320, 320, h, w);
        cur = ResBlock("output_blocks.11.0", cur, tEmb, 640, 320, h, w);
        cur = SpatialTransformer("output_blocks.11.1", cur, context, 320, h, w);

        // ── Final Output ────────────────────────────────────────────────────────
        var outGnW = GetWeight("out.0.weight");
        var outGnB = GetWeight("out.0.bias");
        DiffusionOps.GroupNorm(cur, outGnW, outGnB, 1, 320, h, w, groups: 32);
        DiffusionOps.SiluInPlace(cur);

        return Conv("out.2", cur, 320, h, w, 4, 3);
    }

    public void Dispose()
    {
        ClearContextCache();
        if (_gpuWeights is not null)
        {
            foreach (var t in _gpuWeights.Values) _backend!.Free(t);
            _gpuWeights.Clear();
        }
        if (_gpuWeightsNative is not null)
        {
            foreach (var t in _gpuWeightsNative.Values) _backend!.Free(t);
            _gpuWeightsNative.Clear();
        }
        if (_backend is not null)
        {
            foreach (var t in _gpuBiasCache.Values) _backend.Free(t);
            _gpuBiasCache.Clear();
        }
        _weightReader.Clear();
    }
}


