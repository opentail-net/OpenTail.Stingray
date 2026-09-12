using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.SDXL;

/// <summary>
/// Stable Diffusion XL (SDXL) UNet (2D Condition Model).
/// 3 Resolution Levels: 320 -> 640 -> 1280.
/// Cross-attention context dimension: 2048 (CLIP-L 768 + OpenCLIP-bigG 1280).
/// Addition embedding dimension: 2816 (pooled text 1280 + 6 coordinate/size embeddings 1536).
/// Head dimension: 64. Transformer depths: [0, 2, 10].
/// </summary>
public sealed class SdxlUNet2DConditionModel : IDisposable
{
    private readonly CachedWeightReader _weightReader;
    private readonly IComputeBackend? _backend;
    private readonly Dictionary<string, CoreTensor>? _gpuWeights;

    // Separate cache for the native implicit-GEMM Conv2d shader path (see ConvNative below). Its
    // weight buffer must be plain Float32 (the shader reads `float weight_data[]` directly, no
    // dtype dispatch) -- unlike _gpuWeights above, which stores Half-converted tensors for the
    // Sgemm mixed-precision path (see IImageOpsBackend.Conv2dImplicitGemm's doc comment for why
    // this shader beats naive-accumulation and CPU-im2col+Sgemm for this UNet's channel counts).
    private readonly Dictionary<string, CoreTensor>? _gpuWeightsNative;
    private readonly IImageOpsBackend? _imageOps;

    // Perf (2026-09-11): label_emb's [2816]->[1280] projection of `addEmbeds` (SDXL's
    // micro-conditioning vector: pooled text + width/height/crop Fourier embeddings) is INVARIANT
    // across an entire denoising loop -- SdxlPipeline builds condAddEmbeds/uncondAddEmbeds once,
    // before the loop, and passes the same array reference to every step's Forward() call. Only
    // the timestep-embedding half of ComputeTimeAndAddEmbedding actually varies per step. Without
    // this cache, every step recomputed the exact same two GPU Lin() round-trips for no reason.
    // Keyed by array reference (float[] has no Equals/GetHashCode override, so this Dictionary
    // already does reference-identity lookup) -- a cache hit requires the caller to keep passing
    // the SAME addEmbeds instance, which SdxlPipeline already does by construction.
    private readonly Dictionary<float[], float[]> _addEmbCache = new();

    private const int ModelChannels = 320;
    private const int TimeEmbedDim = 1280;
    private const int ContextDim = 2048;
    private const int HeadDim = 64;
    private const int AdmInChannels = 2816;
    private const int MaxColChunkFloats = 8 * 1024 * 1024;

    public SdxlUNet2DConditionModel(IWeightLoader weights, string prefix = "model.diffusion_model.", IComputeBackend? backend = null)
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

    private float[] GetWeight(string name) => _weightReader.Get(name);

    private float[]? TryGetWeight(string name) => _weightReader.TryGet(name);

    private CoreTensor GetGpuWeight(string name, float[] cpuWeight)
    {
        string fullName = _weightReader.Prefix + name;
        if (_gpuWeights!.TryGetValue(fullName, out var wGpu)) return wGpu;

        // Perf (2026-09-11): Sgemm's mixed-precision path (activation Float32 x weight Float16)
        // is only reachable when the WEIGHT tensor's DType is Float16 -- uploading every weight as
        // Float32 (the previous behavior here) silently forced every SDXL UNet matmul onto the
        // slowest full-fp32 shader even on backends that report BestSgemmPrecision==Fp16 (this
        // iGPU does). sd_xl_turbo_1.0_fp16.safetensors is fp16 on disk already, so converting the
        // already-upconverted CachedWeightReader float[] back to Half here is not a new precision
        // loss for that checkpoint (round-trips losslessly); for a genuinely fp32 checkpoint this
        // matches the same fp32-activation/fp16-weight tradeoff FluxDiT/ZImageDiT's own fp16
        // upload branch already makes elsewhere in this codebase.
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

    /// <summary>
    /// Conv2D via the native GPU implicit-GEMM shader (see IImageOpsBackend.Conv2dImplicitGemm)
    /// instead of the CPU-im2col+Sgemm path -- real GEMM-tiled compute efficiency with zero
    /// CPU-side gather/transpose. Requires stride=1 (guarded by the caller).
    /// </summary>
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
            // The shader always reads a bias buffer -- upload zeros when this conv has none.
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

    // ── GPU-resident (Tensor-in/Tensor-out) primitive layer -- docs/067 Stage 1 ──────────────
    // These mirror ConvNative/Conv above and Lin below exactly in math, but take and return
    // CoreTensor handles instead of float[] -- no Upload/Download for intermediate results, so a
    // chain of these (Stage 2's ResBlockGpu, Stage 3's SpatialTransformer residency) can run with
    // CPU crossings only at the true block boundary. Reuses the exact same cached GPU weights
    // (_gpuWeightsNative/_gpuWeights) and dispatches (Conv2dImplicitGemm/Sgemm) the non-resident
    // path already uses -- no new shader math beyond the two new broadcast-add ops.
    private readonly Dictionary<string, CoreTensor> _gpuBiasCache = new(StringComparer.Ordinal);

    private CoreTensor GetGpuBias(IComputeBackend backend, string name, float[] bF)
    {
        string fullName = _weightReader.Prefix + name;
        if (_gpuBiasCache.TryGetValue(fullName, out var bGpu)) return bGpu;
        bGpu = backend.Upload(bF.AsSpan(), TensorShape.D1(bF.Length));
        _gpuBiasCache[fullName] = bGpu;
        return bGpu;
    }

    /// <summary>Tensor-in/Tensor-out counterpart of <see cref="ConvNative"/> -- stride=1 only.</summary>
    private CoreTensor ConvGpuTensor(IImageOpsBackend imageOps, string name, CoreTensor xGpu, int inCh, int h, int w, int outCh, int k, int padding = -1)
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

        return imageOps.Conv2dImplicitGemm(xGpu, wGpu, bGpu, inCh, outCh, h, w, k, padding);
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

    /// <summary>Tensor-in/Tensor-out GroupNorm+SiLU. Caches the (invariant, small) weight/bias
    /// tensors instead of re-uploading them every call -- a real cost the CPU-only GroupNorm path
    /// never had at all (pure float[] math, no GPU involvement), so uncached uploads here were
    /// silently offsetting part of ResBlockGpu's own win. Measured: caching these turned a ~8%
    /// whole-run improvement into the real win recorded in PerformanceLeague.md's Stage 2 row.</summary>
    private CoreTensor GroupNormSiluGpuTensor(IImageOpsBackend imageOps, string prefix, CoreTensor xGpu, int c, int hw)
    {
        var gnW = GetGpuBias(imageOps, $"{prefix}.weight", GetWeight($"{prefix}.weight"));
        var gnB = GetGpuBias(imageOps, $"{prefix}.bias", GetWeight($"{prefix}.bias"));
        return imageOps.GroupNormSilu(xGpu, gnW, gnB, c, hw, groups: 32);
    }

    // Perf (2026-09-12, docs/067 Stage 2): full GPU residency for the UNet's own ResBlock (the
    // same pattern VaeDecoder.ResBlockGpu already proved out for the VAE decoder's ResBlock).
    // Probed once per backend and cached, same convention as VaeDecoder's _residencySupported.
    private bool? _unetResidencySupported;

    /// <summary>
    /// Fully GPU-resident UNet ResBlock: norm1→silu→conv1→(+tEmb broadcast)→norm2→silu→conv2→
    /// (+skip). Every intermediate stays a GPU Tensor; only the block's input/output cross the
    /// CPU boundary (both already do, via the caller's Upload/Download in <see cref="ResBlock"/>).
    /// </summary>
    private CoreTensor ResBlockGpu(IImageOpsBackend imageOps, string prefix, CoreTensor xGpu, float[] tEmb, int inCh, int h, int w, int outCh)
    {
        int hw = h * w;
        var t1 = GroupNormSiluGpuTensor(imageOps, $"{prefix}.in_layers.0", xGpu, inCh, hw);
        var t2 = ConvGpuTensor(imageOps, $"{prefix}.in_layers.2", t1, inCh, h, w, outCh, 3);
        imageOps.Free(t1);

        // Timestep-embedding injection: SiLU(tEmb) -> Lin -> broadcast-add across every spatial
        // position of t2. tEmb is a single row [1, TimeEmbedDim]; uploaded fresh per ResBlock call
        // (not yet cached across blocks within one step -- a real, later perf opportunity, not a
        // correctness concern for this stage).
        var tEmbAct = (float[])tEmb.Clone();
        DiffusionOps.SiluInPlace(tEmbAct);
        var tEmbGpu = imageOps.Upload(tEmbAct.AsSpan(), TensorShape.D1(tEmbAct.Length));
        CoreTensor tProjGpu;
        try
        {
            tProjGpu = LinGpuTensor($"{prefix}.emb_layers.1", tEmbGpu, 1, TimeEmbedDim, outCh);
        }
        finally
        {
            imageOps.Free(tEmbGpu);
        }
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

    public float[] Conv(string name, float[] x, int inC, int h, int w, int outC, int ksize, int stride = 1, int padding = -1)
    {
        var wF = GetWeight($"{name}.weight");
        var bF = TryGetWeight($"{name}.bias");

        if (_backend is null)
        {
            return DiffusionOps.Conv2D(x, wF, bF, 1, inC, h, w, outC, ksize, ksize, stride, padding);
        }

        // Perf (2026-09-11): the implicit-GEMM native shader only supports stride=1 (its output
        // resolution is always == input resolution). This UNet's two downsample convs
        // (input_blocks.{3,6}.0.op) use stride=2 -- those fall through to the CPU-im2col+Sgemm
        // path below unchanged; everywhere else (the overwhelming majority of conv calls: all
        // ResBlock convs, skip_connections, output_blocks convs) routes through the native shader.
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

                // Perf: this gather was single-threaded scalar code -- at the largest UNet
                // resolutions it does hundreds of millions of boundary-checked gathers per conv
                // while the GPU sits idle waiting for it. Each output row writes a disjoint,
                // directly-computable range of colBuf (row oh occupies
                // [(oh-rowStart)*outW*kPts, (oh-rowStart+1)*outW*kPts)), so rows parallelize
                // cleanly -- no shared mutable index (see the identical fix in VaeDecoder.Im2ColChunk).
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

                var colGpu = _backend.Upload(colBuf.AsSpan(0, colSize), TensorShape.D1(colSize));
                var cGpu = _backend.Allocate(TensorShape.D1(chunkHW * outC));
                try
                {
                    // Perf (2026-09-11): Sgemm's underlying Dispatch() already submits and
                    // fence-waits for THIS dispatch before returning (see VulkanBackend.Dispatch /
                    // ComputePipeline.Dispatch), and Download's own CopyBuffer does its own
                    // separate submit-and-wait for the transfer -- so this explicit Synchronize()
                    // (a full vkDeviceWaitIdle() across the ENTIRE device, not just this queue's
                    // work) was pure redundant overhead on every single conv call, every UNet
                    // block, every denoising step. Removing it changes no ordering guarantee: the
                    // Sgemm dispatch has already completed by the time this line is reached.
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
            // Perf: see the identical comment in Conv() above -- Sgemm's Dispatch() already
            // fence-waits for this specific dispatch; the explicit Synchronize() here was a
            // redundant full-device idle wait.
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
    /// Runs several Lin() projections against the SAME input in one GPU-resident pass: uploads `x`
    /// once (instead of once per projection) and reuses that upload across all of them. Added
    /// 2026-09-11 for SpatialTransformer's Q/K/V projections, which all read the same normed input
    /// (self-attention) or the same `context` (cross-attention's K and V) -- found while chasing
    /// the UNet denoise step's ~7.7x gap vs a real stable-diffusion.cpp reference, where
    /// SpatialTransformer's per-Lin() Upload/Download round-trips (up to ~100 per call at the
    /// deepest, depth=10 blocks) are the dominant remaining cost. This only removes the redundant
    /// upload/allocation, not a full attention-residency rewrite (which would need a new GPU
    /// softmax/attention kernel this codebase doesn't have yet for this non-causal, non-KV-cached
    /// shape) -- a smaller, safer, real win rather than a rushed bigger one.
    /// </summary>
    private float[][] LinMulti(string[] names, float[] x, int n, int inDim, int[] outDims)
    {
        if (_backend is null)
        {
            var cpuResults = new float[names.Length][];
            for (int i = 0; i < names.Length; i++)
                cpuResults[i] = Lin(names[i], x, n, inDim, outDims[i]);
            return cpuResults;
        }

        var xGpu = _backend.Upload(x.AsSpan(0, n * inDim), TensorShape.D1(n * inDim));
        var results = new float[names.Length][];
        try
        {
            for (int idx = 0; idx < names.Length; idx++)
            {
                var wF = GetWeight($"{names[idx]}.weight");
                var bF = TryGetWeight($"{names[idx]}.bias");
                var wGpu = GetGpuWeight($"{names[idx]}.weight", wF);
                int outDim = outDims[idx];
                var cGpu = _backend.Allocate(TensorShape.D1(n * outDim));
                var result = new float[n * outDim];
                try
                {
                    _backend.Sgemm(cGpu, xGpu, wGpu, n, inDim, outDim);
                    _backend.Download(cGpu, result);
                }
                finally
                {
                    _backend.Free(cGpu);
                }

                if (bF is not null)
                {
                    Parallel.For(0, n, i =>
                    {
                        var row = result.AsSpan(i * outDim, outDim);
                        TensorPrimitives.Add(row, bF, row);
                    });
                }
                results[idx] = result;
            }
        }
        finally
        {
            _backend.Free(xGpu);
        }
        return results;
    }

    public float[] ComputeTimeAndAddEmbedding(float timestep, float[] addEmbeds)
    {
        // 1. Timestep embedding: [320] -> [1280]
        int dim = ModelChannels;
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

        var tEmb = Lin("time_embed.0", sinEmb, 1, dim, TimeEmbedDim);
        DiffusionOps.SiluInPlace(tEmb);
        tEmb = Lin("time_embed.2", tEmb, 1, TimeEmbedDim, TimeEmbedDim);

        // 2. Addition embedding (label_emb): [2816] -> [1280] -- invariant across the whole
        // denoising loop for a fixed addEmbeds instance (see the class-level _addEmbCache comment).
        if (addEmbeds.Length == AdmInChannels)
        {
            if (!_addEmbCache.TryGetValue(addEmbeds, out var addEmb))
            {
                addEmb = Lin("label_emb.0.0", addEmbeds, 1, AdmInChannels, TimeEmbedDim);
                DiffusionOps.SiluInPlace(addEmb);
                addEmb = Lin("label_emb.0.2", addEmb, 1, TimeEmbedDim, TimeEmbedDim);
                _addEmbCache[addEmbeds] = addEmb;
            }

            TensorPrimitives.Add(tEmb, addEmb, tEmb);
        }

        return tEmb;
    }

    public float[] ResBlock(string prefix, float[] x, float[] tEmb, int inC, int outC, int h, int w)
    {
        // docs/067 Stage 2: try the fully GPU-resident path first (upload x once, every
        // intermediate stays a GPU Tensor, download once) -- probed once and cached, same
        // fallback discipline as VaeDecoder.ResBlockGpu.
        if (_imageOps is not null && _unetResidencySupported != false)
        {
            try
            {
                var xGpu = _imageOps.Upload(x.AsSpan(0, inC * h * w), TensorShape.D1(inC * h * w));
                CoreTensor resultGpu;
                try
                {
                    resultGpu = ResBlockGpu(_imageOps, prefix, xGpu, tEmb, inC, h, w, outC);
                }
                finally
                {
                    _imageOps.Free(xGpu);
                }
                var result = new float[outC * h * w];
                try
                {
                    _imageOps.Download(resultGpu, result);
                }
                finally
                {
                    _imageOps.Free(resultGpu);
                }
                _unetResidencySupported = true;
                return result;
            }
            catch (NotSupportedException)
            {
                _unetResidencySupported = false;
                // fall through to the CPU-orchestrated path below for this and all future calls.
            }
        }

        var gn1W = GetWeight($"{prefix}.in_layers.0.weight");
        var gn1B = GetWeight($"{prefix}.in_layers.0.bias");
        var hNorm = (float[])x.Clone();
        DiffusionOps.GroupNorm(hNorm, gn1W, gn1B, 1, inC, h, w, groups: 32);
        DiffusionOps.SiluInPlace(hNorm);

        var hOut = Conv($"{prefix}.in_layers.2", hNorm, inC, h, w, outC, 3);

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

        var gn2W = GetWeight($"{prefix}.out_layers.0.weight");
        var gn2B = GetWeight($"{prefix}.out_layers.0.bias");
        DiffusionOps.GroupNorm(hOut, gn2W, gn2B, 1, outC, h, w, groups: 32);
        DiffusionOps.SiluInPlace(hOut);

        hOut = Conv($"{prefix}.out_layers.3", hOut, outC, h, w, outC, 3);

        float[] xRes;
        if (TryGetWeight($"{prefix}.skip_connection.weight") is not null)
        {
            xRes = Conv($"{prefix}.skip_connection", x, inC, h, w, outC, 1, stride: 1, padding: 0);
        }
        else if (inC != outC)
        {
            throw new InvalidOperationException($"ResBlock {prefix} has inC ({inC}) != outC ({outC}) without skip_connection.");
        }
        else
        {
            xRes = x;
        }

        // Perf: vectorized (SIMD) elementwise add instead of a scalar loop -- same math.
        TensorPrimitives.Add(hOut, xRes, hOut);

        return hOut;
    }

    // Perf (2026-09-12, docs/067 Stage 3a): full GPU residency for SpatialTransformer, EXCEPT the
    // attention math itself, which remains a deliberate, explicitly-labeled CPU island (download
    // Q/K/V -> DiffusionOps.MultiHeadAttention -> re-upload). This isolates a real measurement:
    // how much of this block's cost is the ~100 Linear-projection Upload/Download round-trips per
    // call at the deepest (depth=10) blocks, versus the attention math itself. Probed once and
    // cached, same convention as _unetResidencySupported.
    private bool? _spatialTransformerResidencySupported;

    // Stage 3b (docs/067): re-test MultiHeadAttentionTiled now that Q/K/V are already GPU-resident
    // (LinGpuTensor's output, never downloaded) -- a genuinely different experiment from this
    // shader's two prior regressions, both of which were measured with the surrounding per-op
    // Upload/Download tax still present on every OTHER op in the block. Probed once and cached,
    // independent from _spatialTransformerResidencySupported so a GPU-attention regression falls
    // back to Stage 3a's CPU-island resident chain, not all the way to the fully non-resident path.
    private bool? _residentGpuAttentionSupported;

    private CoreTensor CpuAttentionIsland(IImageOpsBackend imageOps, CoreTensor qGpu, CoreTensor kGpu, CoreTensor vGpu, int qSeq, int kvSeq, int c, int nHeads)
    {
        // CPU ISLAND (docs/067 CPU-island discipline: TEMPORARY DIAGNOSTIC FALLBACK, not model
        // logic or a performance bug), used when Stage 3b's GPU attention is unsupported or was
        // measured to regress for this backend.
        var q = new float[qSeq * c];
        var k = new float[kvSeq * c];
        var v = new float[kvSeq * c];
        imageOps.Download(qGpu, q);
        imageOps.Download(kGpu, k);
        imageOps.Download(vGpu, v);
        var attnOut = DiffusionOps.MultiHeadAttention(q, k, v, qSeq, kvSeq, nHeads, HeadDim);
        return imageOps.Upload(attnOut.AsSpan(), TensorShape.D1(attnOut.Length));
    }

    /// <summary>Stage 3b: tries the GPU-resident tiled attention shader first (zero CPU round-trip
    /// -- Q/K/V are already resident GPU tensors), falling back to <see cref="CpuAttentionIsland"/>
    /// if unsupported. NOT yet measured to win or lose in this context -- do not assume either
    /// outcome; the caller/PerformanceLeague.md records the real measurement.</summary>
    private CoreTensor AttentionIsland(IImageOpsBackend imageOps, CoreTensor qGpu, CoreTensor kGpu, CoreTensor vGpu, int qSeq, int kvSeq, int c, int nHeads)
    {
        if (_residentGpuAttentionSupported != false)
        {
            try
            {
                var result = imageOps.MultiHeadAttentionTiled(qGpu, kGpu, vGpu, qSeq, kvSeq, nHeads, HeadDim);
                _residentGpuAttentionSupported = true;
                return result;
            }
            catch (NotSupportedException)
            {
                _residentGpuAttentionSupported = false;
            }
        }
        return CpuAttentionIsland(imageOps, qGpu, kGpu, vGpu, qSeq, kvSeq, c, nHeads);
    }

    private CoreTensor SpatialTransformerGpu(IImageOpsBackend imageOps, string prefix, CoreTensor xGpu, CoreTensor contextGpu, int c, int depth, int h, int w)
    {
        int hw = h * w;
        int nHeads = c / HeadDim;

        var normW = GetGpuBias(imageOps, $"{prefix}.norm.weight", GetWeight($"{prefix}.norm.weight"));
        var normB = GetGpuBias(imageOps, $"{prefix}.norm.bias", GetWeight($"{prefix}.norm.bias"));
        // Plain GroupNorm, NO SiLU -- matches the CPU path's DiffusionOps.GroupNorm exactly.
        // GroupNormSilu (used by ResBlock) is NOT usable here: it fuses an activation this block's
        // pre-proj_in norm doesn't have in the real model.
        var xNorm = imageOps.GroupNormGpu(xGpu, normW, normB, c, hw);

        var xSeq = imageOps.PermuteChwToHwc(xNorm, c, hw);
        imageOps.Free(xNorm);

        var projIn = LinGpuTensor($"{prefix}.proj_in", xSeq, hw, c, c);
        imageOps.Free(xSeq);
        xSeq = projIn;

        for (int d = 0; d < depth; d++)
        {
            string tb = $"{prefix}.transformer_blocks.{d}";

            // 1. Self-attention
            var saNormW = GetGpuBias(imageOps, $"{tb}.norm1.weight", GetWeight($"{tb}.norm1.weight"));
            var saNormB = GetGpuBias(imageOps, $"{tb}.norm1.bias", GetWeight($"{tb}.norm1.bias"));
            var saNorm = imageOps.LayerNormGpu(xSeq, saNormW, saNormB, hw, c);

            var saQ = LinGpuTensor($"{tb}.attn1.to_q", saNorm, hw, c, c);
            var saK = LinGpuTensor($"{tb}.attn1.to_k", saNorm, hw, c, c);
            var saV = LinGpuTensor($"{tb}.attn1.to_v", saNorm, hw, c, c);
            imageOps.Free(saNorm);

            var saAttnOut = AttentionIsland(imageOps, saQ, saK, saV, hw, hw, c, nHeads);
            imageOps.Free(saQ); imageOps.Free(saK); imageOps.Free(saV);

            var saProjOut = LinGpuTensor($"{tb}.attn1.to_out.0", saAttnOut, hw, c, c);
            imageOps.Free(saAttnOut);
            imageOps.AddInPlace(xSeq, saProjOut);
            imageOps.Free(saProjOut);

            // 2. Cross-attention to text context
            var caNormW = GetGpuBias(imageOps, $"{tb}.norm2.weight", GetWeight($"{tb}.norm2.weight"));
            var caNormB = GetGpuBias(imageOps, $"{tb}.norm2.bias", GetWeight($"{tb}.norm2.bias"));
            var caNorm = imageOps.LayerNormGpu(xSeq, caNormW, caNormB, hw, c);

            var caQ = LinGpuTensor($"{tb}.attn2.to_q", caNorm, hw, c, c);
            imageOps.Free(caNorm);
            var caK = LinGpuTensor($"{tb}.attn2.to_k", contextGpu, 77, ContextDim, c);
            var caV = LinGpuTensor($"{tb}.attn2.to_v", contextGpu, 77, ContextDim, c);

            var caAttnOut = AttentionIsland(imageOps, caQ, caK, caV, hw, 77, c, nHeads);
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
        }

        var projOut = LinGpuTensor($"{prefix}.proj_out", xSeq, hw, c, c);
        imageOps.Free(xSeq);

        var xSpatial = imageOps.PermuteHwcToChw(projOut, c, hw);
        imageOps.Free(projOut);
        imageOps.AddInPlace(xSpatial, xGpu);
        return xSpatial;
    }

    public float[] SpatialTransformer(string prefix, float[] x, float[] context, int c, int depth, int h, int w)
    {
        if (_imageOps is not null && _spatialTransformerResidencySupported != false)
        {
            try
            {
                int hw0 = h * w;
                var xGpu = _imageOps.Upload(x.AsSpan(0, c * hw0), TensorShape.D1(c * hw0));
                var contextGpu = _imageOps.Upload(context.AsSpan(0, 77 * ContextDim), TensorShape.D1(77 * ContextDim));
                CoreTensor resultGpu;
                try
                {
                    resultGpu = SpatialTransformerGpu(_imageOps, prefix, xGpu, contextGpu, c, depth, h, w);
                }
                finally
                {
                    _imageOps.Free(xGpu);
                    _imageOps.Free(contextGpu);
                }
                var result = new float[c * hw0];
                try
                {
                    _imageOps.Download(resultGpu, result);
                }
                finally
                {
                    _imageOps.Free(resultGpu);
                }
                _spatialTransformerResidencySupported = true;
                return result;
            }
            catch (NotSupportedException)
            {
                _spatialTransformerResidencySupported = false;
            }
        }
        int hw = h * w;
        int nHeads = c / HeadDim;

        // GroupNorm + Linear proj_in
        var normW = GetWeight($"{prefix}.norm.weight");
        var normB = GetWeight($"{prefix}.norm.bias");
        var xNorm = (float[])x.Clone();
        DiffusionOps.GroupNorm(xNorm, normW, normB, 1, c, h, w, groups: 32);

        // Permute [1, C, H, W] -> [H*W, C] sequence
        var xSeq = new float[hw * c];
        for (int ch = 0; ch < c; ch++)
        {
            int chOff = ch * hw;
            for (int s = 0; s < hw; s++)
                xSeq[s * c + ch] = xNorm[chOff + s];
        }

        xSeq = Lin($"{prefix}.proj_in", xSeq, hw, c, c);

        for (int d = 0; d < depth; d++)
        {
            string tb = $"{prefix}.transformer_blocks.{d}";

            // 1. Self-Attention
            var saNormW = GetWeight($"{tb}.norm1.weight");
            var saNormB = GetWeight($"{tb}.norm1.bias");
            var saNorm = (float[])xSeq.Clone();
            DiffusionOps.LayerNorm(saNorm, saNormW, saNormB, c);

            // Perf: Q/K/V all read the same `saNorm` input -- upload it once instead of 3 times.
            var saQkv = LinMulti(
                [$"{tb}.attn1.to_q", $"{tb}.attn1.to_k", $"{tb}.attn1.to_v"],
                saNorm, hw, c, [c, c, c]);
            var saQ = saQkv[0]; var saK = saQkv[1]; var saV = saQkv[2];

            var saAttnOut = MultiHeadAttention(saQ, saK, saV, hw, hw, c, nHeads, HeadDim);
            var saProjOut = Lin($"{tb}.attn1.to_out.0", saAttnOut, hw, c, c);

            TensorPrimitives.Add(xSeq, saProjOut, xSeq);

            // 2. Cross-Attention to [77, 2048] text context
            var caNormW = GetWeight($"{tb}.norm2.weight");
            var caNormB = GetWeight($"{tb}.norm2.bias");
            var caNorm = (float[])xSeq.Clone();
            DiffusionOps.LayerNorm(caNorm, caNormW, caNormB, c);

            var caQ = Lin($"{tb}.attn2.to_q", caNorm, hw, c, c);
            // Perf: K and V both read the same `context` input -- upload it once instead of twice.
            var caKv = LinMulti(
                [$"{tb}.attn2.to_k", $"{tb}.attn2.to_v"],
                context, 77, ContextDim, [c, c]);
            var caK = caKv[0]; var caV = caKv[1];

            var caAttnOut = MultiHeadAttention(caQ, caK, caV, hw, 77, c, nHeads, HeadDim);
            var caProjOut = Lin($"{tb}.attn2.to_out.0", caAttnOut, hw, c, c);

            TensorPrimitives.Add(xSeq, caProjOut, xSeq);

            // 3. GEGLU FeedForward
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
                for (int m = 0; m < mlpDim; m++)
                {
                    float val = ffH[srcOff + m];
                    float gate = ffH[srcOff + mlpDim + m];
                    float geluGate = 0.5f * gate * (1.0f + MathF.Tanh(0.79788456f * (gate + 0.044715f * gate * gate * gate)));
                    ffGated[dstOff + m] = val * geluGate;
                }
            });

            var ffOut = Lin($"{tb}.ff.net.2", ffGated, hw, mlpDim, c);
            TensorPrimitives.Add(xSeq, ffOut, xSeq);
        }

        xSeq = Lin($"{prefix}.proj_out", xSeq, hw, c, c);

        // Permute back to [1, C, H, W]
        var xSpatial = new float[hw * c];
        for (int ch = 0; ch < c; ch++)
        {
            int chOff = ch * hw;
            for (int s = 0; s < hw; s++)
                xSpatial[chOff + s] = xSeq[s * c + ch];
        }

        TensorPrimitives.Add(xSpatial, x, xSpatial);

        return xSpatial;
    }

    // Perf note (2026-09-11): tried routing this through the NAIVE GPU MultiHeadAttention shader
    // (verified numerically correct against the CPU reference in
    // MultiHeadAttentionGpuParityTests, and confirmed still-correct in this real pipeline too --
    // pixel-identical output). Measured real weights/timing and it was a clear REGRESSION:
    // steady-state denoise step ~19.5s -> ~26.8s. Root cause: the shader's one-thread-per-
    // (query,head) design has each of qSeq*nHeads threads independently re-read the ENTIRE K/V
    // sequence from global memory with zero tiling/shared-memory reuse -- for self-attention at
    // hw=4096 that's on the order of 10+ billion redundant reads for a single call, far more
    // bandwidth-inefficient than the CPU's cache-friendlier SIMD-parallelized version. Same class
    // of lesson as the earlier naive Conv2d-shader regression, but worse here because attention's
    // O(seq^2) math punishes "no tiling" much harder than convolution's bounded kernel size did.
    // The naive shader+parity-test remain in the codebase as a reference building block.
    //
    // Follow-up (2026-09-11): implemented a properly TILED (shared-memory-blocked, flash-
    // attention-style) shader, MultiHeadAttentionTiled, following the row/column-tile + online-
    // softmax technique from ggml-vulkan's real flash_attn.comp (reviewed, not copied -- rewritten
    // for this codebase's [seq, numHeads*headDim] interleaved-head layout instead of ggml's
    // per-head-buffer layout). Verified correct in isolation against the CPU reference within a
    // 5e-3 tolerance (tiled accumulation reorders float rounding vs. the CPU's sequential sum)
    // across 6 shapes including multi-tile qSeq=4096 and 1024x1024 cases -- see
    // MultiHeadAttentionTiledGpuParityTests, which still passes.
    //
    // Wired in here (same probe-once/cache/fallback pattern as VaeDecoder.ResBlockGpu) and
    // measured against real weights: an even WORSE regression than the naive shader. Steady-state
    // denoise step ~19.5s (CPU baseline) -> 52-72s per step, and VAE decode (which has no
    // attention at all -- unrelated to this change, so this is either GPU memory/scheduler
    // contention from the many small tiled-attention dispatches queued ahead of it, or noise from
    // running on a shared iGPU under load) blew up to 361s from a ~19.6s baseline. Reverted back
    // to the CPU path -- do not re-wire without first fixing the tiled kernel's per-call overhead
    // (16 threads/query-row x 256 threads/workgroup means small dispatches are dominated by
    // per-dispatch fixed cost on this iGPU, not by the O(seq^2) math the tiling was meant to fix)
    // and re-measuring on hardware with real dedicated VRAM bandwidth. Shader + parity test remain
    // as a correct-but-not-yet-fast building block.
    private static float[] MultiHeadAttention(float[] q, float[] k, float[] v, int qLen, int kvLen, int c, int nHeads, int headDim)
        => DiffusionOps.MultiHeadAttention(q, k, v, qLen, kvLen, nHeads, headDim);

    // docs/067 Stage 4: cross-block residency. Only 2 real CPU islands remain in the whole
    // forward pass: the two stride=2 downsample convs (Conv2dImplicitGemm/ConvGpuTensor only
    // support stride=1 -- a real, bounded implementation gap, not a performance bug, per the plan's
    // CPU-island discipline). Everything else (every ResBlock, every SpatialTransformer, both
    // Upsample2x + their follow-up convs, both channel concats, the final GroupNorm+SiLU+Conv)
    // stays GPU-resident from the single input Upload to the single final Download.
    private bool? _unetForwardResidencySupported;

    private CoreTensor StridedConvCpuIsland(IImageOpsBackend imageOps, string name, CoreTensor xGpu, int inCh, int h, int w, int outCh, int ksize, int stride)
    {
        // CPU ISLAND (docs/067 CPU-island discipline: KNOWN IMPLEMENTATION GAP, not model logic or
        // a performance bug) -- neither Conv2dImplicitGemm nor ConvGpuTensor support stride>1.
        // Only 2 calls total per Forward() (the UNet's two downsample convs).
        var x = new float[inCh * h * w];
        imageOps.Download(xGpu, x);
        var result = Conv(name, x, inCh, h, w, outCh, ksize, stride: stride);
        return imageOps.Upload(result.AsSpan(), TensorShape.D1(result.Length));
    }

    private CoreTensor ForwardGpu(IImageOpsBackend imageOps, float[] x, float[] tEmb, float[] context, int latH, int latW)
    {
        var contextGpu = imageOps.Upload(context.AsSpan(0, 77 * ContextDim), TensorShape.D1(77 * ContextDim));
        var savedInputs = new List<CoreTensor>(9);
        try
        {
            int h = latH, w = latW;
            var xGpu = imageOps.Upload(x.AsSpan(0, 4 * h * w), TensorShape.D1(4 * h * w));
            var cur = ConvGpuTensor(imageOps, "input_blocks.0.0", xGpu, 4, h, w, 320, 3);
            imageOps.Free(xGpu);
            savedInputs.Add(cur);

            cur = ResBlockGpu(imageOps, "input_blocks.1.0", cur, tEmb, 320, h, w, 320);
            savedInputs.Add(cur);

            cur = ResBlockGpu(imageOps, "input_blocks.2.0", cur, tEmb, 320, h, w, 320);
            savedInputs.Add(cur);

            cur = StridedConvCpuIsland(imageOps, "input_blocks.3.0.op", cur, 320, h, w, 320, 3, stride: 2);
            h /= 2; w /= 2;
            savedInputs.Add(cur);

            cur = ResBlockGpu(imageOps, "input_blocks.4.0", cur, tEmb, 320, h, w, 640);
            cur = SpatialTransformerGpu(imageOps, "input_blocks.4.1", cur, contextGpu, 640, depth: 2, h, w);
            savedInputs.Add(cur);

            cur = ResBlockGpu(imageOps, "input_blocks.5.0", cur, tEmb, 640, h, w, 640);
            cur = SpatialTransformerGpu(imageOps, "input_blocks.5.1", cur, contextGpu, 640, depth: 2, h, w);
            savedInputs.Add(cur);

            cur = StridedConvCpuIsland(imageOps, "input_blocks.6.0.op", cur, 640, h, w, 640, 3, stride: 2);
            h /= 2; w /= 2;
            savedInputs.Add(cur);

            cur = ResBlockGpu(imageOps, "input_blocks.7.0", cur, tEmb, 640, h, w, 1280);
            cur = SpatialTransformerGpu(imageOps, "input_blocks.7.1", cur, contextGpu, 1280, depth: 10, h, w);
            savedInputs.Add(cur);

            cur = ResBlockGpu(imageOps, "input_blocks.8.0", cur, tEmb, 1280, h, w, 1280);
            cur = SpatialTransformerGpu(imageOps, "input_blocks.8.1", cur, contextGpu, 1280, depth: 10, h, w);
            savedInputs.Add(cur);

            cur = ResBlockGpu(imageOps, "middle_block.0", cur, tEmb, 1280, h, w, 1280);
            cur = SpatialTransformerGpu(imageOps, "middle_block.1", cur, contextGpu, 1280, depth: 10, h, w);
            cur = ResBlockGpu(imageOps, "middle_block.2", cur, tEmb, 1280, h, w, 1280);

            // Each skip is freed immediately after its concat consumes it (docs/067 Stage 4 memory
            // discipline) rather than holding all 9 resident for the whole pass.
            CoreTensor CatAndFreeSkip(CoreTensor current, int idx, int curC, int skipC)
            {
                var skip = savedInputs[idx];
                var result = imageOps.CatChannels(current, curC, skip, skipC, h * w);
                imageOps.Free(current);
                imageOps.Free(skip);
                return result;
            }

            cur = CatAndFreeSkip(cur, 8, 1280, 1280);
            cur = ResBlockGpu(imageOps, "output_blocks.0.0", cur, tEmb, 2560, h, w, 1280);
            cur = SpatialTransformerGpu(imageOps, "output_blocks.0.1", cur, contextGpu, 1280, depth: 10, h, w);

            cur = CatAndFreeSkip(cur, 7, 1280, 1280);
            cur = ResBlockGpu(imageOps, "output_blocks.1.0", cur, tEmb, 2560, h, w, 1280);
            cur = SpatialTransformerGpu(imageOps, "output_blocks.1.1", cur, contextGpu, 1280, depth: 10, h, w);

            cur = CatAndFreeSkip(cur, 6, 1280, 640);
            cur = ResBlockGpu(imageOps, "output_blocks.2.0", cur, tEmb, 1920, h, w, 1280);
            cur = SpatialTransformerGpu(imageOps, "output_blocks.2.1", cur, contextGpu, 1280, depth: 10, h, w);
            var upsampled2 = imageOps.Upsample2xGpu(cur, 1280, h, w);
            imageOps.Free(cur);
            h *= 2; w *= 2;
            cur = ConvGpuTensor(imageOps, "output_blocks.2.2.conv", upsampled2, 1280, h, w, 1280, 3);
            imageOps.Free(upsampled2);

            cur = CatAndFreeSkip(cur, 5, 1280, 640);
            cur = ResBlockGpu(imageOps, "output_blocks.3.0", cur, tEmb, 1920, h, w, 640);
            cur = SpatialTransformerGpu(imageOps, "output_blocks.3.1", cur, contextGpu, 640, depth: 2, h, w);

            cur = CatAndFreeSkip(cur, 4, 640, 640);
            cur = ResBlockGpu(imageOps, "output_blocks.4.0", cur, tEmb, 1280, h, w, 640);
            cur = SpatialTransformerGpu(imageOps, "output_blocks.4.1", cur, contextGpu, 640, depth: 2, h, w);

            cur = CatAndFreeSkip(cur, 3, 640, 320);
            cur = ResBlockGpu(imageOps, "output_blocks.5.0", cur, tEmb, 960, h, w, 640);
            cur = SpatialTransformerGpu(imageOps, "output_blocks.5.1", cur, contextGpu, 640, depth: 2, h, w);
            var upsampled5 = imageOps.Upsample2xGpu(cur, 640, h, w);
            imageOps.Free(cur);
            h *= 2; w *= 2;
            cur = ConvGpuTensor(imageOps, "output_blocks.5.2.conv", upsampled5, 640, h, w, 640, 3);
            imageOps.Free(upsampled5);

            cur = CatAndFreeSkip(cur, 2, 640, 320);
            cur = ResBlockGpu(imageOps, "output_blocks.6.0", cur, tEmb, 960, h, w, 320);

            cur = CatAndFreeSkip(cur, 1, 320, 320);
            cur = ResBlockGpu(imageOps, "output_blocks.7.0", cur, tEmb, 640, h, w, 320);

            cur = CatAndFreeSkip(cur, 0, 320, 320);
            cur = ResBlockGpu(imageOps, "output_blocks.8.0", cur, tEmb, 640, h, w, 320);

            var finalNorm = GroupNormSiluGpuTensor(imageOps, "out.0", cur, 320, h * w);
            imageOps.Free(cur);
            var finalOut = ConvGpuTensor(imageOps, "out.2", finalNorm, 320, h, w, 4, 3);
            imageOps.Free(finalNorm);
            return finalOut;
        }
        finally
        {
            imageOps.Free(contextGpu);
        }
    }

    public float[] Forward(float[] x, float timestep, float[] context, float[] addEmbeds, int latH, int latW)
    {
        var tEmb = ComputeTimeAndAddEmbedding(timestep, addEmbeds);

        if (_imageOps is not null && _unetForwardResidencySupported != false)
        {
            try
            {
                var resultGpu = ForwardGpu(_imageOps, x, tEmb, context, latH, latW);
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

        var savedInputs = new List<float[]>(9);

        // ── Input Blocks ────────────────────────────────────────────────────────
        int h = latH, w = latW;
        var cur = Conv("input_blocks.0.0", x, 4, h, w, 320, 3);
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.1.0", cur, tEmb, 320, 320, h, w);
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.2.0", cur, tEmb, 320, 320, h, w);
        savedInputs.Add(cur);

        // Block 3: Downsample (320 -> 320, stride 2)
        cur = Conv("input_blocks.3.0.op", cur, 320, h, w, 320, 3, stride: 2);
        h /= 2; w /= 2;
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.4.0", cur, tEmb, 320, 640, h, w);
        cur = SpatialTransformer("input_blocks.4.1", cur, context, 640, depth: 2, h, w);
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.5.0", cur, tEmb, 640, 640, h, w);
        cur = SpatialTransformer("input_blocks.5.1", cur, context, 640, depth: 2, h, w);
        savedInputs.Add(cur);

        // Block 6: Downsample (640 -> 640, stride 2)
        cur = Conv("input_blocks.6.0.op", cur, 640, h, w, 640, 3, stride: 2);
        h /= 2; w /= 2;
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.7.0", cur, tEmb, 640, 1280, h, w);
        cur = SpatialTransformer("input_blocks.7.1", cur, context, 1280, depth: 10, h, w);
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.8.0", cur, tEmb, 1280, 1280, h, w);
        cur = SpatialTransformer("input_blocks.8.1", cur, context, 1280, depth: 10, h, w);
        savedInputs.Add(cur);

        // ── Middle Block ────────────────────────────────────────────────────────
        cur = ResBlock("middle_block.0", cur, tEmb, 1280, 1280, h, w);
        cur = SpatialTransformer("middle_block.1", cur, context, 1280, depth: 10, h, w);
        cur = ResBlock("middle_block.2", cur, tEmb, 1280, 1280, h, w);

        // ── Output Blocks ───────────────────────────────────────────────────────
        static float[] CatSkip(float[] current, float[] skip, int curC, int skipC, int curH, int curW)
        {
            int hw = curH * curW;
            var cat = new float[(curC + skipC) * hw];
            Array.Copy(current, 0, cat, 0, curC * hw);
            Array.Copy(skip, 0, cat, curC * hw, skipC * hw);
            return cat;
        }

        // Block 0: ResBlock(1280 + 1280 -> 1280) + Attention(depth=10)
        cur = CatSkip(cur, savedInputs[8], 1280, 1280, h, w);
        cur = ResBlock("output_blocks.0.0", cur, tEmb, 2560, 1280, h, w);
        cur = SpatialTransformer("output_blocks.0.1", cur, context, 1280, depth: 10, h, w);

        // Block 1: ResBlock(1280 + 1280 -> 1280) + Attention(depth=10)
        cur = CatSkip(cur, savedInputs[7], 1280, 1280, h, w);
        cur = ResBlock("output_blocks.1.0", cur, tEmb, 2560, 1280, h, w);
        cur = SpatialTransformer("output_blocks.1.1", cur, context, 1280, depth: 10, h, w);

        // Block 2: ResBlock(1280 + 640 -> 1280) + Attention(depth=10) + Upsample(1280)
        cur = CatSkip(cur, savedInputs[6], 1280, 640, h, w);
        cur = ResBlock("output_blocks.2.0", cur, tEmb, 1920, 1280, h, w);
        cur = SpatialTransformer("output_blocks.2.1", cur, context, 1280, depth: 10, h, w);
        cur = DiffusionOps.Upsample2x(cur, 1, 1280, h, w);
        h *= 2; w *= 2;
        cur = Conv("output_blocks.2.2.conv", cur, 1280, h, w, 1280, 3);

        // Block 3: ResBlock(1280 + 640 -> 640) + Attention(depth=2)
        cur = CatSkip(cur, savedInputs[5], 1280, 640, h, w);
        cur = ResBlock("output_blocks.3.0", cur, tEmb, 1920, 640, h, w);
        cur = SpatialTransformer("output_blocks.3.1", cur, context, 640, depth: 2, h, w);

        // Block 4: ResBlock(640 + 640 -> 640) + Attention(depth=2)
        cur = CatSkip(cur, savedInputs[4], 640, 640, h, w);
        cur = ResBlock("output_blocks.4.0", cur, tEmb, 1280, 640, h, w);
        cur = SpatialTransformer("output_blocks.4.1", cur, context, 640, depth: 2, h, w);

        // Block 5: ResBlock(640 + 320 -> 640) + Attention(depth=2) + Upsample(640)
        cur = CatSkip(cur, savedInputs[3], 640, 320, h, w);
        cur = ResBlock("output_blocks.5.0", cur, tEmb, 960, 640, h, w);
        cur = SpatialTransformer("output_blocks.5.1", cur, context, 640, depth: 2, h, w);
        cur = DiffusionOps.Upsample2x(cur, 1, 640, h, w);
        h *= 2; w *= 2;
        cur = Conv("output_blocks.5.2.conv", cur, 640, h, w, 640, 3);

        // Block 6: ResBlock(640 + 320 -> 320)
        cur = CatSkip(cur, savedInputs[2], 640, 320, h, w);
        cur = ResBlock("output_blocks.6.0", cur, tEmb, 960, 320, h, w);

        // Block 7: ResBlock(320 + 320 -> 320)
        cur = CatSkip(cur, savedInputs[1], 320, 320, h, w);
        cur = ResBlock("output_blocks.7.0", cur, tEmb, 640, 320, h, w);

        // Block 8: ResBlock(320 + 320 -> 320)
        cur = CatSkip(cur, savedInputs[0], 320, 320, h, w);
        cur = ResBlock("output_blocks.8.0", cur, tEmb, 640, 320, h, w);

        // ── Final Output ────────────────────────────────────────────────────────
        var outGnW = GetWeight("out.0.weight");
        var outGnB = GetWeight("out.0.bias");
        DiffusionOps.GroupNorm(cur, outGnW, outGnB, 1, 320, h, w, groups: 32);
        DiffusionOps.SiluInPlace(cur);

        return Conv("out.2", cur, 320, h, w, 4, 3);
    }

    public void Dispose()
    {
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
        _weightReader.Clear();
    }
}
