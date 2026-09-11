using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Universal VAE decoder: decodes latent tensors to RGB image [1, 3, H*8, W*8].
/// Supports:
///   - 16-channel latents (FLUX.1, Z-Image-Turbo)
///   - 4-channel latents (Stable Diffusion 1.5, SDXL)
///
/// Handles standalone VAE checkpoints (ae.safetensors) and unified model weights
/// with both CompVis (first_stage_model.decoder.mid.block_1...) and Diffusers (decoder.mid_block.resnets.0...) schemas.
/// </summary>
public sealed class VaeDecoder : IDisposable, IVaeDecoder
{
    private readonly IWeightLoader _st;
    private readonly IComputeBackend? _backend;

    private readonly Dictionary<string, float[]> _cpuWeights = new(StringComparer.Ordinal);
    private readonly Dictionary<string, CoreTensor>? _gpuWeights;

    // Separate cache for the native Conv2d-shader path (see ConvNative below). Its weight buffer
    // must be plain Float32 (the shader reads `float weight_data[]` directly, no dtype dispatch),
    // unlike _gpuWeights above which stores Half-converted tensors for the Sgemm mixed-precision
    // path -- reusing that cache here would misinterpret 2-byte Half values as 4-byte floats.
    private readonly Dictionary<string, CoreTensor>? _gpuWeightsNative;
    private readonly IImageOpsBackend? _imageOps;

    private const float VaeShift = 0.1159f;
    // Perf note (2026-09-11): tried bumping this 4x (32M -> 128M) to cut GPU dispatch count on the
    // theory that fewer, larger chunks would help -- measured real weights/timing and it was a
    // clear REGRESSION (VAE decode 31.01s -> 34.59s, total run 77.0s -> 93.5s), not an improvement.
    // Larger single transfers apparently cost more than they save in reduced dispatch overhead on
    // this iGPU (no overlap between a large Upload/Download and compute). Reverted -- keeping this
    // note so the same idea isn't retried without re-measuring.
    private const int MaxColChunkFloats = 32 * 1024 * 1024;

    public VaeDecoder(string path) => _st = SafetensorsLoader.Open(path);

    public VaeDecoder(IWeightLoader st) => _st = st;

    public VaeDecoder(IWeightLoader st, IComputeBackend? backend = null)
    {
        _st      = st;
        _backend = backend;
        if (backend is not null)
        {
            _gpuWeights = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
            _gpuWeightsNative = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
            _imageOps = backend as IImageOpsBackend;
        }
    }

    private string Resolve(string name)
    {
        if (_st.Contains(name)) return name;
        if (_st.Contains("first_stage_model." + name)) return "first_stage_model." + name;
        if (_st.Contains("vae." + name)) return "vae." + name;
        return name;
    }

    /// <summary>
    /// Decode latent [C, H, W] → RGB float [3, H*8, W*8], values in [0,1].
    ///
    /// <para><b>Real per-checkpoint scale/shift</b>: FLUX.1/Z-Image-Turbo and SD3/3.5 both use a
    /// 16-CHANNEL latent space, but they are DIFFERENT VAE checkpoints with different real
    /// `scaling_factor`/`shift_factor` config values -- channel count alone cannot distinguish
    /// them. <paramref name="scaleOverride"/>/<paramref name="shiftOverride"/> let a caller that
    /// knows which checkpoint it has pass the real values explicitly; omitted, this falls back to
    /// the channel-count heuristic (correct for SD1.5's 4-channel VAE and FLUX/Z-Image's 16-channel
    /// VAE, WRONG for SD3/3.5's 16-channel VAE -- <see cref="Sd3.Sd3Pipeline"/> always passes its
    /// own real values, `1/1.5305` and `0.0609`, confirmed from the real `stabilityai/
    /// stable-diffusion-3.5-medium` `vae/config.json`).</para>
    /// </summary>
    public float[] Decode(float[] latent, int latH, int latW) => Decode(latent, latH, latW, null, null);

    /// <summary>Overload accepting an explicit real scale/shift for a checkpoint the channel-count
    /// heuristic would get wrong (see the class-level remarks above) -- <see cref="Sd3.Sd3Pipeline"/>
    /// always calls this form.</summary>
    public float[] Decode(float[] latent, int latH, int latW, float? scaleOverride, float? shiftOverride)
    {
        int latentCh = latent.Length / (latH * latW);
        float scale = scaleOverride ?? (latentCh == 4 ? (1f / 0.18215f) : (1f / 0.3611f));
        float shift = shiftOverride ?? (latentCh == 4 ? 0f : VaeShift);

        // Perf: vectorized (SIMD) scale+shift instead of a scalar loop.
        var z = new float[latent.Length];
        TensorPrimitives.Multiply(latent, scale, z);
        TensorPrimitives.Add(z, shift, z);

        // post_quant_conv: Conv2D(C→C, 1×1)
        // conv_in: Conv2D(C→512, 3×3)
        // Perf (2026-09-11, first step of the "port VAE to the native GPU Conv2d shader" plan):
        // these two standalone convs go through ConvNative (one direct GPU dispatch, zero CPU-side
        // im2col/transpose) instead of ConvBlock's im2col+Sgemm path when the backend supports it.
        // Not yet extended to ResBlock/mid-block/up-block convs -- proving the integration on the
        // simplest case first, per this session's incremental-verification discipline.
        string pqKey = Resolve("post_quant_conv");
        if (_st.Contains($"{pqKey}.weight"))
        {
            z = _imageOps is not null
                ? ConvNative(_imageOps, pqKey, z, latentCh, latH, latW, latentCh, 1, padding: 0)
                : ConvBlock(pqKey, z, 1, latentCh, latH, latW, latentCh, 1, padding: 0);
        }

        string convInKey = Resolve("decoder.conv_in");
        z = _imageOps is not null
            ? ConvNative(_imageOps, convInKey, z, latentCh, latH, latW, 512, 3)
            : ConvBlock(convInKey, z, 1, latentCh, latH, latW, 512, 3);
        int ch = 512, h = latH, w = latW;

        bool isCompVis = _st.Contains(Resolve("decoder.mid.block_1.conv1.weight"));

        // TEMPORARY diagnostic instrumentation (2026-09-11), gated behind STINGRAY_PROFILE_VAE=1,
        // added while chasing VAE decode's real internal cost breakdown (found to be the single
        // largest stage in SDXL-Turbo's pipeline -- see PerformanceLeague.md). Remove once the
        // real bottleneck substage is found and fixed, matching the same disposable-diagnostic
        // convention as HybridGdnForwardPass's ProfCat instrumentation.
        bool profVae = Environment.GetEnvironmentVariable("STINGRAY_PROFILE_VAE") == "1";
        var vaeSw = profVae ? System.Diagnostics.Stopwatch.StartNew() : null;
        void LogVae(string name)
        {
            if (vaeSw is null) return;
            Console.Error.WriteLine($"[VAE] {name}: {vaeSw.Elapsed.TotalSeconds:F2}s");
            vaeSw.Restart();
        }

        if (isCompVis)
        {
            // CompVis SD1.5 schema
            string dec = Resolve("decoder");
            z = ResBlock($"{dec}.mid.block_1", z, 1, ch, h, w);
            z = MidAttnCompVis($"{dec}.mid.attn_1", z, 1, ch, h, w);
            z = ResBlock($"{dec}.mid.block_2", z, 1, ch, h, w);
            LogVae("mid_block (64x64)");

            // Up blocks: up.3 (512, upsample), up.2 (512, upsample), up.1 (256, upsample), up.0 (128, no upsample)
            (z, ch, h, w) = UpBlockCompVis(z, 1, ch, h, w, $"{dec}.up.3", outCh: 512, upsample: true);
            LogVae("up.3 (64->128, 512ch)");
            (z, ch, h, w) = UpBlockCompVis(z, 1, ch, h, w, $"{dec}.up.2", outCh: 512, upsample: true);
            LogVae("up.2 (128->256, 512ch)");
            (z, ch, h, w) = UpBlockCompVis(z, 1, ch, h, w, $"{dec}.up.1", outCh: 256, upsample: true);
            LogVae("up.1 (256->512, 256ch)");
            (z, ch, h, w) = UpBlockCompVis(z, 1, ch, h, w, $"{dec}.up.0", outCh: 128, upsample: false);
            LogVae("up.0 (512, 128ch, no upsample)");
        }
        else
        {
            // Diffusers / FLUX schema
            string dec = Resolve("decoder");
            z = ResBlock($"{dec}.mid_block.resnets.0", z, 1, ch, h, w);
            z = MidAttnDiffusers($"{dec}.mid_block.attentions.0", z, 1, ch, h, w);
            z = ResBlock($"{dec}.mid_block.resnets.1", z, 1, ch, h, w);
            LogVae("mid_block (64x64)");

            (z, ch, h, w) = UpBlockDiffusers(z, 1, ch, h, w, $"{dec}.up_blocks.0", outCh: 512, upsample: true);
            LogVae("up_blocks.0 (64->128, 512ch)");
            (z, ch, h, w) = UpBlockDiffusers(z, 1, ch, h, w, $"{dec}.up_blocks.1", outCh: 512, upsample: true);
            LogVae("up_blocks.1 (128->256, 512ch)");
            (z, ch, h, w) = UpBlockDiffusers(z, 1, ch, h, w, $"{dec}.up_blocks.2", outCh: 256, upsample: true);
            LogVae("up_blocks.2 (256->512, 256ch)");
            (z, ch, h, w) = UpBlockDiffusers(z, 1, ch, h, w, $"{dec}.up_blocks.3", outCh: 128, upsample: false);
            LogVae("up_blocks.3 (512, 128ch, no upsample)");
        }

        // norm_out
        string normName = _st.Contains(Resolve("decoder.conv_norm_out.weight")) ? Resolve("decoder.conv_norm_out")
                                                                                : Resolve("decoder.norm_out");
        var gnW = Wt($"{normName}.weight");
        var gnB = Wt($"{normName}.bias");
        DiffusionOps.GroupNorm(z, gnW, gnB, 1, ch, h, w, groups: 32);
        DiffusionOps.SiluInPlace(z);

        // conv_out: Conv2D(128→3)
        string convOutKey = Resolve("decoder.conv_out");
        z = _imageOps is not null
            ? ConvNative(_imageOps, convOutKey, z, ch, h, w, 3, 3)
            : ConvBlock(convOutKey, z, 1, ch, h, w, 3, 3);
        LogVae("norm_out + conv_out");

        // Clamp to [0, 1] (vectorized -- this runs over the full-resolution RGB output)
        TensorPrimitives.Add(z, 1f, z);
        TensorPrimitives.Multiply(z, 0.5f, z);
        TensorPrimitives.Clamp(z, 0f, 1f, z);

        return z;
    }

    // ── Building blocks ───────────────────────────────────────────────────

    private (float[] z, int ch, int h, int w) UpBlockDiffusers(
        float[] z, int n, int inCh, int h, int w,
        string prefix, int outCh, bool upsample)
    {
        for (int r = 0; r < 3; r++)
        {
            string resPrefix = $"{prefix}.resnets.{r}";
            z = ResBlock(resPrefix, z, n, r == 0 ? inCh : outCh, h, w, outCh);
        }
        int ch = outCh;

        if (upsample)
        {
            z = DiffusionOps.Upsample2x(z, n, ch, h, w);
            h *= 2; w *= 2;
            string convKey = $"{prefix}.upsamplers.0.conv";
            z = ConvBlock(convKey, z, n, ch, h, w, ch, 3);
        }
        return (z, ch, h, w);
    }

    private (float[] z, int ch, int h, int w) UpBlockCompVis(
        float[] z, int n, int inCh, int h, int w,
        string prefix, int outCh, bool upsample)
    {
        for (int r = 0; r < 3; r++)
        {
            string resPrefix = $"{prefix}.block.{r}";
            z = ResBlock(resPrefix, z, n, r == 0 ? inCh : outCh, h, w, outCh);
        }
        int ch = outCh;

        if (upsample)
        {
            z = DiffusionOps.Upsample2x(z, n, ch, h, w);
            h *= 2; w *= 2;
            string convKey = $"{prefix}.upsample.conv";
            z = ConvBlock(convKey, z, n, ch, h, w, ch, 3);
        }
        return (z, ch, h, w);
    }

    private float[] ResBlock(string prefix, float[] x, int n, int inCh, int h, int w, int outCh = -1)
    {
        if (outCh < 0) outCh = inCh;

        // norm1 + silu + conv1
        var gnW1 = Wt($"{prefix}.norm1.weight");
        var gnB1 = Wt($"{prefix}.norm1.bias");
        var h1 = (float[])x.Clone();
        DiffusionOps.GroupNorm(h1, gnW1, gnB1, n, inCh, h, w, groups: 32);
        DiffusionOps.SiluInPlace(h1);
        h1 = ConvBlock($"{prefix}.conv1", h1, n, inCh, h, w, outCh, 3);

        // norm2 + silu + conv2
        var gnW2 = Wt($"{prefix}.norm2.weight");
        var gnB2 = Wt($"{prefix}.norm2.bias");
        DiffusionOps.GroupNorm(h1, gnW2, gnB2, n, outCh, h, w, groups: 32);
        DiffusionOps.SiluInPlace(h1);
        h1 = ConvBlock($"{prefix}.conv2", h1, n, outCh, h, w, outCh, 3);

        // Skip connection: project input if channels differ
        float[] skip = x;
        if (inCh != outCh)
        {
            string shortcutKey = _st.Contains($"{prefix}.nin_shortcut.weight") ? $"{prefix}.nin_shortcut" : $"{prefix}.conv_shortcut";
            skip = ConvBlock(shortcutKey, x, n, inCh, h, w, outCh, 1, padding: 0);
        }

        TensorPrimitives.Add(h1.AsSpan(), skip.AsSpan(), h1.AsSpan());
        return h1;
    }

    private float[] MidAttnCompVis(string prefix, float[] x, int n, int ch, int h, int w)
    {
        var normed = (float[])x.Clone();
        var gnW = Wt($"{prefix}.norm.weight");
        var gnB = Wt($"{prefix}.norm.bias");
        DiffusionOps.GroupNorm(normed, gnW, gnB, n, ch, h, w, groups: 32);

        int hw = h * w;

        var wQ   = Wt($"{prefix}.q.weight");
        var wK   = Wt($"{prefix}.k.weight");
        var wV   = Wt($"{prefix}.v.weight");
        var wOut = Wt($"{prefix}.proj_out.weight");
        var bOut = _st.Contains($"{prefix}.proj_out.bias") ? Wt($"{prefix}.proj_out.bias") : null;

        return ComputeSpatialSelfAttn(x, normed, wQ, wK, wV, wOut, bOut, n, ch, h, w);
    }

    private float[] MidAttnDiffusers(string prefix, float[] x, int n, int ch, int h, int w)
    {
        var normed = (float[])x.Clone();
        var gnW = Wt($"{prefix}.group_norm.weight");
        var gnB = Wt($"{prefix}.group_norm.bias");
        DiffusionOps.GroupNorm(normed, gnW, gnB, n, ch, h, w, groups: 32);

        var wQ   = Wt($"{prefix}.to_q.weight");
        var wK   = Wt($"{prefix}.to_k.weight");
        var wV   = Wt($"{prefix}.to_v.weight");
        var wOut = Wt($"{prefix}.to_out.0.weight");
        var bOut = _st.Contains($"{prefix}.to_out.0.bias") ? Wt($"{prefix}.to_out.0.bias") : null;

        return ComputeSpatialSelfAttn(x, normed, wQ, wK, wV, wOut, bOut, n, ch, h, w);
    }

    private float[] ComputeSpatialSelfAttn(float[] x, float[] normed, float[] wQ, float[] wK, float[] wV, float[] wOut, float[]? bOut, int n, int ch, int h, int w)
    {
        int hw = h * w;
        float scale = 1f / MathF.Sqrt(ch);
        var result = (float[])x.Clone();

        for (int b = 0; b < n; b++)
        {
            var tokens = new float[hw * ch];
            for (int pos = 0; pos < hw; pos++)
            for (int c2 = 0; c2 < ch; c2++)
                tokens[pos * ch + c2] = normed[b * ch * hw + c2 * hw + pos];

            var q = LinearHW(tokens, wQ, hw, ch);
            var k = LinearHW(tokens, wK, hw, ch);
            var v = LinearHW(tokens, wV, hw, ch);

            var attnOut = new float[hw * ch];
            Parallel.For(0, hw, i =>
            {
                var qi = q.AsSpan(i * ch, ch);
                var localScores = new float[hw];
                for (int j = 0; j < hw; j++)
                    localScores[j] = TensorPrimitives.Dot(qi, k.AsSpan(j * ch, ch)) * scale;

                float maxS = TensorPrimitives.Max(localScores.AsSpan());
                TensorPrimitives.Subtract(localScores.AsSpan(), maxS, localScores.AsSpan());
                TensorPrimitives.Exp(localScores.AsSpan(), localScores.AsSpan());
                TensorPrimitives.Divide(localScores.AsSpan(),
                    TensorPrimitives.Sum(localScores.AsSpan()), localScores.AsSpan());

                var outRow = attnOut.AsSpan(i * ch, ch);
                outRow.Clear();
                for (int j = 0; j < hw; j++)
                    TensorPrimitives.MultiplyAdd<float>(v.AsSpan(j * ch, ch), localScores[j], outRow, outRow);
            });

            var proj = LinearHW(attnOut, wOut, hw, ch);
            for (int pos = 0; pos < hw; pos++)
            for (int c2 = 0; c2 < ch; c2++)
            {
                int nchwOff = b * ch * hw + c2 * hw + pos;
                result[nchwOff] += proj[pos * ch + c2] + (bOut is not null ? bOut[c2] : 0f);
            }
        }

        return result;
    }

    private static float[] LinearHW(float[] tokens, float[] w, int hw, int ch)
    {
        var result = new float[hw * ch];
        Parallel.For(0, hw, pos =>
        {
            var rowIn  = tokens.AsSpan(pos * ch, ch);
            var rowOut = result.AsSpan(pos * ch, ch);
            for (int oc = 0; oc < ch; oc++)
                rowOut[oc] = TensorPrimitives.Dot(rowIn, w.AsSpan(oc * ch, ch));
        });
        return result;
    }

    private float[] ConvBlock(string name, float[] x, int n, int inCh, int h, int w, int outCh, int k, int padding = -1)
    {
        if (_backend is not null)
            return ConvGpu(name, x, inCh, h, w, outCh, k, padding);
        var weight = Wt($"{name}.weight");
        var bias   = _st.Contains($"{name}.bias") ? Wt($"{name}.bias") : null;
        return DiffusionOps.Conv2D(x, weight, bias, n, inCh, h, w, outCh, k, k, stride: 1, padding: padding);
    }

    private float[] ConvGpu(string name, float[] x, int inCh, int h, int w, int outCh, int ksize, int padding)
    {
        if (padding < 0) padding = (ksize - 1) / 2;
        int outH = h, outW = w;
        int hw   = outH * outW;
        int kPts = inCh * ksize * ksize;

        string wKey = $"{name}.weight";
        if (!_gpuWeights!.TryGetValue(wKey, out var wGpu))
        {
            var wf = Wt(wKey);
            // Perf: see the identical comment in SdxlUNet2DConditionModel.GetGpuWeight -- upload
            // weights as Half when the backend prefers fp16 Sgemm, instead of always forcing the
            // slowest full-fp32 path.
            if (_backend!.BestSgemmPrecision == SgemmPrecision.Fp16)
            {
                var half = new Half[wf.Length];
                TensorPrimitives.ConvertToHalf(wf, half);
                wGpu = _backend.UploadHalf(half, TensorShape.D1(wf.Length));
            }
            else
            {
                wGpu = _backend.Upload(wf.AsSpan(), TensorShape.D1(wf.Length));
            }
            _gpuWeights[wKey] = wGpu;
        }

        var biasArr = _st.Contains($"{name}.bias") ? Wt($"{name}.bias") : null;

        int chunkRows = kPts > 0 ? Math.Max(1, Math.Min(outH, MaxColChunkFloats / (outW * kPts))) : outH;
        var output = new float[outCh * hw];

        // Perf note (2026-09-11): tried double-buffering colBuf + a background Task to build the
        // NEXT chunk's im2col while the CURRENT chunk's Sgemm/Download GPU fence-wait was in
        // flight (a distinct idea from the chunk-SIZE change also tried/reverted above -- this one
        // targeted dispatch-count/latency overlap, not raw transfer size). Measured real
        // weights/timing across two runs: no real improvement (31.01s baseline vs 32.23s/32.90s
        // piped, i.e. flat-to-slightly-worse, not the hoped-for overlap win). Most likely cause:
        // Im2ColChunk's own Parallel.For already saturates the thread pool, so a Task.Run'd "next
        // chunk" competes with it for the same worker threads instead of cleanly overlapping with
        // otherwise-idle time. Reverted -- keeping this note so the same idea isn't retried without
        // re-measuring (e.g. a dedicated single background thread instead of Task.Run might behave
        // differently, but that's unverified, not assumed to work).
        var colBuf    = ArrayPool<float>.Shared.Rent(chunkRows * outW * kPts);
        var resultBuf = ArrayPool<float>.Shared.Rent(chunkRows * outW * outCh);
        try
        {
            for (int rowStart = 0; rowStart < outH; rowStart += chunkRows)
            {
                int rowEnd  = Math.Min(rowStart + chunkRows, outH);
                int chunkH  = rowEnd - rowStart;
                int chunkHW = chunkH * outW;
                int colSize = chunkHW * kPts;

                Im2ColChunk(x, inCh, h, w, ksize, padding, rowStart, rowEnd, colBuf);

                var colGpu = _backend!.Upload(colBuf.AsSpan(0, colSize), TensorShape.D1(colSize));
                var cGpu   = _backend.Allocate(TensorShape.D1(chunkHW * outCh));
                try
                {
                    // Perf: Sgemm's Dispatch() already fence-waits for this specific dispatch
                    // before returning, and Download's own CopyBuffer does its own submit-and-wait
                    // for the transfer -- this explicit Synchronize() (a full vkDeviceWaitIdle())
                    // was pure redundant overhead (see the same finding in SdxlUNet2DConditionModel).
                    _backend.Sgemm(cGpu, colGpu, wGpu, chunkHW, kPts, outCh);
                    _backend.Download(cGpu, resultBuf.AsSpan(0, chunkHW * outCh));
                }
                finally
                {
                    _backend.Free(colGpu);
                    _backend.Free(cGpu);
                }

                // Perf: same class of fix as Im2ColChunk -- this transpose+bias-add write-back was
                // single-threaded scalar code at the same scale (chunkHW*outCh writes). Each `pos`
                // writes to disjoint locations across all `oc` (output[oc*hw + absPos]), so rows
                // parallelize cleanly.
                int basePos = rowStart * outW;
                Parallel.For(0, chunkHW, pos =>
                {
                    int absPos = basePos + pos;
                    for (int oc = 0; oc < outCh; oc++)
                        output[oc * hw + absPos] = resultBuf[pos * outCh + oc] + (biasArr?[oc] ?? 0f);
                });
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(colBuf);
            ArrayPool<float>.Shared.Return(resultBuf);
        }
        return output;
    }

    private static void Im2ColChunk(float[] x, int inCh, int h, int w,
                                    int ksize, int padding,
                                    int rowStart, int rowEnd, float[] col)
    {
        int outW = w;
        int kPts = inCh * ksize * ksize;
        // Perf: this gather was single-threaded scalar code -- at VAE decode's largest resolution
        // stages (e.g. 512x512, 128+ channels) it does hundreds of millions of boundary-checked
        // gathers per conv layer while the GPU sits idle waiting for it. Each output row's gather
        // writes a disjoint, directly-computable range of `col` (row r occupies
        // [r*outW*kPts, (r+1)*outW*kPts)), so rows parallelize cleanly -- no shared mutable index.
        Parallel.For(rowStart, rowEnd, oh =>
        {
            int rowBase = (oh - rowStart) * outW * kPts;
            for (int ow = 0; ow < outW; ow++)
            {
                int idx = rowBase + ow * kPts;
                for (int ic = 0; ic < inCh; ic++)
                for (int kh = 0; kh < ksize; kh++)
                for (int kw = 0; kw < ksize; kw++)
                {
                    int ih = oh + kh - padding;
                    int iw = ow + kw - padding;
                    col[idx++] = ((uint)ih < (uint)h && (uint)iw < (uint)w)
                        ? x[ic * h * w + ih * w + iw]
                        : 0f;
                }
            }
        });
    }

    private float[] Wt(string name)
    {
        if (_cpuWeights.TryGetValue(name, out var c)) return c;
        var w = _st.ReadF32(name);
        _cpuWeights[name] = w;
        return w;
    }

    /// <summary>
    /// Conv2D via the native GPU Conv2d compute shader (see Shaders.Conv2d) instead of the
    /// im2col+Sgemm path above -- one direct dispatch, zero CPU-side gather/transpose. Only used
    /// when the backend implements <see cref="IImageOpsBackend"/> (Vulkan; CUDA/CPU fall back to
    /// <see cref="ConvBlock"/>). Requires stride=1, which every VAE conv already uses.
    /// </summary>
    private float[] ConvNative(IImageOpsBackend imageOps, string name, float[] x, int inCh, int h, int w, int outCh, int k, int padding = -1)
    {
        string wKey = $"{name}.weight";
        if (!_gpuWeightsNative!.TryGetValue(wKey, out var wGpu))
        {
            var wf = Wt(wKey);
            wGpu = imageOps.Upload(wf.AsSpan(), TensorShape.D1(wf.Length));
            _gpuWeightsNative[wKey] = wGpu;
        }

        string bKey = $"{name}.bias";
        if (!_gpuWeightsNative.TryGetValue(bKey, out var bGpu))
        {
            // The native shader always reads a bias buffer (acc = bias_data[oc], no null-bias
            // branch) -- upload zeros when this conv has none, matching the im2col path's
            // `biasArr?[oc] ?? 0f` fallback.
            var bf = _st.Contains(bKey) ? Wt(bKey) : new float[outCh];
            bGpu = imageOps.Upload(bf.AsSpan(), TensorShape.D1(bf.Length));
            _gpuWeightsNative[bKey] = bGpu;
        }

        var xGpu = imageOps.Upload(x.AsSpan(0, inCh * h * w), TensorShape.D1(inCh * h * w));
        var yGpu = imageOps.Conv2d(xGpu, wGpu, bGpu, inCh, outCh, h, w, k, padding);
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
        _cpuWeights.Clear();
        _st.Dispose();
    }
}
