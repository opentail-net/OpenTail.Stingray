
namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// F5-TTS's 22-layer Flow-Matching Diffusion Transformer (DiT), real weight-driven port of
/// `examples/f5-tts-py/f5_tts/model/backbones/dit.py`'s `DiT.forward`, verified against the real
/// PyTorch reference loaded with the real checkpoint (see scratch-llamacpp-ref/f5_golden_dit.py --
/// unlike the ONNX-exported VITS-family pipelines this session, F5-TTS ships as a safetensors
/// checkpoint with the original PyTorch source available, so golden verification runs the ACTUAL
/// reference model directly instead of an ONNX re-export).
///
/// x/cond are the noisy/conditioning mel spectrograms, channel-last [numFrames, MelDim]. text is
/// raw (unshifted) character ids. Returns the predicted velocity, same shape as x.
/// </summary>
public static class F5DiTModel
{
    public static float[] ForwardVelocity(F5TtsWeights w, float[] x, float[] cond, ReadOnlySpan<int> text, float timestep, int numFrames) =>
        ForwardVelocity(w, x, cond, text, timestep, numFrames, dropText: false);

    /// <summary>dropText selects CFG's null/unconditional text branch (see <see cref="F5TextEmbedding"/>'s dropText overload doc comment). Callers doing CFG should also pass an all-zero `cond` for the uncond branch (drop_audio_cond).</summary>
    public static float[] ForwardVelocity(F5TtsWeights w, float[] x, float[] cond, ReadOnlySpan<int> text, float timestep, int numFrames, bool dropText)
    {
        var textEmbed = F5TextEmbedding.Forward(w, text, numFrames, dropText);
        var (rotaryCos, rotarySin) = F5RotaryEmbedding.Precompute(w.RotaryInvFreq, numFrames);
        return ForwardVelocity(w, x, cond, textEmbed, timestep, numFrames, rotaryCos, rotarySin);
    }

    /// <summary>Optimized forward pass accepting precomputed/cached textEmbed, rotaryCos, and rotarySin to avoid redundant recomputations across ODE steps.
    /// <paramref name="backend"/>, when supplied, routes every Sgemm-shaped op through it (--backend
    /// vulkan, see docs/052-vulkan-backend-for-tts-engines-plan.md).</summary>
    public static float[] ForwardVelocity(
        F5TtsWeights w,
        float[] x,
        float[] cond,
        float[] textEmbed,
        float timestep,
        int numFrames,
        float[] rotaryCos,
        float[] rotarySin,
        Core.IComputeBackend? backend = null)
    {
        var h = F5InputEmbedding.Forward(w, x, cond, textEmbed, numFrames, backend);
        var tEmb = F5TimestepEmbedding.Forward(w, timestep, backend);

        for (int layer = 0; layer < F5TtsWeights.NumLayers; layer++)
            h = F5DiTBlock.Forward(w, w.Blocks[layer], h, tEmb, numFrames, rotaryCos, rotarySin, backend);

        // norm_out: AdaLayerNorm_Final -- emb = linear(silu(t)) -> chunk2(scale,shift); x = LN_noaffine(x)*(1+scale)+shift.
        int dim = F5TtsWeights.HiddenDim;
        var siluT = new float[dim];
        for (int d = 0; d < dim; d++) siluT[d] = F5Kernels.SiLU(tEmb[d]);
        var modulation = backend is not null
            ? F5Kernels.LinearGpuQ8_0(backend, siluT, 1, dim, w.NormOutLinearQ8, w.NormOutLinearBias, dim * 2)
            : F5Kernels.LinearQ8_0(siluT, 1, dim, w.NormOutLinearQ8, w.NormOutLinearBias, dim * 2);

        var normOut = F5Kernels.LayerNormNoAffine(h, numFrames, dim);
        F5Kernels.ApplyAffineModulationSlice(normOut, normOut, modulation, scaleOffset: 0, shiftOffset: dim, numFrames, dim);

        return backend is not null
            ? F5Kernels.LinearGpuQ8_0(backend, normOut, numFrames, dim, w.ProjOutQ8, w.ProjOutBias, F5TtsWeights.MelDim)
            : F5Kernels.LinearQ8_0(normOut, numFrames, dim, w.ProjOutQ8, w.ProjOutBias, F5TtsWeights.MelDim);
    }

    /// <summary>
    /// GPU-resident forward pass: real per-block VRAM-resident weights (<see cref="F5GpuWeights"/>)
    /// + preallocated workspace (<see cref="F5GpuWorkspace"/>) + a real <see cref="F5DiTBlock.
    /// ForwardGpu"/> per block, batched one <c>BeginBatch()</c>/<c>EndBatch()</c> per block
    /// (mirroring <c>WanModel.ForwardGpu</c>'s own batching granularity) -- eliminates the ~10
    /// synchronous round-trips per block the existing <see cref="ForwardVelocity"/>'s
    /// <see cref="F5Kernels.LinearGpuQ8_0"/> per-op path pays (weights were already GPU-resident
    /// there via its own cache, but every single op forced a `Synchronize()`+`Download()`).
    /// <paramref name="x"/>/<paramref name="cond"/> are packed once by the caller exactly like
    /// <see cref="ForwardVelocity"/>'s own `input_embed` step (kept CPU-side deliberately -- it
    /// runs once per ODE step, not per-block, so its relative cost is small).
    /// </summary>
    public static float[] ForwardVelocityGpu(
        F5TtsWeights w,
        F5GpuWeights gpuWeights,
        F5GpuWorkspace ws,
        float[] x,
        float[] cond,
        float[] textEmbed,
        float timestep,
        int numFrames,
        Core.IImageOpsBackend imageOps)
    {
        int dim = F5TtsWeights.HiddenDim;
        var h = F5InputEmbedding.Forward(w, x, cond, textEmbed, numFrames, null);
        var tEmb = F5TimestepEmbedding.Forward(w, timestep, null);
        var siluT = new float[dim];
        for (int d = 0; d < dim; d++) siluT[d] = F5Kernels.SiLU(tEmb[d]);

        using var xGpu = imageOps.Upload(h, Core.TensorShape.D2(numFrames, dim), exact: true);
        // ws.X is a persistent (Allocate'd, not zero-initialized) workspace buffer reused across
        // ODE steps -- clear then add, the same "ScaleInPlace(0)+AddInPlace" pattern UMT5Encoder.
        // EncodeGpu already established for writing a fresh value into a reused device buffer.
        imageOps.ScaleInPlace(ws.X, 0f);
        imageOps.AddInPlace(ws.X, xGpu);
        using var siluTGpu = imageOps.Upload(siluT, Core.TensorShape.D2(1, dim), exact: true);

        // Fully GPU-resident now (RoPE included, see F5DiTBlock.ForwardGpu's own 2026-09-14
        // update) -- one BeginBatch()/EndBatch() per block, NOT around the whole 22-block loop:
        // FLUX's own real UI-freeze bug (docs/069) came from exactly that "one giant command
        // buffer" mistake on this shared iGPU, so per-block stays the safe granularity here too.
        for (int layer = 0; layer < F5TtsWeights.NumLayers; layer++)
        {
            imageOps.BeginBatch();
            F5DiTBlock.ForwardGpu(gpuWeights.Blocks[layer], ws, numFrames, siluTGpu, imageOps);
            imageOps.EndBatch();
        }

        // norm_out: AdaLayerNorm_Final -- emb = linear(silu(t)) -> chunk2(scale,shift); x = LN_noaffine(x)*(1+scale)+shift.
        var visionOps = (Core.IVisionOpsBackend)imageOps;
        using var normOutModGpu = imageOps.Allocate(Core.TensorShape.D2(1, dim * 2));
        imageOps.Sgemm(normOutModGpu, siluTGpu, gpuWeights.NormOutLinearW, 1, dim, dim * 2);
        imageOps.AddRowBroadcastInPlace(normOutModGpu, gpuWeights.NormOutLinearB, 1, dim * 2);

        using var normOutGpu = imageOps.Allocate(Core.TensorShape.D2(numFrames, dim));
        visionOps.AdaLNModulate(normOutGpu, ws.X, normOutModGpu, numFrames, dim, shiftOffset: dim, scaleOffset: 0, isRmsNorm: false, eps: 1e-6f);

        using var outGpu = imageOps.Allocate(Core.TensorShape.D2(numFrames, F5TtsWeights.MelDim));
        imageOps.Sgemm(outGpu, normOutGpu, gpuWeights.ProjOutW, numFrames, dim, F5TtsWeights.MelDim);
        imageOps.AddRowBroadcastInPlace(outGpu, gpuWeights.ProjOutB, numFrames, F5TtsWeights.MelDim);

        var result = new float[numFrames * F5TtsWeights.MelDim];
        imageOps.Download(outGpu, result);
        return result;
    }

    /// <summary>
    /// Dual-stream CFG forward pass (batch=2: cond + uncond).
    /// Concatenates cond and uncond tokens and evaluates all 22 layers in a single pass, streaming
    /// weights from memory/VRAM once per block rather than twice.
    /// Returns (vCond, vUncond) each [numFrames, MelDim].
    /// </summary>
    public static (float[] VCond, float[] VUncond) ForwardVelocityBatch2(
        F5TtsWeights w,
        float[] x,
        float[] condMel,
        float[] nullCond,
        float[] textEmbedCond,
        float[] textEmbedUncond,
        float timestep,
        int numFrames,
        float[] rotaryCos,
        float[] rotarySin,
        Core.IComputeBackend? backend = null)
    {
        int dim = F5TtsWeights.HiddenDim;
        int halfLen = numFrames * dim;

        var hCond = F5InputEmbedding.Forward(w, x, condMel, textEmbedCond, numFrames, backend);
        var hUncond = F5InputEmbedding.Forward(w, x, nullCond, textEmbedUncond, numFrames, backend);

        var h2 = new float[2 * halfLen];
        hCond.CopyTo(h2.AsSpan(0, halfLen));
        hUncond.CopyTo(h2.AsSpan(halfLen, halfLen));

        var tEmb = F5TimestepEmbedding.Forward(w, timestep, backend);

        for (int layer = 0; layer < F5TtsWeights.NumLayers; layer++)
            h2 = F5DiTBlock.ForwardBatch2(w, w.Blocks[layer], h2, tEmb, numFrames, rotaryCos, rotarySin, backend);

        var siluT = new float[dim];
        for (int d = 0; d < dim; d++) siluT[d] = F5Kernels.SiLU(tEmb[d]);
        var modulation = backend is not null
            ? F5Kernels.LinearGpuQ8_0(backend, siluT, 1, dim, w.NormOutLinearQ8, w.NormOutLinearBias, dim * 2)
            : F5Kernels.LinearQ8_0(siluT, 1, dim, w.NormOutLinearQ8, w.NormOutLinearBias, dim * 2);

        int t2 = 2 * numFrames;
        var normOut = F5Kernels.LayerNormNoAffine(h2, t2, dim);
        F5Kernels.ApplyAffineModulationSlice(normOut, normOut, modulation, scaleOffset: 0, shiftOffset: dim, t2, dim);

        var v2 = backend is not null
            ? F5Kernels.LinearGpuQ8_0(backend, normOut, t2, dim, w.ProjOutQ8, w.ProjOutBias, F5TtsWeights.MelDim)
            : F5Kernels.LinearQ8_0(normOut, t2, dim, w.ProjOutQ8, w.ProjOutBias, F5TtsWeights.MelDim);

        int melLen = numFrames * F5TtsWeights.MelDim;
        var vCond = new float[melLen];
        var vUncond = new float[melLen];
        Array.Copy(v2, 0, vCond, 0, melLen);
        Array.Copy(v2, melLen, vUncond, 0, melLen);
        return (vCond, vUncond);
    }
}
