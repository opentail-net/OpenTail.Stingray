
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// S3Gen stage 2: the CFM (Conditional Flow Matching) flow-matching UNet estimator
/// (examples/chatterbox-tts-py/chatterbox/models/s3gen/decoder.py's ConditionalDecoder) plus the
/// meanflow-distilled 2-step Euler ODE solver (flow_matching.py's CausalConditionalCFM.
/// basic_euler). Turns [mu, cond, spk-embedding] conditioning (from ChatterboxFlowEncoder) plus
/// gaussian noise into a mel-spectrogram.
///
/// The UNet body (down/mid/up ResnetBlock+TransformerBlock stages, final block+proj) is shared
/// with CosyVoice2's CFM decoder via <see cref="CfmUNetKernels"/> -- confirmed architecturally
/// identical by real tensor shapes, extracted once both pipelines had real weights to check
/// against each other (see docs/audio-review-progress.md's CosyVoice section). The meanflow
/// time-embedding (t AND r, concatenated and mixed) below is Chatterbox-SPECIFIC and does NOT
/// generalize to CosyVoice2 (whose checkpoint has no time-mixer tensor at all -- confirmed,
/// not assumed) -- kept local to this file rather than pushed into the shared kernel, which
/// only accepts an already-computed time-embedding vector.
///
/// Structure (channels=[256] in the reference, a single-element list, so both the "down" and
/// "up" stages degrade to no-op-resolution CausalConv1d resamples -- see decoder.py and
/// ChatterboxS3GenWeights's stage-loading comments):
///   down:  CausalResnetBlock1D(320-&gt;256) + 4x BasicTransformerBlock(256) + CausalConv1d(256-&gt;256, k=3)
///   mid:   12x [CausalResnetBlock1D(256-&gt;256) + 4x BasicTransformerBlock(256)]
///   up:    CausalResnetBlock1D(512-&gt;256, input = concat(x, down-stage skip)) + 4x BasicTransformerBlock(256) + CausalConv1d(256-&gt;256, k=3)
///   final: CausalBlock1D(256-&gt;256) + Conv1d(256-&gt;80, k=1)
///
/// Timestep conditioning (meanflow): t and r (start/end of this Euler step) are each embedded via
/// SinusoidalPosEmb(320) -> Linear(320,1024)+SiLU+Linear(1024,1024), concatenated to 2048-dim and
/// mixed down to 1024-dim via a single Linear (no bias) -- decoder.py's time_embed_mixer.
/// </summary>
public static class ChatterboxCfmDecoder
{
    /// <summary>
    /// mu, cond are channel-first [80, T]. spkEmbed is [80]. Returns the generated mel-spectrogram,
    /// channel-first [80, T] (still the FULL prompt+generated length -- caller slices off the
    /// prompt-length prefix, matching flow.py's `feat[:, :, mel_len1:]`).
    /// </summary>
    public static float[] Generate(ChatterboxS3GenWeights w, float[] mu, float[] cond, float[] spkEmbed, int t, Random rng, int nSteps = 2, Core.IComputeBackend? backend = null)
    {
        int mel = w.DecOutChannels; // 80

        var x = new float[mel * t];
        for (int i = 0; i < x.Length; i++) x[i] = SampleGaussian(rng);

        var tSpan = new float[nSteps + 1];
        for (int i = 0; i <= nSteps; i++) tSpan[i] = (float)i / nSteps;

        var muZeros = new float[mu.Length];
        var condZeros = new float[cond.Length];
        var spkZeros = new float[spkEmbed.Length];
        const float cfgRate = 0.7f;

        for (int step = 0; step < nSteps; step++)
        {
            float tCur = tSpan[step];
            float tNext = tSpan[step + 1];
            var timeEmb = MeanflowTimeEmbed(w, tCur, tNext);
            float[] dxdtCond = null!;
            float[] dxdtUncond = null!;

            // Both the CPU and GPU-backed paths run cond/uncond sequentially. IComputeBackend
            // implementations (e.g. VulkanBackend) are not documented/verified thread-safe for
            // concurrent dispatch from two threads onto the same instance, and the CPU path's
            // Parallel.Invoke concurrency was found to be a genuine data race too (2026-09-05:
            // running the two RunEstimator calls concurrently on the same shared weight objects
            // produced NaN and enormous garbage values that varied nondeterministically across
            // otherwise-identical, same-seed runs -- CfmUNetKernels.RunEstimator's internal
            // Parallel.For loops were verified index-disjoint on their own, so the race is in
            // running two full concurrent RunEstimator invocations against it, not in one; making
            // both branches sequential removes the race for a small, worthwhile perf cost).
            dxdtCond = CfmUNetKernels.RunEstimator(
                w.DownStage, (IUnetStageWeights[])w.MidStages, w.UpStage,
                w.FinalBlockConvWeight, w.FinalBlockConvBias, w.FinalBlockLnWeight, w.FinalBlockLnBias,
                w.FinalProjWeight, w.FinalProjBias,
                x, mu, cond, spkEmbed, timeEmb,
                t, mel, w.DecChannels, w.DecNumHeads, w.DecHeadDim, backend);
            dxdtUncond = CfmUNetKernels.RunEstimator(
                w.DownStage, (IUnetStageWeights[])w.MidStages, w.UpStage,
                w.FinalBlockConvWeight, w.FinalBlockConvBias, w.FinalBlockLnWeight, w.FinalBlockLnBias,
                w.FinalProjWeight, w.FinalProjBias,
                x, muZeros, condZeros, spkZeros, timeEmb,
                t, mel, w.DecChannels, w.DecNumHeads, w.DecHeadDim, backend);

            float dt = tNext - tCur;
            float factorCond = (1f + cfgRate) * dt;
            float factorUncond = -cfgRate * dt;

            int vLen = Vector<float>.Count;
            int limit = x.Length - (x.Length % vLen);
            var vFactorCond = new Vector<float>(factorCond);
            var vFactorUncond = new Vector<float>(factorUncond);

            for (int i = 0; i < limit; i += vLen)
            {
                var vx = new Vector<float>(x, i);
                var vc = new Vector<float>(dxdtCond, i);
                var vu = new Vector<float>(dxdtUncond, i);
                var vRes = vx + vc * vFactorCond + vu * vFactorUncond;
                vRes.CopyTo(x, i);
            }
            for (int i = limit; i < x.Length; i++)
            {
                x[i] += factorCond * dxdtCond[i] + factorUncond * dxdtUncond[i];
            }
        }

        return x;
    }

    private static float SampleGaussian(Random rng)
    {
        double u1 = Math.Max(1e-12, rng.NextDouble());
        double u2 = rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    /// <summary>
    /// SinusoidalPosEmb(320) -> Linear(320,1024)+SiLU+Linear(1024,1024) applied to both t and r,
    /// concatenated (2048) and mixed down to 1024 via a bias-free Linear (decoder.py's
    /// time_embed_mixer, only present because meanflow=True).
    /// </summary>
    private static float[] MeanflowTimeEmbed(ChatterboxS3GenWeights w, float tCur, float tNext)
    {
        var tEmbRaw = CfmUNetKernels.SinusoidalPosEmb(tCur, w.DecInChannels);
        var rEmbRaw = CfmUNetKernels.SinusoidalPosEmb(tNext, w.DecInChannels);
        int timeEmbedDim = w.DecChannels * 4; // 1024

        var tEmb = TimeMlp(w, tEmbRaw, timeEmbedDim);
        var rEmb = TimeMlp(w, rEmbRaw, timeEmbedDim);

        var concat = new float[timeEmbedDim * 2];
        Array.Copy(tEmb, 0, concat, 0, timeEmbedDim);
        Array.Copy(rEmb, 0, concat, timeEmbedDim, timeEmbedDim);

        return LinearNoBias(concat, w.TimeMixerWeight, timeEmbedDim * 2, timeEmbedDim);
    }

    private static float[] TimeMlp(ChatterboxS3GenWeights w, float[] emb, int timeEmbedDim)
    {
        var h = Linear(emb, w.TimeMlpLinear1Weight, w.TimeMlpLinear1Bias, w.DecInChannels, timeEmbedDim);
        SiluInPlace(h);
        return Linear(h, w.TimeMlpLinear2Weight, w.TimeMlpLinear2Bias, timeEmbedDim, timeEmbedDim);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization | System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static System.Runtime.Intrinsics.Vector256<float> VectorExp(System.Runtime.Intrinsics.Vector256<float> x)
    {
        var vx = System.Runtime.Intrinsics.X86.Avx.Min(System.Runtime.Intrinsics.X86.Avx.Max(x, System.Runtime.Intrinsics.Vector256.Create(-88.0f)), System.Runtime.Intrinsics.Vector256.Create(88.0f));
        var log2e = System.Runtime.Intrinsics.Vector256.Create(1.4426950408889634f);
        var ln2 = System.Runtime.Intrinsics.Vector256.Create(0.6931471805599453f);
        var half = System.Runtime.Intrinsics.Vector256.Create(0.5f);

        var z = System.Runtime.Intrinsics.X86.Avx.Multiply(vx, log2e);
        var k = System.Runtime.Intrinsics.X86.Avx.Floor(System.Runtime.Intrinsics.X86.Avx.Add(z, half));
        var f = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(k, System.Runtime.Intrinsics.X86.Avx.Subtract(System.Runtime.Intrinsics.Vector256<float>.Zero, ln2), vx);

        var c4 = System.Runtime.Intrinsics.Vector256.Create(0.009618129f);
        var c3 = System.Runtime.Intrinsics.Vector256.Create(0.055504108f);
        var c2 = System.Runtime.Intrinsics.Vector256.Create(0.240226507f);
        var c1 = System.Runtime.Intrinsics.Vector256.Create(0.693147180f);
        var one = System.Runtime.Intrinsics.Vector256.Create(1.0f);

        var p = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(c4, f, c3);
        p = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(p, f, c2);
        p = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(p, f, c1);
        p = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(p, f, one);

        var ki = System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int32(k);
        var expScale = System.Runtime.Intrinsics.X86.Avx2.ShiftLeftLogical(System.Runtime.Intrinsics.X86.Avx2.Add(ki, System.Runtime.Intrinsics.Vector256.Create(127)), 23).AsSingle();

        return System.Runtime.Intrinsics.X86.Avx.Multiply(p, expScale);
    }

    private static unsafe void SiluInPlace(float[] x)
    {
        int len = x.Length;
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported)
        {
            var vone = System.Runtime.Intrinsics.Vector256.Create(1.0f);
            var vzero = System.Runtime.Intrinsics.Vector256<float>.Zero;
            fixed (float* px = x)
            {
                int vecLen = len & ~7;
                for (int i = 0; i < vecLen; i += 8)
                {
                    var vx = System.Runtime.Intrinsics.X86.Avx.LoadVector256(px + i);
                    var vnegX = System.Runtime.Intrinsics.X86.Avx.Subtract(vzero, vx);
                    var vExp = VectorExp(vnegX);
                    var vDenom = System.Runtime.Intrinsics.X86.Avx.Add(vone, vExp);
                    var vRes = System.Runtime.Intrinsics.X86.Avx.Divide(vx, vDenom);
                    System.Runtime.Intrinsics.X86.Avx.Store(px + i, vRes);
                }
                for (int i = vecLen; i < len; i++)
                {
                    px[i] = px[i] / (1f + MathF.Exp(-px[i]));
                }
            }
        }
        else
        {
            for (int i = 0; i < len; i++) x[i] = x[i] / (1f + MathF.Exp(-x[i]));
        }
    }

    private static unsafe float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* w = weight, b = bias, x = input, y = output)
        {
            Cpu.SimdKernels.MatVecF32(y, w, b, x, outDim, inDim);
        }
        return output;
    }

    private static unsafe float[] LinearNoBias(float[] input, float[] weight, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* w = weight, x = input, y = output)
        {
            Cpu.SimdKernels.MatVecF32(y, w, x, outDim, inDim);
        }
        return output;
    }
}
