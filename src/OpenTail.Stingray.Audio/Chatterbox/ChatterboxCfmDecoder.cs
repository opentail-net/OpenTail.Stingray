using System;
using OpenTail.Stingray.Audio.Primitives;

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
    public static float[] Generate(ChatterboxS3GenWeights w, float[] mu, float[] cond, float[] spkEmbed, int t, Random rng, int nSteps = 2)
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
            var dxdtCond = CfmUNetKernels.RunEstimator(
                w.DownStage, (IUnetStageWeights[])w.MidStages, w.UpStage,
                w.FinalBlockConvWeight, w.FinalBlockConvBias, w.FinalBlockLnWeight, w.FinalBlockLnBias,
                w.FinalProjWeight, w.FinalProjBias,
                x, mu, cond, spkEmbed, timeEmb,
                t, mel, w.DecChannels, w.DecNumHeads, w.DecHeadDim);

            var dxdtUncond = CfmUNetKernels.RunEstimator(
                w.DownStage, (IUnetStageWeights[])w.MidStages, w.UpStage,
                w.FinalBlockConvWeight, w.FinalBlockConvBias, w.FinalBlockLnWeight, w.FinalBlockLnBias,
                w.FinalProjWeight, w.FinalProjBias,
                x, muZeros, condZeros, spkZeros, timeEmb,
                t, mel, w.DecChannels, w.DecNumHeads, w.DecHeadDim);

            float dt = tNext - tCur;
            for (int i = 0; i < x.Length; i++)
            {
                float dxdt = (1f + cfgRate) * dxdtCond[i] - cfgRate * dxdtUncond[i];
                x[i] += dt * dxdt;
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

    private static void SiluInPlace(float[] x)
    {
        for (int i = 0; i < x.Length; i++) x[i] = x[i] / (1f + MathF.Exp(-x[i]));
    }

    private static unsafe float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* w = weight, x = input, y = output)
        {
            Cpu.SimdKernels.MatVecF32(y, w, x, outDim, inDim);
        }
        for (int o = 0; o < outDim; o++) output[o] += bias[o];
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
