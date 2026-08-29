
namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// CosyVoice2's CFM (Conditional Flow Matching) flow-decoder: `cosyvoice.flow.flow_matching.
/// CausalConditionalCFM` wrapped around the real `CausalConditionalDecoder` estimator (see
/// <see cref="CosyVoiceCfmDecoderWeights"/> for the full architectural derivation). Turns
/// [mu, spk-embedding] conditioning (from <see cref="CosyVoiceFlowEncoder"/>) plus gaussian
/// noise into a mel-spectrogram via a real 10-step Euler ODE solve over the real cosine
/// schedule `t_span[i] = 1 - cos(0.05*pi*i)` -- the SAME schedule already confirmed real and
/// golden-verified for CosyVoice3's DiT estimator (`CausalConditionalCFM::build_cgraph_one_step`
/// in `examples/cosyvoice.cpp`, and this checkpoint's own real yaml: `solver: euler,
/// t_scheduler: cosine`). Unlike Chatterbox's meanflow-distilled 2-step solver (which embeds
/// BOTH t and a target r and mixes them), this checkpoint's `time_mlp` has no mixer tensor at
/// all -- standard single-timestep flow matching, confirmed by the real checkpoint's tensor
/// list (only `time_mlp.linear_1/2`, no `time_embed_mixer`).
///
/// No `cond` conditioning tensor is used here (CosyVoice2's `CausalConditionalCFM.forward`
/// passes `cond=None` for non-streaming zero-shot synthesis -- there is no separate prompt-mel
/// conditioning channel beyond what's already folded into `mu`), so the 4th concat slot the
/// shared kernel expects is filled with zeros.
/// </summary>
public static class CosyVoiceCfmDecoder
{
    public static float[] Generate(CosyVoiceCfmDecoderWeights w, float[] mu, float[] spkEmbed, int t, Random rng, int nSteps = 10)
    {
        const int mel = CosyVoiceCfmDecoderWeights.OutChannels;

        var x = new float[mel * t];
        for (int i = 0; i < x.Length; i++) x[i] = SampleGaussian(rng);

        var cond = new float[mel * t]; // no separate cond tensor for this checkpoint's non-streaming path

        var tSpan = new float[nSteps + 1];
        for (int i = 0; i <= nSteps; i++) tSpan[i] = 1f - MathF.Cos(0.05f * MathF.PI * i);

        for (int step = 0; step < nSteps; step++)
        {
            float tCur = tSpan[step];
            var timeEmb = TimeEmbed(w, tCur);
            var dxdt = CfmUNetKernels.RunEstimator(
                w.DownStage, (IUnetStageWeights[])w.MidStages, w.UpStage,
                w.FinalBlockConvWeight, w.FinalBlockConvBias, w.FinalBlockLnWeight, w.FinalBlockLnBias,
                w.FinalProjWeight, w.FinalProjBias,
                x, mu, cond, spkEmbed, timeEmb,
                t, mel, CosyVoiceCfmDecoderWeights.Channels, CosyVoiceCfmDecoderWeights.NumHeads, CosyVoiceCfmDecoderWeights.HeadDim);
            float dt = tSpan[step + 1] - tCur;
            for (int i = 0; i < x.Length; i++) x[i] += dt * dxdt[i];
        }

        return x;
    }

    private static float SampleGaussian(Random rng)
    {
        double u1 = Math.Max(1e-12, rng.NextDouble());
        double u2 = rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    /// <summary>SinusoidalPosEmb(320) -> Linear(320,1024)+SiLU+Linear(1024,1024) (decoder.py's `self.time_mlp`, standard single-timestep flow matching -- no meanflow mixer in this checkpoint).</summary>
    private static float[] TimeEmbed(CosyVoiceCfmDecoderWeights w, float t)
    {
        var raw = CfmUNetKernels.SinusoidalPosEmb(t, CosyVoiceCfmDecoderWeights.InChannels);
        var h = Linear(raw, w.TimeMlpLinear1Weight, w.TimeMlpLinear1Bias, CosyVoiceCfmDecoderWeights.InChannels, CosyVoiceCfmDecoderWeights.Channels * 4);
        SiluInPlace(h);
        return Linear(h, w.TimeMlpLinear2Weight, w.TimeMlpLinear2Bias, CosyVoiceCfmDecoderWeights.Channels * 4, CosyVoiceCfmDecoderWeights.Channels * 4);
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
}
