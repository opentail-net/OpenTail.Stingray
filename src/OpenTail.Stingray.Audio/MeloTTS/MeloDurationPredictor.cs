using System;
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.MeloTTS;

/// <summary>
/// MeloTTS's VITS2 duration prediction: `SynthesizerTrn.infer` blends TWO predictors --
/// `logw = sdp(x, g=g, reverse=True, noise_scale=noise_scale_w) * sdp_ratio +
///         dp(x, g=g) * (1 - sdp_ratio)` -- unlike Piper, which only has the SDP half
/// (single-speaker checkpoint, sdp_ratio implicitly 1.0). Ported from
/// `examples/melotts-py/models.py`'s `StochasticDurationPredictor.forward(reverse=True)` and
/// `DurationPredictor.forward`.
///
/// Both halves take the real speaker embedding `g` (`emb_g[sid]`, [gin_channels]) via a `cond`
/// 1x1 conv added to `x` right after `pre`/before `convs` -- Piper's single-speaker checkpoint
/// has no such conditioning (`gin_channels=0` there), so this is a real addition versus the
/// Piper port, not shared code (though the SDP flow math itself IS shared, via
/// <see cref="VitsDurationFlowKernels"/>).
/// </summary>
public static class MeloDurationPredictor
{
    /// <summary>
    /// encoderHidden is the TextEncoder's pre-proj output, channel-first [hidden, T]. g is the
    /// speaker embedding [gin_channels] (`emb_g[speakerId]`). sdpNoise must be a [2*T] array of
    /// raw N(0,1) draws (caller-supplied: real RNG in production, golden-captured for
    /// verification). Returns logw (pre-exp predicted log-duration), length T.
    /// </summary>
    public static float[] PredictLogDuration(
        MeloOnnxWeights w, float[] encoderHidden, int t, float[] g, float[] sdpNoise,
        float noiseScaleW, float sdpRatio)
    {
        float[] logwSdp = PredictSdp(w, encoderHidden, t, g, sdpNoise, noiseScaleW);
        float[] logwDp = PredictDp(w, encoderHidden, t, g);

        var logw = new float[t];
        for (int i = 0; i < t; i++)
            logw[i] = logwSdp[i] * sdpRatio + logwDp[i] * (1f - sdpRatio);
        return logw;
    }

    private static float[] PredictSdp(MeloOnnxWeights w, float[] encoderHidden, int t, float[] g, float[] noise, float noiseScaleW)
    {
        int dim = w.HiddenDim;

        var x = VitsAttentionKernels.Conv1x1(encoderHidden, dim, t, w.SdpPreWeight, w.SdpPreBias, dim);
        AddConditioning(x, t, dim, g, w.SdpCondWeight, w.SdpCondBias);
        x = VitsDurationFlowKernels.DDSConv(x, dim, t, w.SdpConvs);
        x = VitsAttentionKernels.Conv1x1(x, dim, t, w.SdpProjWeight, w.SdpProjBias, dim);

        var z = new float[2 * t];
        for (int i = 0; i < z.Length; i++) z[i] = noise[i] * noiseScaleW;

        // Same pruned execution order as Piper's dp (see PiperDurationPredictor's doc comment):
        // Flip(8) -> ConvFlow(7) -> Flip(6) -> ConvFlow(5) -> Flip(4) -> ConvFlow(3) -> Flip(2) -> ElementwiseAffine(0)
        z = VitsDurationFlowKernels.Flip(z, t);
        z = VitsDurationFlowKernels.ConvFlowReverse(z, t, x, dim, w.SdpFlow7);
        z = VitsDurationFlowKernels.Flip(z, t);
        z = VitsDurationFlowKernels.ConvFlowReverse(z, t, x, dim, w.SdpFlow5);
        z = VitsDurationFlowKernels.Flip(z, t);
        z = VitsDurationFlowKernels.ConvFlowReverse(z, t, x, dim, w.SdpFlow3);
        z = VitsDurationFlowKernels.Flip(z, t);
        z = VitsDurationFlowKernels.ElementwiseAffineReverse(z, t, w.SdpFlow0M, w.SdpFlow0ExpNegLogs);

        var logw = new float[t];
        Array.Copy(z, 0, logw, 0, t);
        return logw;
    }

    /// <summary>Plain DurationPredictor: x = encoderHidden + cond(g); conv_1(k=3)->ReLU->LN->conv_2(k=3)->ReLU->LN->proj(1x1, filter->1).</summary>
    private static float[] PredictDp(MeloOnnxWeights w, float[] encoderHidden, int t, float[] g)
    {
        int dim = w.HiddenDim;
        int filter = w.DpFilterChannels;

        var x = (float[])encoderHidden.Clone();
        AddConditioning(x, t, dim, g, w.DpCondWeight, w.DpCondBias);

        var h = VitsAttentionKernels.Conv1dSamePad(x, dim, t, w.DpConv1Weight, w.DpConv1Bias, filter, 3);
        for (int i = 0; i < h.Length; i++) if (h[i] < 0f) h[i] = 0f; // ReLU
        h = VitsAttentionKernels.LayerNormChannelFirst(h, filter, t, w.DpNorm1Gamma, w.DpNorm1Beta);

        h = VitsAttentionKernels.Conv1dSamePad(h, filter, t, w.DpConv2Weight, w.DpConv2Bias, filter, 3);
        for (int i = 0; i < h.Length; i++) if (h[i] < 0f) h[i] = 0f; // ReLU
        h = VitsAttentionKernels.LayerNormChannelFirst(h, filter, t, w.DpNorm2Gamma, w.DpNorm2Beta);

        var proj = VitsAttentionKernels.Conv1x1(h, filter, t, w.DpProjWeight, w.DpProjBias, 1);
        return proj; // [1, T] -> length T
    }

    /// <summary>x += cond(g) in place; cond is a 1x1 conv (gin_channels -> dim), g is a single [gin_channels] vector broadcast over every timestep.</summary>
    private static void AddConditioning(float[] x, int t, int dim, float[] g, float[] condWeight, float[] condBias)
    {
        int gin = g.Length;
        var condOut = new float[dim];
        for (int o = 0; o < dim; o++)
        {
            float sum = condBias[o];
            int wBase = o * gin;
            for (int i = 0; i < gin; i++) sum += condWeight[wBase + i] * g[i];
            condOut[o] = sum;
        }
        for (int ti = 0; ti < t; ti++)
            for (int c = 0; c < dim; c++)
                x[c * t + ti] += condOut[c];
    }
}
