using System;
using System.Threading.Tasks;

namespace OpenTail.Stingray.Audio.FishSpeech;

/// <summary>
/// Real Fish Speech S2 Pro codec DECODE-only forward pass, transcribed directly from the real
/// `fishaudio/fish-speech` GitHub repo's `fish_speech/models/dac/rvq.py`
/// (`DownsampleResidualVectorQuantize.decode`) and `modded_dac.py` (`DAC.decoder`, `ResidualUnit`,
/// `DecoderBlock`, causal conv helpers) -- see <see cref="FishSpeechCodecWeights"/>'s doc comment
/// for the full derivation, including the real correction that the 8-layer transformer
/// (`pre_module`) is NOT needed for decode-only inference.
///
/// <para><b>Real decode chain</b>: codes (1 semantic + 9 residual, all same time resolution) -&gt;
/// per-quantizer embedding lookup -&gt; out_proj -&gt; SUM (semantic + residual, no time-
/// upsampling within the quantizer itself, same flat pattern as Parler-TTS's DAC) -&gt; real
/// `post_module` (a bare RMSNorm, NOT a transformer) -&gt; 2x upsample stage (causal
/// ConvTranspose1d(k=2,stride=2) -&gt; `ConvNeXtBlock`) -&gt; `DAC.decoder` (causal conv -&gt; 4x
/// causal `DecoderBlock` -&gt; causal conv -&gt; Tanh).</para>
///
/// <para><b>All convolutions are CAUSAL (left-pad only)</b> -- confirmed from the real
/// `CausalConvNet`/`CausalTransConvNet` classes. For a same-resolution conv (kernel=7,
/// dilation=d, stride=1): pad LEFT by `(kernel-1)*dilation`, pad RIGHT by 0 -- output length
/// equals input length (derived from the real `get_extra_padding_for_conv1d` formula, confirmed
/// to reduce to a pure left-pad for stride=1). For the causal `ConvTranspose1d` (kernel=2*stride,
/// stride=stride): run the RAW (unpadded) transpose conv, giving length `(T+1)*stride`, then crop
/// `stride` samples off the RIGHT only (confirmed from the real `CausalTransConvNet.forward`'s
/// `unpad1d(x, (0, padding_right))` call, `padding_left=0` always for this kernel/stride
/// relationship).</para>
/// </summary>
public static class FishSpeechCodec
{
    /// <summary>Real `ResidualVectorQuantize.from_codes`-equivalent for one quantizer set: embedding lookup -> pointwise out_proj, summed across all quantizers in the set (semantic has 1, residual has 9). Same time resolution for every codebook, no upsampling.</summary>
    private static float[] QuantizerSetFromCodes(FishSpeechCodecQuantizerWeights[] quantizers, int[][] codes, int t)
    {
        var zq = new float[FishSpeechCodecWeights.LatentDim * t];
        for (int qi = 0; qi < quantizers.Length; qi++)
        {
            var q = quantizers[qi];
            var embed = new float[FishSpeechCodecWeights.CodebookDim * t];
            for (int ti = 0; ti < t; ti++)
            {
                int code = codes[qi][ti];
                int cbBase = code * FishSpeechCodecWeights.CodebookDim;
                for (int d = 0; d < FishSpeechCodecWeights.CodebookDim; d++)
                    embed[d * t + ti] = q.Codebook[cbBase + d];
            }
            var proj = FullConv1d(embed, FishSpeechCodecWeights.CodebookDim, FishSpeechCodecWeights.LatentDim, t, q.OutProjWeight, q.OutProjBias, kernel: 1, dilation: 1, causalPadLeft: 0);
            for (int i = 0; i < zq.Length; i++) zq[i] += proj[i];
        }
        return zq;
    }

    /// <summary>Real `nn.RMSNorm`-equivalent used for `post_module`: per-position RMS over the channel dim, weight only (no bias, matching the real bare `RMSNorm` module).</summary>
    private static float[] RmsNormChannels(float[] x, int channels, int t, float[] weight, float eps = 1e-5f)
    {
        var output = new float[x.Length];
        Parallel.For(0, t, ti =>
        {
            float sumSq = 0f;
            for (int c = 0; c < channels; c++) { float v = x[c * t + ti]; sumSq += v * v; }
            float invRms = 1f / MathF.Sqrt(sumSq / channels + eps);
            for (int c = 0; c < channels; c++) output[c * t + ti] = x[c * t + ti] * invRms * weight[c];
        });
        return output;
    }

    /// <summary>Real causal FULL Conv1d: left-pad only by `(kernel-1)*dilation` (equivalently `causalPadLeft`), zero right-pad. Weight layout [out, in, kernel] flat row-major.</summary>
    private static float[] FullConv1d(float[] x, int inCh, int outCh, int t, float[] weight, float[] bias, int kernel, int dilation, int causalPadLeft)
    {
        var output = new float[outCh * t];
        Parallel.For(0, outCh, oc =>
        {
            float b = bias[oc];
            int wOcBase = oc * inCh * kernel;
            int outBase = oc * t;
            for (int ti = 0; ti < t; ti++)
            {
                float sum = b;
                for (int ic = 0; ic < inCh; ic++)
                {
                    int xBase = ic * t;
                    int wBase = wOcBase + ic * kernel;
                    for (int k = 0; k < kernel; k++)
                    {
                        int src = ti - causalPadLeft + k * dilation;
                        if ((uint)src < (uint)t) sum += x[xBase + src] * weight[wBase + k];
                    }
                }
                output[outBase + ti] = sum;
            }
        });
        return output;
    }

    /// <summary>Real causal depthwise Conv1d (groups=channels), same left-pad convention as <see cref="FullConv1d"/>. Weight layout [channels, 1, kernel] flat -> effectively [channels, kernel].</summary>
    private static float[] DepthwiseConv1d(float[] x, int channels, int t, float[] weight, float[] bias, int kernel, int causalPadLeft)
    {
        var output = new float[x.Length];
        Parallel.For(0, channels, c =>
        {
            float b = bias[c];
            int wBase = c * kernel;
            int xBase = c * t;
            for (int ti = 0; ti < t; ti++)
            {
                float sum = b;
                for (int k = 0; k < kernel; k++)
                {
                    int src = ti - causalPadLeft + k;
                    if ((uint)src < (uint)t) sum += x[xBase + src] * weight[wBase + k];
                }
                output[xBase + ti] = sum;
            }
        });
        return output;
    }

    /// <summary>
    /// Real causal ConvTranspose1d: raw unpadded transpose conv (length becomes `(T-1)*stride+
    /// kernel`), then crop `pad = kernel - stride` samples off the RIGHT only (`padding_left=0`
    /// always, confirmed from the real `CausalTransConvNet.forward`'s `padding_left = pad -
    /// ceil(pad)` which is 0 whenever `pad` is already an integer, true for every real call site
    /// here). Weight layout [in, out, kernel] flat row-major.
    ///
    /// <para><b>Real bug found and fixed this fire</b>: the quantizer's own upsample stages call
    /// this with `kernel=stride` (real `rvq.py`: `transconvnet_type(..., kernel_size=factor,
    /// stride=factor)`), NOT `kernel=2*stride` like the DAC decoder's `DecoderBlock` -- an
    /// earlier version of this method hardcoded the crop to `stride`, which is only correct for
    /// the `kernel=2*stride` case (crop=stride) and silently produced HALF the correct output
    /// length for the `kernel=stride` case (crop=0, should be a no-op). Caught via the golden
    /// oracle producing an unexpectedly short PCM length (1024 instead of the expected 4096 for
    /// a 2-timestep input) before any cosine-similarity check was even needed.</para>
    /// </summary>
    private static float[] CausalConvTranspose1d(float[] x, int inCh, int outCh, int t, float[] weight, float[] bias, int kernel, int stride)
    {
        int rawT = (t - 1) * stride + kernel; // padding=0 in the underlying nn.ConvTranspose1d
        var raw = new float[outCh * rawT];
        Parallel.For(0, outCh, oc =>
        {
            float b = bias[oc];
            int dstBase = oc * rawT;
            for (int ti = 0; ti < rawT; ti++) raw[dstBase + ti] = b;

            for (int ic = 0; ic < inCh; ic++)
            {
                int srcBase = ic * t;
                int wBase = (ic * outCh + oc) * kernel;
                for (int ti = 0; ti < t; ti++)
                {
                    float v = x[srcBase + ti];
                    int outStart = ti * stride;
                    for (int k = 0; k < kernel; k++)
                        raw[dstBase + outStart + k] += v * weight[wBase + k];
                }
            }
        });

        int cropRight = kernel - stride; // real `pad = kernel_size - stride`, padding_left=0
        int outT = rawT - cropRight;
        var output = new float[outCh * outT];
        for (int oc = 0; oc < outCh; oc++)
            Array.Copy(raw, oc * rawT, output, oc * outT, outT);
        return output;
    }

    private static float[] Snake1d(float[] x, int channels, int t, float[] alpha)
    {
        var output = new float[x.Length];
        Parallel.For(0, channels, c =>
        {
            float a = alpha[c];
            float invA = 1f / (a + 1e-9f);
            int baseIdx = c * t;
            for (int i = 0; i < t; i++)
            {
                float v = x[baseIdx + i];
                float s = MathF.Sin(a * v);
                output[baseIdx + i] = v + invA * s * s;
            }
        });
        return output;
    }

    /// <summary>Real `ConvNeXtBlock`: causal depthwise conv (k=7) -> per-position (channels-last) LayerNorm -> Linear expand (4x) -> GELU -> Linear project -> per-channel LayerScale (`gamma`) -> residual.</summary>
    private static float[] ConvNeXtBlock(float[] x, int channels, int t, FishSpeechConvNeXtBlockWeights w)
    {
        var y = DepthwiseConv1d(x, channels, t, w.DwConvWeight, w.DwConvBias, kernel: 7, causalPadLeft: 6);

        int hidden = channels * 4;
        var output = new float[x.Length];
        Parallel.For(0, t, ti =>
        {
            // Gather this position's channel vector (channels-last for LayerNorm/Linear).
            var row = new float[channels];
            for (int c = 0; c < channels; c++) row[c] = y[c * t + ti];

            float mean = 0f;
            for (int c = 0; c < channels; c++) mean += row[c];
            mean /= channels;
            float variance = 0f;
            for (int c = 0; c < channels; c++) { float d = row[c] - mean; variance += d * d; }
            variance /= channels;
            float invStd = 1f / MathF.Sqrt(variance + 1e-6f);
            for (int c = 0; c < channels; c++) row[c] = (row[c] - mean) * invStd * w.NormWeight[c] + w.NormBias[c];

            var h = new float[hidden];
            for (int o = 0; o < hidden; o++)
            {
                float sum = w.PwConv1Bias[o];
                int wBase = o * channels;
                for (int c = 0; c < channels; c++) sum += row[c] * w.PwConv1Weight[wBase + c];
                h[o] = Gelu(sum);
            }

            for (int c = 0; c < channels; c++)
            {
                float sum = w.PwConv2Bias[c];
                int wBase = c * hidden;
                for (int o = 0; o < hidden; o++) sum += h[o] * w.PwConv2Weight[wBase + o];
                output[c * t + ti] = x[c * t + ti] + sum * w.Gamma[c];
            }
        });
        return output;
    }

    private static float Gelu(float x) => 0.5f * x * (1f + Erf(x / 1.4142135f));

    private static float Erf(float x)
    {
        float sign = MathF.Sign(x);
        x = MathF.Abs(x);
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float tt = 1f / (1f + p * x);
        float y = 1f - (((((a5 * tt + a4) * tt) + a3) * tt + a2) * tt + a1) * tt * MathF.Exp(-x * x);
        return sign * y;
    }

    private static float[] ResidualUnit(float[] x, int channels, int t, FishSpeechCodecResidualUnitWeights w, int dilation)
    {
        int padLeft = (7 - 1) * dilation;
        var y = Snake1d(x, channels, t, w.Alpha0);
        y = FullConv1d(y, channels, channels, t, w.Conv0Weight, w.Conv0Bias, kernel: 7, dilation: dilation, causalPadLeft: padLeft);
        y = Snake1d(y, channels, t, w.Alpha1);
        y = FullConv1d(y, channels, channels, t, w.Conv1Weight, w.Conv1Bias, kernel: 1, dilation: 1, causalPadLeft: 0);

        var output = new float[y.Length];
        for (int i = 0; i < y.Length; i++) output[i] = x[i] + y[i];
        return output;
    }

    private static (float[] Data, int T) DecoderBlock(float[] x, int inCh, int outCh, int t, FishSpeechCodecDecoderBlockWeights w, int stride)
    {
        var y = Snake1d(x, inCh, t, w.Alpha);
        int kernel = 2 * stride;
        var up = CausalConvTranspose1d(y, inCh, outCh, t, w.UpWeight, w.UpBias, kernel, stride);
        int outT = t * stride;

        var cur = up;
        cur = ResidualUnit(cur, outCh, outT, w.Res[0], dilation: 1);
        cur = ResidualUnit(cur, outCh, outT, w.Res[1], dilation: 3);
        cur = ResidualUnit(cur, outCh, outT, w.Res[2], dilation: 9);
        return (cur, outT);
    }

    /// <summary>Full real decode: 10 codebook streams (1 semantic + 9 residual, same time resolution) -> mono float32 PCM at 44.1kHz, range [-1, 1] (post-Tanh).</summary>
    public static float[] Decode(FishSpeechCodecWeights w, int[] semanticCodes, int[][] residualCodes)
    {
        int t = semanticCodes.Length;

        var zqSemantic = QuantizerSetFromCodes([w.SemanticQuantizer], [semanticCodes], t);
        var zqResidual = QuantizerSetFromCodes(w.ResidualQuantizers, residualCodes, t);
        var zq = new float[zqSemantic.Length];
        for (int i = 0; i < zq.Length; i++) zq[i] = zqSemantic[i] + zqResidual[i];

        zq = RmsNormChannels(zq, FishSpeechCodecWeights.LatentDim, t, w.PostModuleNormWeight);

        var x = zq;
        int curT = t;
        foreach (var stage in w.UpsampleStages)
        {
            x = CausalConvTranspose1d(x, FishSpeechCodecWeights.LatentDim, FishSpeechCodecWeights.LatentDim, curT, stage.ConvWeight, stage.ConvBias, kernel: 2, stride: 2);
            curT *= 2;
            x = ConvNeXtBlock(x, FishSpeechCodecWeights.LatentDim, curT, stage.Block);
        }

        x = FullConv1d(x, FishSpeechCodecWeights.LatentDim, FishSpeechCodecWeights.DecoderDim, curT, w.DecIn0Weight, w.DecIn0Bias, kernel: 7, dilation: 1, causalPadLeft: 6);

        int ch = FishSpeechCodecWeights.DecoderDim;
        for (int i = 0; i < FishSpeechCodecWeights.DecoderRates.Length; i++)
        {
            int outCh = ch / 2;
            (x, curT) = DecoderBlock(x, ch, outCh, curT, w.DecBlocks[i], FishSpeechCodecWeights.DecoderRates[i]);
            ch = outCh;
        }

        x = Snake1d(x, ch, curT, w.DecOutAlpha);
        var pcm = FullConv1d(x, ch, 1, curT, w.DecOutWeight, w.DecOutBias, kernel: 7, dilation: 1, causalPadLeft: 6);

        for (int i = 0; i < pcm.Length; i++) pcm[i] = MathF.Tanh(pcm[i]);
        return pcm;
    }
}
