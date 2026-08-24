using System;
using System.Threading.Tasks;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Real forward pass for the Qwen3-TTS 12Hz codec's DAC decoder chain: causal pre-conv (k=7,
/// 1024-&gt;1536) -&gt; 4 real `DecoderBlock`s (SnakeBeta -&gt; causal `ConvTranspose1d`(kernel=2×rate)
/// -&gt; 3x `ResidualUnit`) -&gt; final SnakeBeta -&gt; final causal conv (96-&gt;1, k=7) -&gt; clamp(-1,1).
/// Transcribed from the real official DAC decoder / local `examples/qwentts.cpp`'s
/// `dac-decoder-v2.h`.
/// </summary>
public static class QwenTtsCodecDac
{
    /// <summary>input: [T][1024] (post-upsample codec latents). output: raw waveform samples (mono).</summary>
    public static float[] Forward(QwenTtsCodecDacWeights w, float[][] input)
    {
        var x = CausalConv1d(input, w.PreConvWeight, w.PreConvBias, inCh: 1024, outCh: 1536, kernel: 7, dilation: 1);

        for (int b = 0; b < 4; b++)
            x = DecoderBlock(x, w.Blocks[b], QwenTtsCodecDacWeights.Channels[b], QwenTtsCodecDacWeights.Channels[b + 1], QwenTtsCodecDacWeights.Rates[b]);

        x = SnakeBeta(x, w.FinalSnakeAlpha, w.FinalSnakeBeta);
        var wav2d = CausalConv1d(x, w.FinalConvWeight, w.FinalConvBias, inCh: 96, outCh: 1, kernel: 7, dilation: 1);

        var wav = new float[wav2d.Length];
        for (int i = 0; i < wav.Length; i++) wav[i] = Math.Clamp(wav2d[i][0], -1f, 1f);
        return wav;
    }

    internal static float[][] DecoderBlockForTest(float[][] x, QwenTtsCodecDacBlockWeights w, int inCh, int outCh, int rate) => DecoderBlock(x, w, inCh, outCh, rate);

    internal static float[][] CausalConv1dForTest(float[][] input, float[] weight, float[] bias, int inCh, int outCh, int kernel, int dilation) => CausalConv1d(input, weight, bias, inCh, outCh, kernel, dilation);

    private static float[][] DecoderBlock(float[][] x, QwenTtsCodecDacBlockWeights w, int inCh, int outCh, int rate)
    {
        var snaked = SnakeBeta(x, w.SnakeAlpha, w.SnakeBeta);
        var up = CausalConvTranspose1d(snaked, w.ConvTWeight, w.ConvTBias, inCh, outCh, kernel: 2 * rate, stride: rate);

        var r = up;
        int[] dilations = [1, 3, 9];
        for (int i = 0; i < 3; i++)
            r = ResidualUnit(r, w.Res[i], outCh, dilations[i]);
        return r;
    }

    private static float[][] ResidualUnit(float[][] x, QwenTtsCodecResidualUnitWeights w, int ch, int dilation)
    {
        var y = SnakeBeta(x, w.Act1Alpha, w.Act1Beta);
        y = CausalConv1d(y, w.Conv1Weight, w.Conv1Bias, inCh: ch, outCh: ch, kernel: 7, dilation: dilation);
        y = SnakeBeta(y, w.Act2Alpha, w.Act2Beta);
        y = CausalConv1d(y, w.Conv2Weight, w.Conv2Bias, inCh: ch, outCh: ch, kernel: 1, dilation: 1);

        var output = new float[x.Length][];
        for (int t = 0; t < x.Length; t++)
        {
            var row = new float[ch];
            for (int c = 0; c < ch; c++) row[c] = x[t][c] + y[t][c];
            output[t] = row;
        }
        return output;
    }

    /// <summary>Real SnakeBeta: stored alpha/beta are EXPONENTIATED before use (known porting trap): x + (1/beta_exp) * sin(alpha_exp*x)^2.</summary>
    private static float[][] SnakeBeta(float[][] x, float[] alpha, float[] beta)
    {
        int t = x.Length;
        int ch = alpha.Length;
        var alphaExp = new float[ch];
        var betaExp = new float[ch];
        for (int c = 0; c < ch; c++)
        {
            alphaExp[c] = MathF.Exp(alpha[c]);
            betaExp[c] = MathF.Exp(beta[c]);
        }

        var output = new float[t][];
        for (int ti = 0; ti < t; ti++)
        {
            var src = x[ti];
            var row = new float[ch];
            for (int c = 0; c < ch; c++)
            {
                float s = MathF.Sin(alphaExp[c] * src[c]);
                row[c] = src[c] + (1f / (betaExp[c] + 1e-9f)) * s * s;
            }
            output[ti] = row;
        }
        return output;
    }

    /// <summary>
    /// Real causal Conv1d: left-zero-pad by (kernel-1)*dilation. Weight layout [out,in,kernel] flat row-major.
    ///
    /// <para>im2col + GEMM (see <c>FishSpeechCodec.FullConv1d</c>'s doc comment for the technique):
    /// `input[srcT]` rows are already contiguous per timestep (time-major [T][C] layout here,
    /// unlike the channel-major [C,T] layout in the other codecs), so the weight is transposed
    /// once per call from [oc,ic,k] to [oc,k,ic] to match, letting the gather use plain
    /// `Array.Copy` per (ti,k) slice instead of a scattered per-element loop. Each output channel
    /// then reduces to one AVX2/FMA <see cref="SimdKernels.DotF32"/> call per timestep.</para>
    /// </summary>
    private static unsafe float[][] CausalConv1d(float[][] input, float[] weight, float[] bias, int inCh, int outCh, int kernel, int dilation)
    {
        int t = input.Length;
        int padLeft = (kernel - 1) * dilation;
        int rowLen = kernel * inCh;

        var weightT = new float[outCh * rowLen]; // [oc][k][ic]
        Parallel.For(0, outCh, oc =>
        {
            int wOcBase = oc * inCh * kernel;
            int wtOcBase = oc * rowLen;
            for (int ic = 0; ic < inCh; ic++)
                for (int k = 0; k < kernel; k++)
                    weightT[wtOcBase + k * inCh + ic] = weight[wOcBase + ic * kernel + k];
        });

        var col = new float[t * rowLen]; // [ti][k][ic]
        Parallel.For(0, t, ti =>
        {
            int rowBase = ti * rowLen;
            for (int k = 0; k < kernel; k++)
            {
                int srcT = ti - padLeft + k * dilation;
                if (srcT < 0) continue;
                Array.Copy(input[srcT], 0, col, rowBase + k * inCh, inCh);
            }
        });

        var output = new float[t][];
        for (int ti = 0; ti < t; ti++) output[ti] = new float[outCh];
        fixed (float* colPtr = col, weightPtr = weightT)
        {
            var colLocal = colPtr;
            var weightLocal = weightPtr;
            Parallel.For(0, outCh, oc =>
            {
                float b = bias[oc];
                float* wOc = weightLocal + oc * rowLen;
                for (int ti = 0; ti < t; ti++)
                    output[ti][oc] = b + SimdKernels.DotF32(wOc, colLocal + ti * rowLen, rowLen);
            });
        }
        return output;
    }

    /// <summary>Real causal ConvTranspose1d: kernel=2*rate, stride=rate, crop=kernel-stride=rate (real DAC convention, different ratio than the ConvNeXt upsample's kernel=stride case). Native weight layout [in,out,kernel].</summary>
    private static float[][] CausalConvTranspose1d(float[][] input, float[] weight, float[] bias, int inCh, int outCh, int kernel, int stride)
    {
        int t = input.Length;
        int fullT = (t - 1) * stride + kernel;
        int crop = kernel - stride;

        var full = new float[fullT][];
        for (int i = 0; i < fullT; i++) full[i] = (float[])bias.Clone();

        for (int ti = 0; ti < t; ti++)
        {
            var src = input[ti];
            int outStart = ti * stride;
            for (int ic = 0; ic < inCh; ic++)
            {
                float v = src[ic];
                if (v == 0f) continue;
                int wIcBase = ic * outCh * kernel;
                for (int k = 0; k < kernel; k++)
                {
                    var dstRow = full[outStart + k];
                    int wBase = wIcBase + k;
                    for (int oc = 0; oc < outCh; oc++)
                        dstRow[oc] += v * weight[wBase + oc * kernel];
                }
            }
        }

        int outT = fullT - crop;
        var output = new float[outT][];
        Array.Copy(full, output, outT);
        return output;
    }
}
