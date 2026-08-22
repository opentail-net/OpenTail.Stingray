using System;

namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Real forward pass for the Qwen3-TTS ECAPA-TDNN-style speaker encoder. Input: real log-mel
/// features `[T, 128]` (frontend mel extraction is out of scope here -- this operates on mel
/// features directly, same isolation strategy already used for the codec's RVQ decode test).
/// Real conv padding convention: `padding="same", padding_mode="reflect"` on every
/// `TimeDelayNetBlock` conv (distinct from the codec decoder's causal left-zero-pad).
/// </summary>
public static class QwenTtsSpeakerEncoder
{
    public static float[] Forward(QwenTtsSpeakerEncoderWeights w, float[][] mel)
    {
        var x = ReflectPadConv1d(mel, w.Conv0Weight, w.Conv0Bias, inCh: QwenTtsSpeakerEncoderWeights.MelDim, outCh: QwenTtsSpeakerEncoderWeights.Channels, kernel: 5, dilation: 1);
        Relu(x);

        var blockOutputs = new float[3][][];
        var cur = x;
        for (int b = 0; b < 3; b++)
        {
            cur = SeRes2NetBlock(cur, w.Blocks[b]);
            blockOutputs[b] = cur;
        }

        int t = mel.Length;
        var concat = new float[t][];
        for (int ti = 0; ti < t; ti++)
        {
            var row = new float[QwenTtsSpeakerEncoderWeights.MfaOutDim];
            for (int b = 0; b < 3; b++)
                Array.Copy(blockOutputs[b][ti], 0, row, b * QwenTtsSpeakerEncoderWeights.Channels, QwenTtsSpeakerEncoderWeights.Channels);
            concat[ti] = row;
        }

        var mfa = ReflectPadConv1d(concat, w.MfaWeight, w.MfaBias, inCh: QwenTtsSpeakerEncoderWeights.MfaOutDim, outCh: QwenTtsSpeakerEncoderWeights.MfaOutDim, kernel: 1, dilation: 1);
        Relu(mfa);

        var pooled = AttentiveStatisticsPooling(mfa, w.Asp); // [3072]

        return Linear1x1(pooled, w.FcWeight, w.FcBias, inDim: pooled.Length, outDim: w.EncDim);
    }

    private static float[][] SeRes2NetBlock(float[][] x, QwenTtsSpeakerEncoderBlockWeights w)
    {
        int t = x.Length;
        int ch = QwenTtsSpeakerEncoderWeights.Channels;

        var h = ReflectPadConv1d(x, w.Tdnn1Weight, w.Tdnn1Bias, inCh: ch, outCh: ch, kernel: 1, dilation: 1);
        Relu(h);

        h = Res2Net(h, w);
        Relu(h);

        h = ReflectPadConv1d(h, w.Tdnn2Weight, w.Tdnn2Bias, inCh: ch, outCh: ch, kernel: 1, dilation: 1);
        Relu(h);

        h = SqueezeExcitation(h, w);

        var output = new float[t][];
        for (int ti = 0; ti < t; ti++)
        {
            var row = new float[ch];
            for (int c = 0; c < ch; c++) row[c] = x[ti][c] + h[ti][c];
            output[ti] = row;
        }
        return output;
    }

    /// <summary>Real Res2Net: input split into 8 chunks of 64 channels. Branch 0 passes through unconvolved. Branches 1-7 each apply a dilated conv, with branches 2-7 taking `x[i]+output[i-1]` as input (cascading) -- 7 real conv branches, not 8.</summary>
    private static float[][] Res2Net(float[][] x, QwenTtsSpeakerEncoderBlockWeights w)
    {
        int t = x.Length;
        int branchCh = QwenTtsSpeakerEncoderWeights.Res2NetBranchChannels;
        int scale = QwenTtsSpeakerEncoderWeights.Res2NetScale;

        var splits = new float[scale][][];
        for (int s = 0; s < scale; s++)
        {
            splits[s] = new float[t][];
            for (int ti = 0; ti < t; ti++)
            {
                var row = new float[branchCh];
                Array.Copy(x[ti], s * branchCh, row, 0, branchCh);
                splits[s][ti] = row;
            }
        }

        var outputs = new float[scale][][];
        outputs[0] = splits[0];
        for (int s = 1; s < scale; s++)
        {
            float[][] branchInput;
            if (s == 1)
            {
                branchInput = splits[1];
            }
            else
            {
                branchInput = new float[t][];
                for (int ti = 0; ti < t; ti++)
                {
                    var row = new float[branchCh];
                    for (int c = 0; c < branchCh; c++) row[c] = splits[s][ti][c] + outputs[s - 1][ti][c];
                    branchInput[ti] = row;
                }
            }
            var conv = ReflectPadConv1d(branchInput, w.Res2NetWeight[s - 1], w.Res2NetBias[s - 1], inCh: branchCh, outCh: branchCh, kernel: 3, dilation: w.Dilation);
            Relu(conv);
            outputs[s] = conv;
        }

        var result = new float[t][];
        for (int ti = 0; ti < t; ti++)
        {
            var row = new float[QwenTtsSpeakerEncoderWeights.Channels];
            for (int s = 0; s < scale; s++)
                Array.Copy(outputs[s][ti], 0, row, s * branchCh, branchCh);
            result[ti] = row;
        }
        return result;
    }

    /// <summary>Real SE: mean-over-time -&gt; 1x1 conv 512-&gt;128 -&gt; ReLU -&gt; 1x1 conv 128-&gt;512 -&gt; Sigmoid -&gt; multiply (broadcast over time).</summary>
    private static float[][] SqueezeExcitation(float[][] x, QwenTtsSpeakerEncoderBlockWeights w)
    {
        int t = x.Length;
        int ch = QwenTtsSpeakerEncoderWeights.Channels;
        var mean = new float[ch];
        for (int ti = 0; ti < t; ti++)
            for (int c = 0; c < ch; c++) mean[c] += x[ti][c];
        for (int c = 0; c < ch; c++) mean[c] /= t;

        var squeezed = Linear1x1(mean, w.SeConv1Weight, w.SeConv1Bias, inDim: ch, outDim: QwenTtsSpeakerEncoderWeights.AttentionChannels);
        for (int i = 0; i < squeezed.Length; i++) squeezed[i] = MathF.Max(0f, squeezed[i]);

        var gate = Linear1x1(squeezed, w.SeConv2Weight, w.SeConv2Bias, inDim: QwenTtsSpeakerEncoderWeights.AttentionChannels, outDim: ch);
        for (int i = 0; i < gate.Length; i++) gate[i] = 1f / (1f + MathF.Exp(-gate[i]));

        var output = new float[t][];
        for (int ti = 0; ti < t; ti++)
        {
            var row = new float[ch];
            for (int c = 0; c < ch; c++) row[c] = x[ti][c] * gate[c];
            output[ti] = row;
        }
        return output;
    }

    /// <summary>Real ASP: concat(x, repeated-mean, repeated-std)=4608 -&gt; TDNN 4608-&gt;128 (ReLU) -&gt; Tanh -&gt; 1x1 conv 128-&gt;1536 -&gt; softmax over time -&gt; weighted mean+std (each 1536) -&gt; concat=3072.</summary>
    private static float[] AttentiveStatisticsPooling(float[][] x, QwenTtsSpeakerEncoderAspWeights w)
    {
        int t = x.Length;
        int ch = QwenTtsSpeakerEncoderWeights.MfaOutDim;

        var mean = new float[ch];
        for (int ti = 0; ti < t; ti++)
            for (int c = 0; c < ch; c++) mean[c] += x[ti][c];
        for (int c = 0; c < ch; c++) mean[c] /= t;

        var std = new float[ch];
        for (int ti = 0; ti < t; ti++)
            for (int c = 0; c < ch; c++) { float d = x[ti][c] - mean[c]; std[c] += d * d; }
        for (int c = 0; c < ch; c++) std[c] = MathF.Sqrt(MathF.Max(std[c] / t, 1e-12f));

        var concat = new float[t][];
        for (int ti = 0; ti < t; ti++)
        {
            var row = new float[ch * 3];
            Array.Copy(x[ti], 0, row, 0, ch);
            Array.Copy(mean, 0, row, ch, ch);
            Array.Copy(std, 0, row, ch * 2, ch);
            concat[ti] = row;
        }

        var hidden = ReflectPadConv1d(concat, w.TdnnWeight, w.TdnnBias, inCh: ch * 3, outCh: QwenTtsSpeakerEncoderWeights.AttentionChannels, kernel: 1, dilation: 1);
        for (int ti = 0; ti < t; ti++)
            for (int c = 0; c < hidden[ti].Length; c++)
                hidden[ti][c] = MathF.Max(0f, hidden[ti][c]);

        for (int ti = 0; ti < t; ti++)
            for (int c = 0; c < hidden[ti].Length; c++)
                hidden[ti][c] = MathF.Tanh(hidden[ti][c]);

        var logits = ReflectPadConv1d(hidden, w.ConvWeight, w.ConvBias, inCh: QwenTtsSpeakerEncoderWeights.AttentionChannels, outCh: ch, kernel: 1, dilation: 1);

        var attn = new float[t][];
        for (int ti = 0; ti < t; ti++) attn[ti] = new float[ch];
        for (int c = 0; c < ch; c++)
        {
            float maxV = float.NegativeInfinity;
            for (int ti = 0; ti < t; ti++) maxV = MathF.Max(maxV, logits[ti][c]);
            float sum = 0f;
            for (int ti = 0; ti < t; ti++)
            {
                float e = MathF.Exp(logits[ti][c] - maxV);
                attn[ti][c] = e;
                sum += e;
            }
            for (int ti = 0; ti < t; ti++) attn[ti][c] /= sum;
        }

        var wMean = new float[ch];
        for (int ti = 0; ti < t; ti++)
            for (int c = 0; c < ch; c++) wMean[c] += attn[ti][c] * x[ti][c];

        var wVar = new float[ch];
        for (int ti = 0; ti < t; ti++)
            for (int c = 0; c < ch; c++) { float d = x[ti][c] - wMean[c]; wVar[c] += attn[ti][c] * d * d; }
        var wStd = new float[ch];
        for (int c = 0; c < ch; c++) wStd[c] = MathF.Sqrt(MathF.Max(wVar[c], 1e-12f));

        var result = new float[ch * 2];
        Array.Copy(wMean, 0, result, 0, ch);
        Array.Copy(wStd, 0, result, ch, ch);
        return result;
    }

    private static void Relu(float[][] x)
    {
        for (int ti = 0; ti < x.Length; ti++)
            for (int c = 0; c < x[ti].Length; c++)
                x[ti][c] = MathF.Max(0f, x[ti][c]);
    }

    private static float[] Linear1x1(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = bias[o];
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += input[i] * weight[wBase + i];
            output[o] = sum;
        }
        return output;
    }

    /// <summary>Real "same" padding with reflect mode: effective_kernel=(kernel-1)*dilation+1, total pad=effective_kernel-1 split as floor/ceil across left/right, reflecting off the sequence edges (no boundary-sample duplication).</summary>
    private static float[][] ReflectPadConv1d(float[][] input, float[] weight, float[] bias, int inCh, int outCh, int kernel, int dilation)
    {
        int t = input.Length;
        if (kernel == 1)
        {
            var output1 = new float[t][];
            for (int ti = 0; ti < t; ti++) output1[ti] = Linear1x1(input[ti], weight, bias, inCh, outCh);
            return output1;
        }

        int effectiveKernel = (kernel - 1) * dilation + 1;
        int totalPad = effectiveKernel - 1;
        int padLeft = totalPad / 2;
        int padRight = totalPad - padLeft;

        var output = new float[t][];
        for (int ti = 0; ti < t; ti++)
        {
            var row = new float[outCh];
            for (int oc = 0; oc < outCh; oc++)
            {
                float sum = bias[oc];
                int wOcBase = oc * inCh * kernel;
                for (int k = 0; k < kernel; k++)
                {
                    int srcT = ti - padLeft + k * dilation;
                    srcT = ReflectIndex(srcT, t);
                    var srcRow = input[srcT];
                    int wBase = wOcBase + k;
                    for (int ic = 0; ic < inCh; ic++)
                        sum += srcRow[ic] * weight[wBase + ic * kernel];
                }
                row[oc] = sum;
            }
            output[ti] = row;
        }
        _ = padRight;
        return output;
    }

    /// <summary>PyTorch reflect-pad index mapping: no boundary-sample duplication (period 2*(t-1)).</summary>
    private static int ReflectIndex(int idx, int t)
    {
        if (t == 1) return 0;
        int period = 2 * (t - 1);
        int m = ((idx % period) + period) % period;
        return m < t ? m : period - m;
    }
}
