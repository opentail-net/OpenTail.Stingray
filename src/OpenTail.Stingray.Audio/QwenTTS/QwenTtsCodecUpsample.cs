using System;
using System.IO;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Real weights for one Qwen3-TTS 12Hz codec upsample stage (`tok_dec.upsample.{0,1}.*`): a
/// causal `ConvTranspose1d(kernel=2,stride=2)` followed by a real `ConvNeXtBlock`. Two stages
/// total (`upsampling_ratios=[2,2]`), 4x temporal upsampling before the DAC decoder chain.
/// </summary>
public sealed class QwenTtsCodecUpsampleWeights
{
    public const int Channels = 1024;
    public const int ConvKernel = 2;
    public const int ConvStride = 2;
    public const int DwKernel = 7;
    public const int Expansion = 4096; // Channels * 4

    public float[] ConvWeight { get; } // real ConvTranspose1d native [in,out,kernel]
    public float[] ConvBias { get; }
    public float[] DwConvWeight { get; } // real depthwise [channels,1,kernel]
    public float[] DwConvBias { get; }
    public float[] NormWeight { get; }
    public float[] NormBias { get; }
    public float[] PwConv1Weight { get; } // Linear dim->4dim, native [out,in]
    public float[] PwConv1Bias { get; }
    public float[] PwConv2Weight { get; } // Linear 4dim->dim
    public float[] PwConv2Bias { get; }
    public float[] Gamma { get; } // real LEARNED LayerScale (initialized 1e-6, read from checkpoint)

    public QwenTtsCodecUpsampleWeights(GgufModel model, int stage)
    {
        string p = $"tok_dec.upsample.{stage}";
        ConvWeight = GetF32(model, $"{p}.conv.weight");
        ConvBias = GetF32(model, $"{p}.conv.bias");
        DwConvWeight = GetF32(model, $"{p}.dwconv.weight");
        DwConvBias = GetF32(model, $"{p}.dwconv.bias");
        NormWeight = GetF32(model, $"{p}.norm.weight");
        NormBias = GetF32(model, $"{p}.norm.bias");
        PwConv1Weight = GetF32(model, $"{p}.pwconv1.weight");
        PwConv1Bias = GetF32(model, $"{p}.pwconv1.bias");
        PwConv2Weight = GetF32(model, $"{p}.pwconv2.weight");
        PwConv2Bias = GetF32(model, $"{p}.pwconv2.bias");
        Gamma = GetF32(model, $"{p}.gamma");
    }

    private static float[] GetF32(GgufModel model, string name)
    {
        var info = model.FindTensor(name) ?? throw new InvalidDataException($"QwenTTS codec GGUF missing required tensor '{name}'.");
        var bytes = model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }
}

/// <summary>
/// Real forward for one upsample stage, transcribed from the real official
/// `Qwen3TTSTokenizerV2ConvNeXtBlock`/`Qwen3TTSTokenizerV2CausalTransConvNet` (cross-checked
/// against the equivalent, already golden-verified pattern in this project's
/// `FishSpeechCodec.CausalConvTranspose1d` -- same real `kernel=stride` case, crop=0).
/// </summary>
public static class QwenTtsCodecUpsample
{
    public static float[][] Forward(QwenTtsCodecUpsampleWeights w, float[][] input)
    {
        var upsampled = CausalConvTranspose1d(input, w.ConvWeight, w.ConvBias);
        return ConvNeXtBlock(upsampled, w);
    }

    /// <summary>Real causal ConvTranspose1d(kernel=stride=2): crop = kernel-stride = 0 (no crop needed) -- same real formula already proven for Fish Speech's codec upsample stages (also kernel=stride).</summary>
    private static float[][] CausalConvTranspose1d(float[][] input, float[] weight, float[] bias)
    {
        int t = input.Length;
        int ch = QwenTtsCodecUpsampleWeights.Channels;
        int kernel = QwenTtsCodecUpsampleWeights.ConvKernel;
        int stride = QwenTtsCodecUpsampleWeights.ConvStride;
        int outT = t * stride; // crop = kernel - stride = 0

        var output = new float[outT][];
        for (int i = 0; i < outT; i++) output[i] = (float[])bias.Clone();

        // Real ConvTranspose1d native weight layout [in, out, kernel].
        for (int ti = 0; ti < t; ti++)
        {
            var src = input[ti];
            int outStart = ti * stride;
            for (int ic = 0; ic < ch; ic++)
            {
                float v = src[ic];
                if (v == 0f) continue;
                int wIcBase = ic * ch * kernel;
                for (int k = 0; k < kernel; k++)
                {
                    var dstRow = output[outStart + k];
                    int wBase = wIcBase + k;
                    for (int oc = 0; oc < ch; oc++)
                        dstRow[oc] += v * weight[wBase + oc * kernel];
                }
            }
        }
        return output;
    }

    /// <summary>Real ConvNeXtBlock: causal depthwise conv (k=7, left-pad=6) -&gt; channels-last LayerNorm -&gt; Linear expand 4x -&gt; GELU -&gt; Linear project -&gt; per-channel gamma (LEARNED LayerScale) -&gt; residual.</summary>
    private static float[][] ConvNeXtBlock(float[][] x, QwenTtsCodecUpsampleWeights w)
    {
        int t = x.Length;
        int ch = QwenTtsCodecUpsampleWeights.Channels;
        int hidden = QwenTtsCodecUpsampleWeights.Expansion;
        int dwKernel = QwenTtsCodecUpsampleWeights.DwKernel;
        int padLeft = dwKernel - 1;

        var dw = new float[t][];
        Parallel.For(0, t, ti =>
        {
            var row = new float[ch];
            for (int c = 0; c < ch; c++)
            {
                float sum = w.DwConvBias[c];
                int wBase = c * dwKernel; // depthwise [channels,1,kernel] flat
                for (int k = 0; k < dwKernel; k++)
                {
                    int srcT = ti - padLeft + k;
                    if (srcT < 0) continue;
                    sum += x[srcT][c] * w.DwConvWeight[wBase + k];
                }
                row[c] = sum;
            }
            dw[ti] = row;
        });

        var output = new float[t][];
        Parallel.For(0, t, ti =>
        {
            var row = dw[ti];
            float mean = 0f;
            for (int c = 0; c < ch; c++) mean += row[c];
            mean /= ch;
            float variance = 0f;
            for (int c = 0; c < ch; c++) { float d = row[c] - mean; variance += d * d; }
            variance /= ch;
            float invStd = 1f / MathF.Sqrt(variance + 1e-6f);

            var normed = new float[ch];
            for (int c = 0; c < ch; c++) normed[c] = (row[c] - mean) * invStd * w.NormWeight[c] + w.NormBias[c];

            var expanded = new float[hidden];
            for (int o = 0; o < hidden; o++)
            {
                float sum = w.PwConv1Bias[o];
                int wBase = o * ch;
                for (int c = 0; c < ch; c++) sum += normed[c] * w.PwConv1Weight[wBase + c];
                expanded[o] = Gelu(sum);
            }

            var outRow = new float[ch];
            for (int c = 0; c < ch; c++)
            {
                float sum = w.PwConv2Bias[c];
                int wBase = c * hidden;
                for (int o = 0; o < hidden; o++) sum += expanded[o] * w.PwConv2Weight[wBase + o];
                outRow[c] = x[ti][c] + sum * w.Gamma[c];
            }
            output[ti] = outRow;
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
}
