
namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Real weights + forward for the Qwen3-TTS 12Hz codec's pre-conv (`tok_dec.pre_conv.*`): a
/// single causal Conv1d, 512(RVQ hidden)-&gt;1024(latent_dim), kernel=3, bridging the RVQ decode
/// output into <see cref="QwenTtsCodecTransformer"/>'s input. Real causal padding formula
/// (`Qwen3TTSTokenizerV2CausalConvNet`): `effective_kernel=(kernel-1)*dilation+1`,
/// `padding=effective_kernel-stride` -- for kernel=3/stride=1/dilation=1: padding=2, all on the
/// LEFT (zero-pad), none on the right.
/// </summary>
public sealed class QwenTtsCodecPreConvWeights
{
    public const int InChannels = 512;
    public const int OutChannels = 1024;
    public const int Kernel = 3;

    public float[] Weight { get; }
    public float[] Bias { get; }

    public QwenTtsCodecPreConvWeights(GgufModel model)
    {
        Weight = GetF32(model, "tok_dec.pre_conv.weight");
        Bias = GetF32(model, "tok_dec.pre_conv.bias");
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

public static class QwenTtsCodecPreConv
{
    /// <summary>Real causal Conv1d: input[T][512] -&gt; output[T][1024], left-zero-pad by (kernel-1)=2, weight layout [out,in,kernel] flat row-major (real PyTorch native order, confirmed via this project's established "GGUF displayed shape reversed, flat bytes match PyTorch row-major" convention).</summary>
    public static float[][] Forward(QwenTtsCodecPreConvWeights w, float[][] input)
    {
        int t = input.Length;
        int inCh = QwenTtsCodecPreConvWeights.InChannels;
        int outCh = QwenTtsCodecPreConvWeights.OutChannels;
        int kernel = QwenTtsCodecPreConvWeights.Kernel;
        int padLeft = kernel - 1;

        var output = new float[t][];
        Parallel.For(0, t, ti =>
        {
            var row = new float[outCh];
            for (int oc = 0; oc < outCh; oc++)
            {
                float sum = w.Bias[oc];
                int wOcBase = oc * inCh * kernel;
                for (int k = 0; k < kernel; k++)
                {
                    int srcT = ti - padLeft + k;
                    if (srcT < 0) continue;
                    var srcRow = input[srcT];
                    int wBase = wOcBase + k; // weight[oc, ic, k], flat index = oc*inCh*kernel + ic*kernel + k
                    for (int ic = 0; ic < inCh; ic++)
                        sum += srcRow[ic] * w.Weight[wBase + ic * kernel];
                }
                row[oc] = sum;
            }
            output[ti] = row;
        });
        return output;
    }
}
