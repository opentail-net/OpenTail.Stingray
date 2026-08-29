
namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Real weights for the Qwen3-TTS 12Hz codec's DAC decoder chain (`tok_dec.dec.{0..6}.*`).
///
/// <para>Real structure, confirmed via `list-tensors` real tensor shapes and cross-checked
/// against the official `Qwen3TTSTokenizerV2DecoderDecoderBlock`/`...ResidualUnit`: `dec.0` =
/// pre-conv (causal k=7, 1024-&gt;1536); `dec.{1..4}` = 4 real `DecoderBlock`s, each
/// SnakeBeta(in_dim) -&gt; causal `ConvTranspose1d`(kernel=2×rate, stride=rate) -&gt; 3x
/// `ResidualUnit` (dilations 1,3,9); real channel progression 1536-&gt;768-&gt;384-&gt;192-&gt;96,
/// real rates `[8,5,4,3]` (confirmed via each `conv_t` tensor's real kernel width = 2×rate);
/// `dec.5` = final SnakeBeta (96 channels); `dec.6` = final causal conv (96-&gt;1, k=7).</para>
///
/// <para>Real SnakeBeta gotcha (flagged explicitly, not silently assumed): the stored `alpha`/
/// `beta` values must be EXPONENTIATED before use in <c>x + (1/beta)*sin(alpha*x)^2</c> -- a
/// known real porting trap.</para>
/// </summary>
public sealed class QwenTtsCodecDacWeights
{
    public static readonly int[] Rates = [8, 5, 4, 3];
    public static readonly int[] Channels = [1536, 768, 384, 192, 96];

    public float[] PreConvWeight { get; }
    public float[] PreConvWeightT { get; }
    public float[] PreConvBias { get; }
    public QwenTtsCodecDacBlockWeights[] Blocks { get; } = new QwenTtsCodecDacBlockWeights[4];
    public float[] FinalSnakeAlpha { get; }
    public float[] FinalSnakeBeta { get; }
    public float[] FinalConvWeight { get; }
    public float[] FinalConvWeightT { get; }
    public float[] FinalConvBias { get; }

    public QwenTtsCodecDacWeights(GgufModel model)
    {
        PreConvWeight = GetF32(model, "tok_dec.dec.0.conv.weight");
        PreConvWeightT = TransposeConvWeight(PreConvWeight, inCh: 1024, outCh: 1536, kernel: 7);
        PreConvBias = GetF32(model, "tok_dec.dec.0.conv.bias");

        for (int b = 0; b < 4; b++)
        {
            int inCh = Channels[b];
            int outCh = Channels[b + 1];
            string p = $"tok_dec.dec.{b + 1}";
            var res = new QwenTtsCodecResidualUnitWeights[3];
            for (int r = 0; r < 3; r++)
            {
                string rp = $"{p}.res.{r}";
                var c1w = GetF32(model, $"{rp}.conv1.weight");
                var c2w = GetF32(model, $"{rp}.conv2.weight");
                res[r] = new QwenTtsCodecResidualUnitWeights
                {
                    Act1Alpha = GetF32(model, $"{rp}.act1.alpha"),
                    Act1Beta = GetF32(model, $"{rp}.act1.beta"),
                    Conv1Weight = c1w,
                    Conv1WeightT = TransposeConvWeight(c1w, inCh: outCh, outCh: outCh, kernel: 7),
                    Conv1Bias = GetF32(model, $"{rp}.conv1.bias"),
                    Act2Alpha = GetF32(model, $"{rp}.act2.alpha"),
                    Act2Beta = GetF32(model, $"{rp}.act2.beta"),
                    Conv2Weight = c2w,
                    Conv2WeightT = TransposeConvWeight(c2w, inCh: outCh, outCh: outCh, kernel: 1),
                    Conv2Bias = GetF32(model, $"{rp}.conv2.bias"),
                };
            }
            var convTw = GetF32(model, $"{p}.conv_t.weight");
            int kernelT = 2 * Rates[b];
            Blocks[b] = new QwenTtsCodecDacBlockWeights
            {
                SnakeAlpha = GetF32(model, $"{p}.snake.alpha"),
                SnakeBeta = GetF32(model, $"{p}.snake.beta"),
                ConvTWeight = convTw,
                ConvTWeightT = TransposeConvTWeight(convTw, inCh, outCh, kernelT),
                ConvTBias = GetF32(model, $"{p}.conv_t.bias"),
                Res = res,
            };
        }

        FinalSnakeAlpha = GetF32(model, "tok_dec.dec.5.snake.alpha");
        FinalSnakeBeta = GetF32(model, "tok_dec.dec.5.snake.beta");
        FinalConvWeight = GetF32(model, "tok_dec.dec.6.conv.weight");
        FinalConvWeightT = TransposeConvWeight(FinalConvWeight, inCh: 96, outCh: 1, kernel: 7);
        FinalConvBias = GetF32(model, "tok_dec.dec.6.conv.bias");
    }

    public static float[] TransposeConvWeight(float[] weight, int inCh, int outCh, int kernel)
    {
        int rowLen = kernel * inCh;
        var weightT = new float[outCh * rowLen];
        for (int oc = 0; oc < outCh; oc++)
        {
            int wOcBase = oc * inCh * kernel;
            int wtOcBase = oc * rowLen;
            for (int ic = 0; ic < inCh; ic++)
                for (int k = 0; k < kernel; k++)
                    weightT[wtOcBase + k * inCh + ic] = weight[wOcBase + ic * kernel + k];
        }
        return weightT;
    }

    public static float[] TransposeConvTWeight(float[] weight, int inCh, int outCh, int kernel)
    {
        int rowLen = kernel * outCh;
        var weightT = new float[inCh * rowLen];
        for (int ic = 0; ic < inCh; ic++)
        {
            int wIcBase = ic * outCh * kernel;
            int wtIcBase = ic * rowLen;
            for (int oc = 0; oc < outCh; oc++)
                for (int k = 0; k < kernel; k++)
                    weightT[wtIcBase + k * outCh + oc] = weight[wIcBase + oc * kernel + k];
        }
        return weightT;
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

public sealed class QwenTtsCodecResidualUnitWeights
{
    public required float[] Act1Alpha { get; init; }
    public required float[] Act1Beta { get; init; }
    public required float[] Conv1Weight { get; init; } // causal, dilated, k=7
    public required float[] Conv1WeightT { get; init; } // pre-transposed [oc, k, ic]
    public required float[] Conv1Bias { get; init; }
    public required float[] Act2Alpha { get; init; }
    public required float[] Act2Beta { get; init; }
    public required float[] Conv2Weight { get; init; } // causal, k=1
    public required float[] Conv2WeightT { get; init; } // pre-transposed [oc, k, ic]
    public required float[] Conv2Bias { get; init; }
}

public sealed class QwenTtsCodecDacBlockWeights
{
    public required float[] SnakeAlpha { get; init; }
    public required float[] SnakeBeta { get; init; }
    public required float[] ConvTWeight { get; init; } // causal ConvTranspose1d, kernel=2*rate
    public required float[] ConvTWeightT { get; init; } // pre-transposed [inCh, kernel, outCh]
    public required float[] ConvTBias { get; init; }
    public required QwenTtsCodecResidualUnitWeights[] Res { get; init; }
}
