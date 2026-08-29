
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
    public float[] PreConvBias { get; }
    public QwenTtsCodecDacBlockWeights[] Blocks { get; } = new QwenTtsCodecDacBlockWeights[4];
    public float[] FinalSnakeAlpha { get; }
    public float[] FinalSnakeBeta { get; }
    public float[] FinalConvWeight { get; }
    public float[] FinalConvBias { get; }

    public QwenTtsCodecDacWeights(GgufModel model)
    {
        PreConvWeight = GetF32(model, "tok_dec.dec.0.conv.weight");
        PreConvBias = GetF32(model, "tok_dec.dec.0.conv.bias");

        for (int b = 0; b < 4; b++)
        {
            string p = $"tok_dec.dec.{b + 1}";
            var res = new QwenTtsCodecResidualUnitWeights[3];
            for (int r = 0; r < 3; r++)
            {
                string rp = $"{p}.res.{r}";
                res[r] = new QwenTtsCodecResidualUnitWeights
                {
                    Act1Alpha = GetF32(model, $"{rp}.act1.alpha"),
                    Act1Beta = GetF32(model, $"{rp}.act1.beta"),
                    Conv1Weight = GetF32(model, $"{rp}.conv1.weight"),
                    Conv1Bias = GetF32(model, $"{rp}.conv1.bias"),
                    Act2Alpha = GetF32(model, $"{rp}.act2.alpha"),
                    Act2Beta = GetF32(model, $"{rp}.act2.beta"),
                    Conv2Weight = GetF32(model, $"{rp}.conv2.weight"),
                    Conv2Bias = GetF32(model, $"{rp}.conv2.bias"),
                };
            }
            Blocks[b] = new QwenTtsCodecDacBlockWeights
            {
                SnakeAlpha = GetF32(model, $"{p}.snake.alpha"),
                SnakeBeta = GetF32(model, $"{p}.snake.beta"),
                ConvTWeight = GetF32(model, $"{p}.conv_t.weight"),
                ConvTBias = GetF32(model, $"{p}.conv_t.bias"),
                Res = res,
            };
        }

        FinalSnakeAlpha = GetF32(model, "tok_dec.dec.5.snake.alpha");
        FinalSnakeBeta = GetF32(model, "tok_dec.dec.5.snake.beta");
        FinalConvWeight = GetF32(model, "tok_dec.dec.6.conv.weight");
        FinalConvBias = GetF32(model, "tok_dec.dec.6.conv.bias");
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
    public required float[] Conv1Bias { get; init; }
    public required float[] Act2Alpha { get; init; }
    public required float[] Act2Beta { get; init; }
    public required float[] Conv2Weight { get; init; } // causal, k=1
    public required float[] Conv2Bias { get; init; }
}

public sealed class QwenTtsCodecDacBlockWeights
{
    public required float[] SnakeAlpha { get; init; }
    public required float[] SnakeBeta { get; init; }
    public required float[] ConvTWeight { get; init; } // causal ConvTranspose1d, kernel=2*rate
    public required float[] ConvTBias { get; init; }
    public required QwenTtsCodecResidualUnitWeights[] Res { get; init; }
}
