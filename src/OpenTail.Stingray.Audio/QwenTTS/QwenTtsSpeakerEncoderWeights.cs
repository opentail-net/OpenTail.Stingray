using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Real weights for the Qwen3-TTS ECAPA-TDNN-style speaker encoder (`spk_enc.*`, stored in the
/// Talker GGUF). Real structure confirmed via real tensor shapes: `conv0` (TDNN, mel=128-&gt;512,
/// k=5) -&gt; 3x <see cref="QwenTtsSpeakerEncoderBlockWeights"/> (SE-Res2Net, dilations 2/3/4) -&gt;
/// `mfa` (Conv1d 1536-&gt;1536, k=1, over the channel-concat of the 3 block outputs) -&gt;
/// <see cref="QwenTtsSpeakerEncoderAspWeights"/> (Attentive Statistics Pooling) -&gt; `fc`
/// (Linear 3072-&gt;enc_dim). Real per-source class names: `TimeDelayNetBlock`, `Res2NetBlock`,
/// `SqueezeExcitation`, `SqueezeExcitationRes2NetBlock`, `AttentiveStatisticsPooling`,
/// `SpeakerEncoder` (`qwen_tts/core/models/modeling_qwen3_tts.py`).
/// </summary>
public sealed class QwenTtsSpeakerEncoderWeights
{
    public const int MelDim = 128;
    public const int Channels = 512;
    public const int MfaOutDim = 1536; // 512 * 3 (concat of the 3 block outputs)
    public const int AttentionChannels = 128;
    public const int Res2NetScale = 8;
    public const int Res2NetBranchChannels = 64; // Channels / Res2NetScale

    public float[] Conv0Weight { get; } // [512,128,5]
    public float[] Conv0Bias { get; }
    public QwenTtsSpeakerEncoderBlockWeights[] Blocks { get; } = new QwenTtsSpeakerEncoderBlockWeights[3];
    public float[] MfaWeight { get; } // [1536,1536,1]
    public float[] MfaBias { get; }
    public QwenTtsSpeakerEncoderAspWeights Asp { get; }
    public float[] FcWeight { get; } // [enc_dim,3072,1]
    public float[] FcBias { get; }
    public int EncDim { get; }

    public QwenTtsSpeakerEncoderWeights(GgufModel model)
    {
        Conv0Weight = GetF32(model, "spk_enc.conv0.weight");
        Conv0Bias = GetF32(model, "spk_enc.conv0.bias");

        int[] dilations = [2, 3, 4];
        for (int b = 0; b < 3; b++)
        {
            string p = $"spk_enc.blk.{b + 1}";
            Blocks[b] = new QwenTtsSpeakerEncoderBlockWeights
            {
                Dilation = dilations[b],
                Tdnn1Weight = GetF32(model, $"{p}.tdnn1.weight"),
                Tdnn1Bias = GetF32(model, $"{p}.tdnn1.bias"),
                Res2NetWeight = new float[7][],
                Res2NetBias = new float[7][],
                Tdnn2Weight = GetF32(model, $"{p}.tdnn2.weight"),
                Tdnn2Bias = GetF32(model, $"{p}.tdnn2.bias"),
                SeConv1Weight = GetF32(model, $"{p}.se.conv1.weight"),
                SeConv1Bias = GetF32(model, $"{p}.se.conv1.bias"),
                SeConv2Weight = GetF32(model, $"{p}.se.conv2.weight"),
                SeConv2Bias = GetF32(model, $"{p}.se.conv2.bias"),
            };
            for (int r = 0; r < 7; r++)
            {
                Blocks[b].Res2NetWeight[r] = GetF32(model, $"{p}.res2net.{r}.weight");
                Blocks[b].Res2NetBias[r] = GetF32(model, $"{p}.res2net.{r}.bias");
            }
        }

        MfaWeight = GetF32(model, "spk_enc.mfa.weight");
        MfaBias = GetF32(model, "spk_enc.mfa.bias");

        Asp = new QwenTtsSpeakerEncoderAspWeights
        {
            TdnnWeight = GetF32(model, "spk_enc.asp.tdnn.weight"),
            TdnnBias = GetF32(model, "spk_enc.asp.tdnn.bias"),
            ConvWeight = GetF32(model, "spk_enc.asp.conv.weight"),
            ConvBias = GetF32(model, "spk_enc.asp.conv.bias"),
        };

        FcWeight = GetF32(model, "spk_enc.fc.weight");
        FcBias = GetF32(model, "spk_enc.fc.bias");
        EncDim = FcBias.Length;
    }

    private static float[] GetF32(GgufModel model, string name)
    {
        var info = model.FindTensor(name) ?? throw new InvalidDataException($"QwenTTS speaker encoder GGUF missing required tensor '{name}'.");
        var bytes = model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }
}

public sealed class QwenTtsSpeakerEncoderBlockWeights
{
    public required int Dilation { get; init; }
    public required float[] Tdnn1Weight { get; init; } // [512,512,1]
    public required float[] Tdnn1Bias { get; init; }
    public required float[][] Res2NetWeight { get; init; } // 7x [64,64,3]
    public required float[][] Res2NetBias { get; init; }
    public required float[] Tdnn2Weight { get; init; } // [512,512,1]
    public required float[] Tdnn2Bias { get; init; }
    public required float[] SeConv1Weight { get; init; } // [128,512,1]
    public required float[] SeConv1Bias { get; init; }
    public required float[] SeConv2Weight { get; init; } // [512,128,1]
    public required float[] SeConv2Bias { get; init; }
}

public sealed class QwenTtsSpeakerEncoderAspWeights
{
    public required float[] TdnnWeight { get; init; } // [128,4608,1]
    public required float[] TdnnBias { get; init; }
    public required float[] ConvWeight { get; init; } // [1536,128,1]
    public required float[] ConvBias { get; init; }
}
