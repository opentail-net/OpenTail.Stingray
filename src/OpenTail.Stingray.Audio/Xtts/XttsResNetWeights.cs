
namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 `hifigan_decoder.speaker_encoder` weights -- a real `ResNetSpeakerEncoder`
/// (`TTS/encoder/models/resnet.py`), construction confirmed from `HifiDecoder.__init__`:
/// `input_dim=64, proj_dim=512, log_input=True, use_torch_spec=True`, class defaults
/// `layers=[3,4,6,3], num_filters=[32,64,128,256], encoder_type="ASP"`.
/// </summary>
public sealed class XttsResNetWeights
{
    public static readonly int[] Layers = [3, 4, 6, 3];
    public static readonly int[] NumFilters = [32, 64, 128, 256];
    public const int InputDim = 64; // NumMels
    public const int ProjDim = 512;
    public const int OutmapSize = InputDim / 8; // 8 (3 stride-2 stages)
    public const int AttnInDim = NumFiltersLast * OutmapSize; // 2048
    private const int NumFiltersLast = 256;

    public float[] Conv1Weight { get; } // [32,1,3,3]
    public float[] Conv1Bias { get; } // real stem conv HAS a bias (unlike every SEBasicBlock conv, which are bias=False) -- easy to miss since it's the one exception
    public XttsBnWeights Bn1 { get; }

    public XttsResBlockWeights[][] ResLayers { get; } = new XttsResBlockWeights[4][];

    public float[] Attn0Weight { get; } // [128, 2048, 1] -> read as [128,2048]
    public float[] Attn0Bias { get; }
    public XttsBnWeights Attn2Bn { get; }
    public float[] Attn3Weight { get; } // [2048, 128, 1] -> [2048,128]
    public float[] Attn3Bias { get; }

    public float[] FcWeight { get; } // [512, 4096]
    public float[] FcBias { get; }

    public XttsResNetWeights(SafetensorsLoader loader, string prefix)
    {
        Conv1Weight = loader.ReadF32($"{prefix}.conv1.weight");
        Conv1Bias = loader.ReadF32($"{prefix}.conv1.bias");
        Bn1 = new XttsBnWeights(loader, $"{prefix}.bn1");

        for (int layerIdx = 0; layerIdx < 4; layerIdx++)
        {
            int numBlocks = Layers[layerIdx];
            ResLayers[layerIdx] = new XttsResBlockWeights[numBlocks];
            for (int b = 0; b < numBlocks; b++)
            {
                bool hasDownsample = b == 0 && layerIdx > 0; // stride/channel change only on first block of layers 2/3/4 (layer1 keeps 32->32, stride1, no downsample)
                int reduction = NumFilters[layerIdx] / 8;
                ResLayers[layerIdx][b] = new XttsResBlockWeights(loader, $"{prefix}.layer{layerIdx + 1}.{b}", reduction, hasDownsample);
            }
        }

        Attn0Weight = loader.ReadF32($"{prefix}.attention.0.weight");
        Attn0Bias = loader.ReadF32($"{prefix}.attention.0.bias");
        Attn2Bn = new XttsBnWeights(loader, $"{prefix}.attention.2");
        Attn3Weight = loader.ReadF32($"{prefix}.attention.3.weight");
        Attn3Bias = loader.ReadF32($"{prefix}.attention.3.bias");

        FcWeight = loader.ReadF32($"{prefix}.fc.weight");
        FcBias = loader.ReadF32($"{prefix}.fc.bias");
    }
}

public sealed class XttsBnWeights
{
    public float[] Weight { get; }
    public float[] Bias { get; }
    public float[] RunningMean { get; }
    public float[] RunningVar { get; }

    public XttsBnWeights(SafetensorsLoader loader, string prefix)
    {
        Weight = loader.ReadF32($"{prefix}.weight");
        Bias = loader.ReadF32($"{prefix}.bias");
        RunningMean = loader.ReadF32($"{prefix}.running_mean");
        RunningVar = loader.ReadF32($"{prefix}.running_var");
    }
}

/// <summary>Real `SEBasicBlock`: conv1(k3,pad1,stride) -> relu -> bn1 -> conv2(k3,pad1,stride1) -> bn2 -> SE -> (+downsample(x) or +x) -> relu.</summary>
public sealed class XttsResBlockWeights
{
    public float[] Conv1Weight { get; }
    public XttsBnWeights Bn1 { get; }
    public float[] Conv2Weight { get; }
    public XttsBnWeights Bn2 { get; }
    public float[] SeFc0Weight { get; }
    public float[] SeFc0Bias { get; }
    public float[] SeFc2Weight { get; }
    public float[] SeFc2Bias { get; }
    public int SeReducedCh { get; }

    public float[]? DownsampleConvWeight { get; }
    public XttsBnWeights? DownsampleBn { get; }

    public XttsResBlockWeights(SafetensorsLoader loader, string prefix, int reduction, bool hasDownsample)
    {
        Conv1Weight = loader.ReadF32($"{prefix}.conv1.weight");
        Bn1 = new XttsBnWeights(loader, $"{prefix}.bn1");
        Conv2Weight = loader.ReadF32($"{prefix}.conv2.weight");
        Bn2 = new XttsBnWeights(loader, $"{prefix}.bn2");

        SeFc0Weight = loader.ReadF32($"{prefix}.se.fc.0.weight");
        SeFc0Bias = loader.ReadF32($"{prefix}.se.fc.0.bias");
        SeFc2Weight = loader.ReadF32($"{prefix}.se.fc.2.weight");
        SeFc2Bias = loader.ReadF32($"{prefix}.se.fc.2.bias");
        SeReducedCh = reduction;

        if (hasDownsample)
        {
            DownsampleConvWeight = loader.ReadF32($"{prefix}.downsample.0.weight");
            DownsampleBn = new XttsBnWeights(loader, $"{prefix}.downsample.1");
        }
    }
}
