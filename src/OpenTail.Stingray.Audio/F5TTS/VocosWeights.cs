
namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// Real Vocos vocoder weights (charactr/vocos-mel-24khz), loaded from `models/
/// vocos-mel-24khz.safetensors` (converted from the repo's `pytorch_model.bin` -- a pickled
/// torch state_dict, not directly loadable by this project's `SafetensorsLoader`; converted
/// via `torch.load` + `safetensors.torch.save_file`, see scratch-llamacpp-ref/
/// vocos_golden_decode.py's header for the exact key names). A `.gguf` copy also exists at
/// `models/vocos-mel-24khz.gguf` (written with the official `gguf` PyPI package) for callers
/// that prefer GGUF; this loader uses the safetensors copy since the project's
/// `SafetensorsLoader` already handles this checkpoint's flat `[out,in,kernel]` torch-native
/// tensor layout directly with no transpose/anonymization quirks.
///
/// Config confirmed via `models/vocos-mel-24khz-config.yaml` + real key/shape inspection:
/// `VocosBackbone(input_channels=100, dim=512, intermediate_dim=1536, num_layers=8)` (no
/// AdaLayerNorm conditioning -- this checkpoint has no `adanorm` embeddings), `ISTFTHead(dim=512,
/// n_fft=1024, hop_length=256, padding="center")`.
/// </summary>
public sealed class VocosWeights : IDisposable
{
    public SafetensorsLoader Loader { get; }

    public const int MelDim = 100;
    public const int HiddenDim = 512;
    public const int IntermediateDim = 1536;
    public const int NumLayers = 8;
    public const int NFft = 1024;
    public const int HopLength = 256;
    public const int NumBins = NFft / 2 + 1; // 513

    public float[] EmbedWeight { get; } // [512,100,7]
    public float[] EmbedBias { get; }
    public float[] NormWeight { get; } // plain LayerNorm (no adanorm)
    public float[] NormBias { get; }
    public VocosConvNeXtBlockWeights[] Blocks { get; } = new VocosConvNeXtBlockWeights[NumLayers];
    public float[] FinalNormWeight { get; }
    public float[] FinalNormBias { get; }
    public float[] HeadOutWeight { get; } // [1026,512]
    public float[] HeadOutBias { get; }

    public VocosWeights(string safetensorsPath)
    {
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException($"Vocos safetensors model file not found: {safetensorsPath}");

        Loader = SafetensorsLoader.Open(safetensorsPath);

        EmbedWeight = Read("backbone.embed.weight");
        EmbedBias = Read("backbone.embed.bias");
        NormWeight = Read("backbone.norm.weight");
        NormBias = Read("backbone.norm.bias");
        for (int i = 0; i < NumLayers; i++)
            Blocks[i] = new VocosConvNeXtBlockWeights(this, i);
        FinalNormWeight = Read("backbone.final_layer_norm.weight");
        FinalNormBias = Read("backbone.final_layer_norm.bias");
        HeadOutWeight = Read("head.out.weight");
        HeadOutBias = Read("head.out.bias");
    }

    public float[] Read(string name) => Loader.ReadF32(name);

    public void Dispose() => Loader.Dispose();
}

/// <summary>One `backbone.convnext.{i}` ConvNeXtBlock (no GRN -- Vocos uses a simpler learned per-channel `gamma` layer-scale instead, unlike F5-TTS's ConvNeXtV2Block/GRN).</summary>
public sealed class VocosConvNeXtBlockWeights
{
    public float[] DwConvWeight { get; } // [512,1,7]
    public float[] DwConvBias { get; }
    public float[] NormWeight { get; }
    public float[] NormBias { get; }
    public float[] PwConv1Weight { get; } // [1536,512]
    public float[] PwConv1Bias { get; }
    public float[] PwConv2Weight { get; } // [512,1536]
    public float[] PwConv2Bias { get; }
    public float[] Gamma { get; } // [512]

    public VocosConvNeXtBlockWeights(VocosWeights w, int i)
    {
        string p = $"backbone.convnext.{i}";
        DwConvWeight = w.Read($"{p}.dwconv.weight");
        DwConvBias = w.Read($"{p}.dwconv.bias");
        NormWeight = w.Read($"{p}.norm.weight");
        NormBias = w.Read($"{p}.norm.bias");
        PwConv1Weight = w.Read($"{p}.pwconv1.weight");
        PwConv1Bias = w.Read($"{p}.pwconv1.bias");
        PwConv2Weight = w.Read($"{p}.pwconv2.weight");
        PwConv2Bias = w.Read($"{p}.pwconv2.bias");
        Gamma = w.Read($"{p}.gamma");
    }
}
