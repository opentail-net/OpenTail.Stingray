
namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// Real F5-TTS DiT (Flow-Matching Diffusion Transformer) weights, loaded directly from the
/// `.safetensors` checkpoint's tensors (raw torch state_dict layout -- unlike the ONNX exports
/// used by the VITS-family pipelines this session, safetensors tensors keep their native torch
/// `nn.Linear.weight` layout of `[outFeatures, inFeatures]` row-major with NO transpose quirk, and
/// no weight_norm-fusion anonymization -- every name below is the tensor's real, clean name).
///
/// Config confirmed by loading these exact weights into the real PyTorch `f5_tts.model.backbones.
/// dit.DiT` class (examples/f5-tts-py) with `strict=True` and getting zero missing/unexpected
/// keys (see scratch-llamacpp-ref/f5_golden_dit.py): dim=1024, depth=22, heads=16, dim_head=64,
/// ff_mult=2 (NOT the DiT default of 4 -- confirmed via `ff.ff.0.0.weight`'s [2048,1024] shape),
/// mel_dim=100, text_num_embeds=2545, text_dim=512, conv_layers=4 (ConvNeXtV2 text blocks),
/// qk_norm=None (no q_norm/k_norm tensors present), long_skip_connection=False (no tensor
/// present), attn_mask_enabled=False (an inference-time config flag, not a weight).
/// </summary>
public sealed class F5TtsWeights : IDisposable
{
    public SafetensorsLoader Loader { get; }

    public const int HiddenDim = 1024;
    public const int NumHeads = 16;
    public const int HeadDim = HiddenDim / NumHeads; // 64
    public const int NumLayers = 22;
    public const int FfnDim = 2048; // ff_mult=2, NOT the more common 4
    public const int MelDim = 100;
    public const int TextDim = 512;
    public const int TextFfnDim = 1024; // ConvNeXtV2Block intermediate_dim = text_dim * conv_mult(2)
    public const int NumTextBlocks = 4;
    public const int TimeFreqDim = 256; // SinusPositionEmbedding(256) inside TimestepEmbedding
    public const int VocabSize = 2545; // text_num_embeds (real embedding table is [2546,512]: +1 for the "0 = filler" token)

    // --- text_embed.* ---
    public float[] TextEmbedWeight { get; } // [2546, 512]
    public F5TextBlockWeights[] TextBlocks { get; } = new F5TextBlockWeights[NumTextBlocks];

    // --- time_embed.* ---
    public float[] TimeMlp0Weight { get; }
    public float[] TimeMlp0Bias { get; }
    public float[] TimeMlp2Weight { get; }
    public float[] TimeMlp2Bias { get; }

    // --- input_embed.* ---
    public float[] InputProjWeight { get; } // [1024, 712]
    public float[] InputProjBias { get; }
    public float[] ConvPos1Weight { get; } // [1024, 64, 31] (groups=16 -> 1024/16=64 in-channels per group)
    public float[] ConvPos1Bias { get; }
    public float[] ConvPos2Weight { get; }
    public float[] ConvPos2Bias { get; }

    // --- rotary_embed.* ---
    public float[] RotaryInvFreq { get; } // [32] = HeadDim/2

    // --- transformer_blocks.* ---
    public F5DiTBlockWeights[] Blocks { get; } = new F5DiTBlockWeights[NumLayers];

    // --- norm_out / proj_out ---
    public float[] NormOutLinearWeight { get; } // [2048, 1024]
    public float[] NormOutLinearBias { get; }
    public float[] ProjOutWeight { get; } // [100, 1024]
    public float[] ProjOutBias { get; }

    public F5TtsWeights(string safetensorsPath)
    {
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException($"F5-TTS safetensors model file not found: {safetensorsPath}");

        Loader = SafetensorsLoader.Open(safetensorsPath);

        TextEmbedWeight = Read("text_embed.text_embed.weight");
        for (int i = 0; i < NumTextBlocks; i++)
            TextBlocks[i] = new F5TextBlockWeights(this, i);

        TimeMlp0Weight = Read("time_embed.time_mlp.0.weight");
        TimeMlp0Bias = Read("time_embed.time_mlp.0.bias");
        TimeMlp2Weight = Read("time_embed.time_mlp.2.weight");
        TimeMlp2Bias = Read("time_embed.time_mlp.2.bias");

        InputProjWeight = Read("input_embed.proj.weight");
        InputProjBias = Read("input_embed.proj.bias");
        ConvPos1Weight = Read("input_embed.conv_pos_embed.conv1d.0.weight");
        ConvPos1Bias = Read("input_embed.conv_pos_embed.conv1d.0.bias");
        ConvPos2Weight = Read("input_embed.conv_pos_embed.conv1d.2.weight");
        ConvPos2Bias = Read("input_embed.conv_pos_embed.conv1d.2.bias");

        RotaryInvFreq = Read("rotary_embed.inv_freq");

        for (int i = 0; i < NumLayers; i++)
            Blocks[i] = new F5DiTBlockWeights(this, i);

        NormOutLinearWeight = Read("norm_out.linear.weight");
        NormOutLinearBias = Read("norm_out.linear.bias");
        ProjOutWeight = Read("proj_out.weight");
        ProjOutBias = Read("proj_out.bias");
    }

    public float[] Read(string name) => Loader.ReadF32($"ema_model.transformer.{name}");

    public void Dispose() => Loader.Dispose();
}

/// <summary>One `text_embed.text_blocks.{i}` ConvNeXtV2Block.</summary>
public sealed class F5TextBlockWeights
{
    public float[] DwConvWeight { get; } // [512, 1, 7] depthwise
    public float[] DwConvBias { get; }
    public float[] NormWeight { get; } // LayerNorm gamma [512]
    public float[] NormBias { get; }
    public float[] PwConv1Weight { get; } // [1024, 512]
    public float[] PwConv1Bias { get; }
    public float[] GrnGamma { get; } // [1024]
    public float[] GrnBeta { get; }
    public float[] PwConv2Weight { get; } // [512, 1024]
    public float[] PwConv2Bias { get; }

    public F5TextBlockWeights(F5TtsWeights w, int i)
    {
        string p = $"text_embed.text_blocks.{i}";
        DwConvWeight = w.Read($"{p}.dwconv.weight");
        DwConvBias = w.Read($"{p}.dwconv.bias");
        NormWeight = w.Read($"{p}.norm.weight");
        NormBias = w.Read($"{p}.norm.bias");
        PwConv1Weight = w.Read($"{p}.pwconv1.weight");
        PwConv1Bias = w.Read($"{p}.pwconv1.bias");
        GrnGamma = w.Read($"{p}.grn.gamma");
        GrnBeta = w.Read($"{p}.grn.beta");
        PwConv2Weight = w.Read($"{p}.pwconv2.weight");
        PwConv2Bias = w.Read($"{p}.pwconv2.bias");
    }
}

/// <summary>One `transformer_blocks.{i}` DiTBlock.</summary>
public sealed class F5DiTBlockWeights
{
    public float[] AttnNormLinearWeight { get; } // [6144, 1024]
    public float[] AttnNormLinearBias { get; }
    public float[] ToQWeight { get; }
    public float[] ToQBias { get; }
    public float[] ToKWeight { get; }
    public float[] ToKBias { get; }
    public float[] ToVWeight { get; }
    public float[] ToVBias { get; }
    public float[] ToOutWeight { get; }
    public float[] ToOutBias { get; }
    public float[] FfInWeight { get; } // [2048, 1024]
    public float[] FfInBias { get; }
    public float[] FfOutWeight { get; } // [1024, 2048]
    public float[] FfOutBias { get; }

    public F5DiTBlockWeights(F5TtsWeights w, int i)
    {
        string p = $"transformer_blocks.{i}";
        AttnNormLinearWeight = w.Read($"{p}.attn_norm.linear.weight");
        AttnNormLinearBias = w.Read($"{p}.attn_norm.linear.bias");
        ToQWeight = w.Read($"{p}.attn.to_q.weight");
        ToQBias = w.Read($"{p}.attn.to_q.bias");
        ToKWeight = w.Read($"{p}.attn.to_k.weight");
        ToKBias = w.Read($"{p}.attn.to_k.bias");
        ToVWeight = w.Read($"{p}.attn.to_v.weight");
        ToVBias = w.Read($"{p}.attn.to_v.bias");
        ToOutWeight = w.Read($"{p}.attn.to_out.0.weight");
        ToOutBias = w.Read($"{p}.attn.to_out.0.bias");
        FfInWeight = w.Read($"{p}.ff.ff.0.0.weight");
        FfInBias = w.Read($"{p}.ff.ff.0.0.bias");
        FfOutWeight = w.Read($"{p}.ff.ff.2.weight");
        FfOutBias = w.Read($"{p}.ff.ff.2.bias");
    }
}
