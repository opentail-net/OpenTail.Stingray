namespace OpenTail.Stingray.Audio.NeuTts;

public sealed class NeuTtsResnetBlockWeights
{
    public required float[] Norm1Weight { get; init; }
    public required float[] Norm1Bias { get; init; }
    public required float[] Conv1Weight { get; init; } // [C,C,3]
    public required float[] Conv1Bias { get; init; }
    public required float[] Norm2Weight { get; init; }
    public required float[] Norm2Bias { get; init; }
    public required float[] Conv2Weight { get; init; } // [C,C,3]
    public required float[] Conv2Bias { get; init; }
}

public sealed class NeuTtsCodecTransformerLayerWeights
{
    public required float[] AttnNormWeight { get; init; }
    public required float[] QWeight { get; init; }
    public required float[] KWeight { get; init; }
    public required float[] VWeight { get; init; }
    public required float[] OutWeight { get; init; }
    public required float[] FfnNormWeight { get; init; }
    public required float[] Fc1Weight { get; init; }
    public required float[] Fc2Weight { get; init; }
}

/// <summary>
/// Real weights for NeuTTS's FSQ acoustic-decoder codec, ported from the shared
/// `examples/audio.cpp/src/framework/codecs/fsq_audio_codec_runtime.cpp` (not guessed). Real
/// config, confirmed from the checkpoint's own embedded `neucodec_config.json` (the acoustic
/// decoder's own top-level fields, NOT `semantic_model_config` -- that's the encode-only
/// wav2vec2-bert branch, not needed for decode): `hidden_size=1024, intermediate_size=4096,
/// num_hidden_layers=12, num_attention_heads=16, head_dim=64` (plain MHA -- `num_key_value_heads`
/// equals `num_attention_heads` in this checkpoint, no GQA), `quantization_dim=2048,
/// quantization_levels=[4,4,4,4,4,4,4,4]` (8 dims, level 4 each -&gt; codebook_size=4^8=65536,
/// exactly matching NeuTTS's real `&lt;|speech_0|&gt;..&lt;|speech_65535|&gt;` vocab range),
/// `rope_theta=10000, rms_norm_eps=1e-6`. Real, NOT-in-JSON struct defaults kept from the
/// reference's own `FsqAudioCodecConfig`: `hop_length=480` (the reference's own comment: "the
/// decoder is constructed by the reference code with hop_length=480", not read from config),
/// `prior_blocks=2, post_blocks=2`.
/// </summary>
public sealed class NeuTtsAudioDecoderWeights
{
    public const int HiddenSize = 1024;
    public const int IntermediateSize = 4096;
    public const int NumLayers = 12;
    public const int NumHeads = 16;
    public const int HeadDim = 64;
    public const int QuantizationDim = 2048;
    public static readonly int[] QuantizationLevels = [4, 4, 4, 4, 4, 4, 4, 4];
    public const float RopeTheta = 10000f;
    public const float RmsNormEps = 1e-6f;
    public const int HopLength = 480;
    public const int PriorBlocks = 2;
    public const int PostBlocks = 2;
    public const int SampleRate = 24000;
    public const int NFft = HopLength * 4;

    public required float[] QuantizerProjectOutWeight { get; init; } // [QuantizationDim, NumLevels]
    public required float[] QuantizerProjectOutBias { get; init; }
    public required float[] AcousticFcWeight { get; init; } // [HiddenSize, QuantizationDim]
    public required float[] AcousticFcBias { get; init; }
    public required float[] EmbedConvWeight { get; init; } // [HiddenSize, HiddenSize, 7]
    public required float[] EmbedConvBias { get; init; }
    public required NeuTtsResnetBlockWeights[] PriorNet { get; init; }
    public required NeuTtsCodecTransformerLayerWeights[] Layers { get; init; }
    public required NeuTtsResnetBlockWeights[] PostNet { get; init; }
    public required float[] FinalNormWeight { get; init; }
    public required float[] FinalNormBias { get; init; }
    public required float[] IstftHeadWeight { get; init; } // [HopLength*4+2, HiddenSize]
    public required float[] IstftHeadBias { get; init; }

    public static NeuTtsAudioDecoderWeights Load(Func<string, float[]> get)
    {
        NeuTtsResnetBlockWeights LoadResnet(string p) => new()
        {
            Norm1Weight = get($"{p}.norm1.weight"),
            Norm1Bias = get($"{p}.norm1.bias"),
            Conv1Weight = get($"{p}.conv1.weight"),
            Conv1Bias = get($"{p}.conv1.bias"),
            Norm2Weight = get($"{p}.norm2.weight"),
            Norm2Bias = get($"{p}.norm2.bias"),
            Conv2Weight = get($"{p}.conv2.weight"),
            Conv2Bias = get($"{p}.conv2.bias"),
        };

        var priorNet = new NeuTtsResnetBlockWeights[PriorBlocks];
        for (int i = 0; i < PriorBlocks; i++) priorNet[i] = LoadResnet($"codec/acoustic_decoder.prior_net.{i}");

        var postNet = new NeuTtsResnetBlockWeights[PostBlocks];
        for (int i = 0; i < PostBlocks; i++) postNet[i] = LoadResnet($"codec/acoustic_decoder.post_net.{i}");

        var layers = new NeuTtsCodecTransformerLayerWeights[NumLayers];
        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"codec/acoustic_decoder.layers.{i}";
            layers[i] = new NeuTtsCodecTransformerLayerWeights
            {
                AttnNormWeight = get($"{p}.input_layernorm.weight"),
                QWeight = get($"{p}.self_attn.q_proj.weight"),
                KWeight = get($"{p}.self_attn.k_proj.weight"),
                VWeight = get($"{p}.self_attn.v_proj.weight"),
                OutWeight = get($"{p}.self_attn.o_proj.weight"),
                FfnNormWeight = get($"{p}.post_attention_layernorm.weight"),
                Fc1Weight = get($"{p}.mlp.fc1.weight"),
                Fc2Weight = get($"{p}.mlp.fc2.weight"),
            };
        }

        return new NeuTtsAudioDecoderWeights
        {
            QuantizerProjectOutWeight = get("codec/quantizer.project_out.weight"),
            QuantizerProjectOutBias = get("codec/quantizer.project_out.bias"),
            AcousticFcWeight = get("codec/acoustic_decoder.fc.weight"),
            AcousticFcBias = get("codec/acoustic_decoder.fc.bias"),
            EmbedConvWeight = get("codec/acoustic_decoder.embed.weight"),
            EmbedConvBias = get("codec/acoustic_decoder.embed.bias"),
            PriorNet = priorNet,
            Layers = layers,
            PostNet = postNet,
            FinalNormWeight = get("codec/acoustic_decoder.norm.weight"),
            FinalNormBias = get("codec/acoustic_decoder.norm.bias"),
            IstftHeadWeight = get("codec/acoustic_decoder.head.linear.weight"),
            IstftHeadBias = get("codec/acoustic_decoder.head.linear.bias"),
        };
    }
}
