
namespace OpenTail.Stingray.Audio.VoxtralRealtime;

/// <summary>
/// Real weight loader for Voxtral-Mini-4B-Realtime's audio tower + multimodal projector,
/// transcribed directly from `examples/audio.cpp/src/models/voxtral_realtime/audio_encoder.cpp`'s
/// `load_weights` (not guessed) -- see `docs/audio-review-progress.md`'s Voxtral section. The text
/// decoder half (a standard Mistral: 26 layers, GQA 32/8 heads, head_dim=128, RoPE theta=1e6,
/// sliding_window=8192) is NOT re-implemented here -- it is architecturally identical to Mistral
/// models this engine already runs via `ForwardPass`/`ModelCompatibility`'s existing allowlist, so
/// only the audio tower (a 32-layer, RoPE+MHA+SwiGLU transformer over mel features, NOT a
/// Whisper-style encoder) and the projector that maps its output into the text embedding space are
/// new work. Real defaults from the checkpoint's own `config.json` (`VoxtralRealtimeAudioConfig`
/// in `assets.h`): hidden=1280, intermediate=5120, 32 layers, 32 heads == 32 kv-heads (full MHA,
/// no GQA reduction in the audio tower despite the text decoder using GQA), head_dim=64,
/// mel_bins=128, sliding_window=750 (audio-tower-local causal window, unrelated to the text
/// decoder's own 8192), rope_theta=1e6, downsample_factor=4.
/// </summary>
public sealed class VoxtralAudioEncoderWeights
{
    public const int HiddenSize = 1280;
    public const int IntermediateSize = 5120;
    public const int NumLayers = 32;
    public const int NumHeads = 32;
    public const int NumKvHeads = 32; // full MHA -- audio tower does not use GQA
    public const int HeadDim = 64;
    public const int NumMelBins = 128;
    public const int SlidingWindow = 750;
    public const float RmsNormEps = 1e-5f;
    public const float RopeTheta = 1_000_000f;
    public const int DownsampleFactor = 4;
    public const int TextHiddenSize = 3072; // projector output width, matches the Mistral decoder's hidden size

    public float[] Conv1Weight { get; } // [hidden, melBins, 3]
    public float[] Conv1Bias { get; }
    public float[] Conv2Weight { get; } // [hidden, hidden, 3], stride 2
    public float[] Conv2Bias { get; }
    public VoxtralAudioLayerWeights[] Layers { get; } = new VoxtralAudioLayerWeights[NumLayers];
    public float[] NormWeight { get; } // final RMSNorm
    public float[] Projector1Weight { get; } // [TextHiddenSize, HiddenSize*DownsampleFactor], no bias
    public float[] Projector2Weight { get; } // [TextHiddenSize, TextHiddenSize], no bias

    public VoxtralAudioEncoderWeights(SafetensorsLoader loader)
    {
        Conv1Weight = loader.ReadF32("audio_tower.embedder.conv1.weight");
        Conv1Bias = loader.ReadF32("audio_tower.embedder.conv1.bias");
        Conv2Weight = loader.ReadF32("audio_tower.embedder.conv2.weight");
        Conv2Bias = loader.ReadF32("audio_tower.embedder.conv2.bias");

        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"audio_tower.layers.{i}";
            Layers[i] = new VoxtralAudioLayerWeights
            {
                AttnNorm = loader.ReadF32($"{p}.self_attn_layer_norm.weight"),
                QWeight = loader.ReadF32($"{p}.self_attn.q_proj.weight"),
                QBias = loader.ReadF32($"{p}.self_attn.q_proj.bias"),
                KWeight = loader.ReadF32($"{p}.self_attn.k_proj.weight"), // no bias -- real checkpoint has none
                VWeight = loader.ReadF32($"{p}.self_attn.v_proj.weight"),
                VBias = loader.ReadF32($"{p}.self_attn.v_proj.bias"),
                OWeight = loader.ReadF32($"{p}.self_attn.o_proj.weight"),
                OBias = loader.ReadF32($"{p}.self_attn.o_proj.bias"),
                FinalNorm = loader.ReadF32($"{p}.final_layer_norm.weight"),
                GateWeight = loader.ReadF32($"{p}.mlp.gate_proj.weight"),
                UpWeight = loader.ReadF32($"{p}.mlp.up_proj.weight"),
                DownWeight = loader.ReadF32($"{p}.mlp.down_proj.weight"),
                DownBias = loader.ReadF32($"{p}.mlp.down_proj.bias"),
            };
        }

        NormWeight = loader.ReadF32("audio_tower.norm.weight");
        Projector1Weight = loader.ReadF32("multi_modal_projector.linear_1.weight");
        Projector2Weight = loader.ReadF32("multi_modal_projector.linear_2.weight");
    }

    private VoxtralAudioEncoderWeights(Func<int, float[]> rand)
    {
        Conv1Weight = rand(HiddenSize * NumMelBins * 3);
        Conv1Bias = rand(HiddenSize);
        Conv2Weight = rand(HiddenSize * HiddenSize * 3);
        Conv2Bias = rand(HiddenSize);
        for (int i = 0; i < NumLayers; i++)
        {
            Layers[i] = new VoxtralAudioLayerWeights
            {
                AttnNorm = rand(HiddenSize),
                QWeight = rand(NumHeads * HeadDim * HiddenSize),
                QBias = rand(NumHeads * HeadDim),
                KWeight = rand(NumKvHeads * HeadDim * HiddenSize),
                VWeight = rand(NumKvHeads * HeadDim * HiddenSize),
                VBias = rand(NumKvHeads * HeadDim),
                OWeight = rand(HiddenSize * NumHeads * HeadDim),
                OBias = rand(HiddenSize),
                FinalNorm = rand(HiddenSize),
                GateWeight = rand(IntermediateSize * HiddenSize),
                UpWeight = rand(IntermediateSize * HiddenSize),
                DownWeight = rand(HiddenSize * IntermediateSize),
                DownBias = rand(HiddenSize),
            };
        }
        NormWeight = rand(HiddenSize);
        Projector1Weight = rand(TextHiddenSize * HiddenSize * DownsampleFactor);
        Projector2Weight = rand(TextHiddenSize * TextHiddenSize);
    }

    /// <summary>Test-only: builds weights from a caller-supplied random generator, for structural
    /// (shape/finiteness) tests that don't need the real ~9GB checkpoint.</summary>
    public static VoxtralAudioEncoderWeights CreateSynthetic(Func<int, float[]> randomArray) => new(randomArray);
}

public sealed class VoxtralAudioLayerWeights
{
    public float[] AttnNorm { get; set; } = [];
    public float[] QWeight { get; set; } = [];
    public float[] QBias { get; set; } = [];
    public float[] KWeight { get; set; } = [];
    public float[] VWeight { get; set; } = [];
    public float[] VBias { get; set; } = [];
    public float[] OWeight { get; set; } = [];
    public float[] OBias { get; set; } = [];
    public float[] FinalNorm { get; set; } = [];
    public float[] GateWeight { get; set; } = [];
    public float[] UpWeight { get; set; } = [];
    public float[] DownWeight { get; set; } = [];
    public float[] DownBias { get; set; } = [];
}
