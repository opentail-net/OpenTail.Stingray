
namespace OpenTail.Stingray.Audio.VoxtralRealtime;

/// <summary>
/// Real weight loader for Voxtral Realtime's text decoder (`language_model.model.*` in
/// `model.safetensors`), transcribed directly from
/// `examples/audio.cpp/src/models/voxtral_realtime/text_decoder.cpp`'s `FullContextGraph` (not
/// guessed). A real Mistral GQA decoder (26 layers, 32/8 heads, head_dim=128, hidden=3072,
/// RoPE theta=1e6) PLUS a real AdaLN-Zero-style delay-conditioning gate per layer
/// (`ada_rms_norm.linear{1,2}.weight`, no bias, confirmed via a real tensor dump) that this
/// checkpoint's config alone does not reveal -- see `docs/audio-review-progress.md`'s Voxtral
/// section for the correction this represents (an earlier pass in this session wrongly assumed
/// the decoder needed zero new code). Tied embeddings (`config.json`'s
/// `text_config.tie_word_embeddings=true`, confirmed by the real tensor list having no separate
/// `lm_head.weight`).
/// </summary>
public sealed class VoxtralTextDecoderWeights
{
    public const int HiddenSize = 3072;
    public const int IntermediateSize = 9216;
    public const int NumLayers = 26;
    public const int NumHeads = 32;
    public const int NumKvHeads = 8;
    public const int HeadDim = 128;
    public const float RmsNormEps = 1e-5f;
    public const float RopeTheta = 1_000_000f;
    public const int VocabSize = 131072;
    public const int AdaHiddenDim = 32;

    public float[] EmbedTokensWeight { get; } // [131072, 3072] -- also used as the tied lm_head
    public VoxtralTextLayerWeights[] Layers { get; } = new VoxtralTextLayerWeights[NumLayers];
    public float[] NormWeight { get; }

    public VoxtralTextDecoderWeights(SafetensorsLoader loader)
    {
        EmbedTokensWeight = loader.ReadF32("language_model.model.embed_tokens.weight");
        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"language_model.model.layers.{i}";
            Layers[i] = new VoxtralTextLayerWeights
            {
                InputNorm = loader.ReadF32($"{p}.input_layernorm.weight"),
                QWeight = loader.ReadF32($"{p}.self_attn.q_proj.weight"),
                KWeight = loader.ReadF32($"{p}.self_attn.k_proj.weight"),
                VWeight = loader.ReadF32($"{p}.self_attn.v_proj.weight"),
                OWeight = loader.ReadF32($"{p}.self_attn.o_proj.weight"),
                PostNorm = loader.ReadF32($"{p}.post_attention_layernorm.weight"),
                GateWeight = loader.ReadF32($"{p}.mlp.gate_proj.weight"),
                UpWeight = loader.ReadF32($"{p}.mlp.up_proj.weight"),
                DownWeight = loader.ReadF32($"{p}.mlp.down_proj.weight"),
                Ada1Weight = loader.ReadF32($"{p}.ada_rms_norm.linear1.weight"),
                Ada2Weight = loader.ReadF32($"{p}.ada_rms_norm.linear2.weight"),
            };
        }
        NormWeight = loader.ReadF32("language_model.model.norm.weight");
    }
}

public sealed class VoxtralTextLayerWeights
{
    public float[] InputNorm { get; set; } = [];
    public float[] QWeight { get; set; } = [];
    public float[] KWeight { get; set; } = [];
    public float[] VWeight { get; set; } = [];
    public float[] OWeight { get; set; } = [];
    public float[] PostNorm { get; set; } = [];
    public float[] GateWeight { get; set; } = [];
    public float[] UpWeight { get; set; } = [];
    public float[] DownWeight { get; set; } = [];
    public float[] Ada1Weight { get; set; } = []; // [32, 3072], no bias
    public float[] Ada2Weight { get; set; } = []; // [3072, 32], no bias
}
