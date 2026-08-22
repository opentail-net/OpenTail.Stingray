using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.Parler;

/// <summary>
/// Real Parler-TTS decoder weight loader (MusicGen-style causal decoder), loaded from the real
/// `parler-tts-mini-v1` checkpoint's `model.safetensors` (`models/parler-tts-mini-v1.safetensors`,
/// already downloaded this session for the T5 encoder -- same file, no new download).
///
/// <para>Real config, confirmed from the checkpoint's own `config.json`'s `decoder` block (not
/// guessed): `hidden_size=1024`, `num_hidden_layers=24`, `num_attention_heads=16`,
/// `num_key_value_heads=16` (full MHA, NOT GQA), `num_cross_attention_key_value_heads=16` (cross-
/// attn also full MHA), `ffn_dim=4096`, `activation_function=gelu` (plain, NOT gated),
/// `rope_embeddings=false` (real sinusoidal positional embeddings, precomputed and stored
/// directly as a real tensor -- `embed_positions.weights` -- not a formula to reimplement),
/// `num_codebooks=9`, `vocab_size=1088` (note: the INPUT embedding tables are sized
/// `vocab_size+1=1089` -- a real, intentional discrepancy confirmed from the real source's own
/// "TODO... +1 for pad token id... too late to change now" comment, not a bug), `bias=False` on
/// every Linear (self-attn/cross-attn/fc1/fc2).</para>
///
/// <para>Real tensor names (`decoder.model.decoder.*` prefix for the trunk,
/// `decoder.lm_heads.{0..8}.weight` for the 9 separate output heads,
/// `decoder.model.decoder.embed_tokens.{0..8}.weight` for the 9 separate input embedding
/// tables) -- confirmed via the real safetensors header, not assumed from the earlier session's
/// GGUF-derived names (which used a DIFFERENT prefix convention, `decoder.layers.*` without the
/// doubled `model.decoder`).</para>
/// </summary>
public sealed class ParlerDecoderWeights
{
    public const int HiddenDim = 1024;
    public const int NumLayers = 24;
    public const int NumHeads = 16;
    public const int FfnDim = 4096;
    public const int NumCodebooks = 9;
    public const int InputVocabSize = 1089; // embed_tokens: vocab_size + 1
    public const int OutputVocabSize = 1088; // lm_heads: vocab_size
    public const int MaxPositions = 4096;
    public const float LayerNormEps = 1e-5f; // real nn.LayerNorm default

    /// <summary>[NumCodebooks][InputVocabSize, HiddenDim] -- 9 separate real embedding tables.</summary>
    public float[][] EmbedTokens { get; } = new float[NumCodebooks][];
    /// <summary>[MaxPositions, HiddenDim] -- real precomputed sinusoidal positional embedding buffer, loaded directly (not recomputed).</summary>
    public float[] EmbedPositions { get; }
    public ParlerDecoderLayerWeights[] Layers { get; } = new ParlerDecoderLayerWeights[NumLayers];
    public float[] FinalLayerNormWeight { get; }
    public float[] FinalLayerNormBias { get; }
    /// <summary>[NumCodebooks][OutputVocabSize, HiddenDim] -- 9 separate real output heads, NOT tied to EmbedTokens.</summary>
    public float[][] LmHeads { get; } = new float[NumCodebooks][];

    public ParlerDecoderWeights(SafetensorsLoader loader)
    {
        for (int cb = 0; cb < NumCodebooks; cb++)
        {
            EmbedTokens[cb] = loader.ReadF32($"decoder.model.decoder.embed_tokens.{cb}.weight");
            LmHeads[cb] = loader.ReadF32($"decoder.lm_heads.{cb}.weight");
        }
        EmbedPositions = loader.ReadF32("decoder.model.decoder.embed_positions.weights");
        FinalLayerNormWeight = loader.ReadF32("decoder.model.decoder.layer_norm.weight");
        FinalLayerNormBias = loader.ReadF32("decoder.model.decoder.layer_norm.bias");

        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"decoder.model.decoder.layers.{i}";
            Layers[i] = new ParlerDecoderLayerWeights
            {
                SelfAttnQWeight = loader.ReadF32($"{p}.self_attn.q_proj.weight"),
                SelfAttnKWeight = loader.ReadF32($"{p}.self_attn.k_proj.weight"),
                SelfAttnVWeight = loader.ReadF32($"{p}.self_attn.v_proj.weight"),
                SelfAttnOutWeight = loader.ReadF32($"{p}.self_attn.out_proj.weight"),
                SelfAttnLayerNormWeight = loader.ReadF32($"{p}.self_attn_layer_norm.weight"),
                SelfAttnLayerNormBias = loader.ReadF32($"{p}.self_attn_layer_norm.bias"),
                CrossAttnQWeight = loader.ReadF32($"{p}.encoder_attn.q_proj.weight"),
                CrossAttnKWeight = loader.ReadF32($"{p}.encoder_attn.k_proj.weight"),
                CrossAttnVWeight = loader.ReadF32($"{p}.encoder_attn.v_proj.weight"),
                CrossAttnOutWeight = loader.ReadF32($"{p}.encoder_attn.out_proj.weight"),
                CrossAttnLayerNormWeight = loader.ReadF32($"{p}.encoder_attn_layer_norm.weight"),
                CrossAttnLayerNormBias = loader.ReadF32($"{p}.encoder_attn_layer_norm.bias"),
                Fc1Weight = loader.ReadF32($"{p}.fc1.weight"),
                Fc2Weight = loader.ReadF32($"{p}.fc2.weight"),
                FinalLayerNormWeight = loader.ReadF32($"{p}.final_layer_norm.weight"),
                FinalLayerNormBias = loader.ReadF32($"{p}.final_layer_norm.bias"),
            };
        }
    }
}

public sealed class ParlerDecoderLayerWeights
{
    public required float[] SelfAttnQWeight { get; init; }
    public required float[] SelfAttnKWeight { get; init; }
    public required float[] SelfAttnVWeight { get; init; }
    public required float[] SelfAttnOutWeight { get; init; }
    public required float[] SelfAttnLayerNormWeight { get; init; }
    public required float[] SelfAttnLayerNormBias { get; init; }
    public required float[] CrossAttnQWeight { get; init; }
    public required float[] CrossAttnKWeight { get; init; }
    public required float[] CrossAttnVWeight { get; init; }
    public required float[] CrossAttnOutWeight { get; init; }
    public required float[] CrossAttnLayerNormWeight { get; init; }
    public required float[] CrossAttnLayerNormBias { get; init; }
    public required float[] Fc1Weight { get; init; }
    public required float[] Fc2Weight { get; init; }
    public required float[] FinalLayerNormWeight { get; init; }
    public required float[] FinalLayerNormBias { get; init; }
}
