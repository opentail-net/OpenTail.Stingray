
namespace OpenTail.Stingray.Audio.MusicGen;

/// <summary>
/// Weight loader for MusicGen's text conditioning encoder. **Correction, 2026-09-02**: the
/// initial version of this file assumed the text encoder had to be loaded from a SEPARATE stock
/// `t5-base` checkpoint (reasoning: HF's `MusicgenForConditionalGeneration` documentation
/// describes composing sub-models via `AutoModel.from_pretrained`). Real inspection of
/// musicgen-small's own `model.safetensors` header disproved that: it contains a full,
/// self-contained `text_encoder.*` tensor tree (`text_encoder.shared.weight`,
/// `text_encoder.encoder.block.{i}.*`, `text_encoder.encoder.final_layer_norm.weight`) --
/// EXACTLY the same "bundled, not composed" convention Parler-TTS's
/// <see cref="Parler.T5EncoderWeights"/> already documented. Loads from the SAME loader as
/// <see cref="MusicGenTransformerWeights"/> (musicgen-small's single checkpoint file), not a
/// separate `t5-base` download.
///
/// <para><b>Real t5-base is NOT gated</b> (unlike Parler's flan-t5-large): `config.json`'s
/// `is_gated_act: false`, `feed_forward_proj: "relu"` -- the FFN is the original
/// `T5DenseActDense` (`wo(relu(wi(x)))`, ONE `wi` matrix), not `T5DenseGatedActDense`'s
/// `wi_0`/`wi_1` pair. See <see cref="MusicGenTextEncoder"/> for the forward pass; do not reuse
/// <see cref="Parler.T5Encoder"/>'s gated-GELU FFN math here.</para>
/// </summary>
public sealed class MusicGenTextEncoderWeights
{
    public float[] SharedEmbedding { get; } // [vocab(32128), DModel]
    public MusicGenTextLayerWeights[] Layers { get; } = new MusicGenTextLayerWeights[MusicGenConfig.TextNumLayers];
    public float[] FinalLayerNormWeight { get; }
    public float[] RelativeAttentionBias { get; } // [RelativeAttentionNumBuckets, NumHeads], block 0 only

    public MusicGenTextEncoderWeights(SafetensorsLoader loader)
    {
        SharedEmbedding = loader.ReadF32("text_encoder.shared.weight");
        FinalLayerNormWeight = loader.ReadF32("text_encoder.encoder.final_layer_norm.weight");
        RelativeAttentionBias = loader.ReadF32("text_encoder.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight");

        int qkvDim = MusicGenConfig.TextNumHeads * MusicGenConfig.TextDKv; // 768, equals DModel here

        for (int i = 0; i < MusicGenConfig.TextNumLayers; i++)
        {
            string p = $"text_encoder.encoder.block.{i}";
            Layers[i] = new MusicGenTextLayerWeights
            {
                SelfAttnQWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.layer.0.SelfAttention.q.weight"), outDim: qkvDim, inDim: MusicGenConfig.TextDModel),
                SelfAttnKWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.layer.0.SelfAttention.k.weight"), outDim: qkvDim, inDim: MusicGenConfig.TextDModel),
                SelfAttnVWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.layer.0.SelfAttention.v.weight"), outDim: qkvDim, inDim: MusicGenConfig.TextDModel),
                SelfAttnOWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.layer.0.SelfAttention.o.weight"), outDim: MusicGenConfig.TextDModel, inDim: qkvDim),
                SelfAttnLayerNormWeight = loader.ReadF32($"{p}.layer.0.layer_norm.weight"),
                FfnWiWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.layer.1.DenseReluDense.wi.weight"), outDim: MusicGenConfig.TextDFf, inDim: MusicGenConfig.TextDModel),
                FfnWoWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.layer.1.DenseReluDense.wo.weight"), outDim: MusicGenConfig.TextDModel, inDim: MusicGenConfig.TextDFf),
                FfnLayerNormWeight = loader.ReadF32($"{p}.layer.1.layer_norm.weight"),
            };
        }
    }
}

public sealed class MusicGenTextLayerWeights
{
    public required CfmLinearWeight SelfAttnQWeight { get; init; }
    public required CfmLinearWeight SelfAttnKWeight { get; init; }
    public required CfmLinearWeight SelfAttnVWeight { get; init; }
    public required CfmLinearWeight SelfAttnOWeight { get; init; }
    public required float[] SelfAttnLayerNormWeight { get; init; }
    public required CfmLinearWeight FfnWiWeight { get; init; }
    public required CfmLinearWeight FfnWoWeight { get; init; }
    public required float[] FfnLayerNormWeight { get; init; }
}
