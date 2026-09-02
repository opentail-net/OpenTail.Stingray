
namespace OpenTail.Stingray.Audio.MusicGen;

/// <summary>
/// Weight loader for MusicGen's decoder-only LM (the actual autoregressive Transformer over
/// audio codebook tokens), from musicgen-small's own `model.safetensors`
/// (`decoder.*` prefix). Real tensor names confirmed against the checkpoint's own safetensors
/// header (2026-09-02, see docs/062-musicgen-implementation-plan.md) -- standard OPT/Bart-style
/// pre-norm decoder layer: separate un-biased Q/K/V/O for both self- and cross-attention, plain
/// `fc1`/`fc2` FFN (GELU, un-biased), LayerNorm (WITH bias, unlike T5's bias-free RMSNorm) before
/// each sub-block. One embedding table AND one output head PER codebook (4 of each) -- MusicGen
/// sums the 4 codebook embeddings rather than concatenating them (see
/// <see cref="MusicGenTransformer"/>), and predicts each codebook's logits independently from a
/// shared hidden state.
/// </summary>
public sealed class MusicGenTransformerWeights
{
    public float[][] EmbedTokens { get; } = new float[MusicGenConfig.NumCodebooks][]; // [codebook][2049 * hidden]
    public float[] EmbedPositions { get; } // [MaxPositionEmbeddings, hidden] -- real learned/precomputed sinusoidal buffer, loaded verbatim
    public MusicGenDecoderLayerWeights[] Layers { get; } = new MusicGenDecoderLayerWeights[MusicGenConfig.DecoderNumLayers];
    public float[] FinalLayerNormWeight { get; }
    public float[] FinalLayerNormBias { get; }
    public CfmLinearWeight[] LmHeads { get; } = new CfmLinearWeight[MusicGenConfig.NumCodebooks]; // [codebook]: hidden -> CodebookSize (2048, no pad row)

    /// <summary>Real `MusicgenForConditionalGeneration.enc_to_dec_proj`: projects the T5 text encoder's 768-dim output up to the decoder's 1024-dim hidden size BEFORE cross-attention ever sees it. Easy to miss since it lives outside both `decoder.*` and `text_encoder.*` (top-level `enc_to_dec_proj.{weight,bias}`) -- found via a real array-length crash when cross-attention K/V projection was first wired to consume raw 768-dim T5 output.</summary>
    public CfmLinearWeight EncToDecProjWeight { get; }
    public float[] EncToDecProjBias { get; }

    public MusicGenTransformerWeights(SafetensorsLoader loader)
    {
        EncToDecProjWeight = CfmLinearWeight.FromF32(loader.ReadF32("enc_to_dec_proj.weight"), outDim: MusicGenConfig.DecoderHiddenSize, inDim: MusicGenConfig.TextDModel);
        EncToDecProjBias = loader.ReadF32("enc_to_dec_proj.bias");
        for (int q = 0; q < MusicGenConfig.NumCodebooks; q++)
            EmbedTokens[q] = loader.ReadF32($"decoder.model.decoder.embed_tokens.{q}.weight");

        EmbedPositions = loader.ReadF32("decoder.model.decoder.embed_positions.weights");
        FinalLayerNormWeight = loader.ReadF32("decoder.model.decoder.layer_norm.weight");
        FinalLayerNormBias = loader.ReadF32("decoder.model.decoder.layer_norm.bias");

        int hidden = MusicGenConfig.DecoderHiddenSize;
        for (int q = 0; q < MusicGenConfig.NumCodebooks; q++)
            LmHeads[q] = CfmLinearWeight.FromF32(loader.ReadF32($"decoder.lm_heads.{q}.weight"), outDim: MusicGenConfig.CodebookSize, inDim: hidden);

        for (int i = 0; i < MusicGenConfig.DecoderNumLayers; i++)
        {
            string p = $"decoder.model.decoder.layers.{i}";
            Layers[i] = new MusicGenDecoderLayerWeights
            {
                SelfAttnQWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.self_attn.q_proj.weight"), outDim: hidden, inDim: hidden),
                SelfAttnKWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.self_attn.k_proj.weight"), outDim: hidden, inDim: hidden),
                SelfAttnVWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.self_attn.v_proj.weight"), outDim: hidden, inDim: hidden),
                SelfAttnOWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.self_attn.out_proj.weight"), outDim: hidden, inDim: hidden),
                SelfAttnLayerNormWeight = loader.ReadF32($"{p}.self_attn_layer_norm.weight"),
                SelfAttnLayerNormBias = loader.ReadF32($"{p}.self_attn_layer_norm.bias"),

                CrossAttnQWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.encoder_attn.q_proj.weight"), outDim: hidden, inDim: hidden),
                CrossAttnKWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.encoder_attn.k_proj.weight"), outDim: hidden, inDim: hidden),
                CrossAttnVWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.encoder_attn.v_proj.weight"), outDim: hidden, inDim: hidden),
                CrossAttnOWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.encoder_attn.out_proj.weight"), outDim: hidden, inDim: hidden),
                CrossAttnLayerNormWeight = loader.ReadF32($"{p}.encoder_attn_layer_norm.weight"),
                CrossAttnLayerNormBias = loader.ReadF32($"{p}.encoder_attn_layer_norm.bias"),

                Fc1Weight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.fc1.weight"), outDim: MusicGenConfig.DecoderFfnDim, inDim: hidden),
                Fc2Weight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.fc2.weight"), outDim: hidden, inDim: MusicGenConfig.DecoderFfnDim),
                FinalLayerNormWeight = loader.ReadF32($"{p}.final_layer_norm.weight"),
                FinalLayerNormBias = loader.ReadF32($"{p}.final_layer_norm.bias"),
            };
        }
    }
}

public sealed class MusicGenDecoderLayerWeights
{
    public required CfmLinearWeight SelfAttnQWeight { get; init; }
    public required CfmLinearWeight SelfAttnKWeight { get; init; }
    public required CfmLinearWeight SelfAttnVWeight { get; init; }
    public required CfmLinearWeight SelfAttnOWeight { get; init; }
    public required float[] SelfAttnLayerNormWeight { get; init; }
    public required float[] SelfAttnLayerNormBias { get; init; }

    public required CfmLinearWeight CrossAttnQWeight { get; init; }
    public required CfmLinearWeight CrossAttnKWeight { get; init; }
    public required CfmLinearWeight CrossAttnVWeight { get; init; }
    public required CfmLinearWeight CrossAttnOWeight { get; init; }
    public required float[] CrossAttnLayerNormWeight { get; init; }
    public required float[] CrossAttnLayerNormBias { get; init; }

    public required CfmLinearWeight Fc1Weight { get; init; }
    public required CfmLinearWeight Fc2Weight { get; init; }
    public required float[] FinalLayerNormWeight { get; init; }
    public required float[] FinalLayerNormBias { get; init; }
}
