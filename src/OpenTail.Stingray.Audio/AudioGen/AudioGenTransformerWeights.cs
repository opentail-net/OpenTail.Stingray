
namespace OpenTail.Stingray.Audio.AudioGen;

/// <summary>
/// Weight loader for AudioGen's decoder-only LM, from the real native AudioCraft checkpoint
/// converted to safetensors (`audiogen-medium-lm.safetensors`, tensor names preserved verbatim
/// from the real `state_dict.bin`'s `best_state` dict -- see
/// docs/063-audiogen-implementation-plan.md for the conversion). Unlike MusicGen's HF-remapped
/// checkpoint, this is native AudioCraft `StreamingTransformerLayer` naming: FUSED
/// `in_proj_weight` (`[3*dim, dim]`, Q/K/V concatenated in that order -- confirmed from the real
/// `audiocraft.modules.transformer` source) for BOTH self- and cross-attention, `linear1`/`linear2`
/// FFN (no bias -- real config `bias_ff: false`/`bias_attn: false`/`bias_proj: false`), and
/// `norm1`/`norm_cross`/`norm2` LayerNorms (WITH bias -- real `nn.LayerNorm` always carries one).
/// One embedding table AND one output head per codebook (`emb.{q}`/`linears.{q}`), same convention
/// as MusicGen. No `enc_to_dec_proj` -- the analogous role is played directly by
/// `condition_provider.conditioners.description.output_proj` (WITH bias, unlike MusicGen's).
/// </summary>
public sealed class AudioGenTransformerWeights
{
    public float[][] EmbedTokens { get; } = new float[AudioGenConfig.NumCodebooks][]; // [codebook][2049 * hidden]
    public AudioGenDecoderLayerWeights[] Layers { get; } = new AudioGenDecoderLayerWeights[AudioGenConfig.NumLayers];
    public float[] OutNormWeight { get; }
    public float[] OutNormBias { get; }
    public CfmLinearWeight[] LmHeads { get; } = new CfmLinearWeight[AudioGenConfig.NumCodebooks]; // hidden -> CodebookSize, no bias (bias_proj=false)

    /// <summary>Real `condition_provider.conditioners.description.output_proj`: projects T5-large's 1024-dim output up to the transformer's 1536-dim hidden size, WITH bias (unlike MusicGen's bias-free `enc_to_dec_proj`).</summary>
    public CfmLinearWeight OutputProjWeight { get; }
    public float[] OutputProjBias { get; }

    public AudioGenTransformerWeights(SafetensorsLoader loader)
    {
        for (int q = 0; q < AudioGenConfig.NumCodebooks; q++)
            EmbedTokens[q] = loader.ReadF32($"emb.{q}.weight");

        OutNormWeight = loader.ReadF32("out_norm.weight");
        OutNormBias = loader.ReadF32("out_norm.bias");

        int hidden = AudioGenConfig.HiddenSize;
        for (int q = 0; q < AudioGenConfig.NumCodebooks; q++)
            LmHeads[q] = CfmLinearWeight.FromF32(loader.ReadF32($"linears.{q}.weight"), outDim: AudioGenConfig.CodebookSize, inDim: hidden);

        OutputProjWeight = CfmLinearWeight.FromF32(
            loader.ReadF32("condition_provider.conditioners.description.output_proj.weight"),
            outDim: hidden, inDim: AudioGenConfig.TextDModel);
        OutputProjBias = loader.ReadF32("condition_provider.conditioners.description.output_proj.bias");

        for (int i = 0; i < AudioGenConfig.NumLayers; i++)
        {
            string p = $"transformer.layers.{i}";
            var selfInProj = loader.ReadF32($"{p}.self_attn.in_proj_weight"); // [3*hidden, hidden], Q/K/V rows concatenated
            var crossInProj = loader.ReadF32($"{p}.cross_attention.in_proj_weight");

            Layers[i] = new AudioGenDecoderLayerWeights
            {
                SelfAttnQWeight = CfmLinearWeight.FromF32(SliceRows(selfInProj, hidden, hidden, 0), outDim: hidden, inDim: hidden),
                SelfAttnKWeight = CfmLinearWeight.FromF32(SliceRows(selfInProj, hidden, hidden, hidden), outDim: hidden, inDim: hidden),
                SelfAttnVWeight = CfmLinearWeight.FromF32(SliceRows(selfInProj, hidden, hidden, 2 * hidden), outDim: hidden, inDim: hidden),
                SelfAttnOutProjWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.self_attn.out_proj.weight"), outDim: hidden, inDim: hidden),
                Norm1Weight = loader.ReadF32($"{p}.norm1.weight"),
                Norm1Bias = loader.ReadF32($"{p}.norm1.bias"),

                CrossAttnQWeight = CfmLinearWeight.FromF32(SliceRows(crossInProj, hidden, hidden, 0), outDim: hidden, inDim: hidden),
                CrossAttnKWeight = CfmLinearWeight.FromF32(SliceRows(crossInProj, hidden, hidden, hidden), outDim: hidden, inDim: hidden),
                CrossAttnVWeight = CfmLinearWeight.FromF32(SliceRows(crossInProj, hidden, hidden, 2 * hidden), outDim: hidden, inDim: hidden),
                CrossAttnOutProjWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.cross_attention.out_proj.weight"), outDim: hidden, inDim: hidden),
                NormCrossWeight = loader.ReadF32($"{p}.norm_cross.weight"),
                NormCrossBias = loader.ReadF32($"{p}.norm_cross.bias"),

                Linear1Weight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.linear1.weight"), outDim: AudioGenConfig.FfnDim, inDim: hidden),
                Linear2Weight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.linear2.weight"), outDim: hidden, inDim: AudioGenConfig.FfnDim),
                Norm2Weight = loader.ReadF32($"{p}.norm2.weight"),
                Norm2Bias = loader.ReadF32($"{p}.norm2.bias"),
            };
        }
    }

    /// <summary>Extracts rows `[rowOffset, rowOffset+rowCount)` from a flat `[totalRows, cols]` row-major matrix -- used once at load time to split each layer's fused `in_proj_weight` into its Q/K/V thirds (real `audiocraft` slicing order: `[:dim]`=Q, `[dim:2*dim]`=K, `[2*dim:]`=V).</summary>
    private static float[] SliceRows(float[] flat, int rowCount, int cols, int rowOffset)
    {
        var result = new float[rowCount * cols];
        Array.Copy(flat, rowOffset * cols, result, 0, rowCount * cols);
        return result;
    }
}

public sealed class AudioGenDecoderLayerWeights
{
    /// <summary>Pre-split from the real fused `in_proj_weight` (`[3*hidden,hidden]`) once at load time -- real `audiocraft` slicing order: rows `[0,hidden)`=Q, `[hidden,2*hidden)`=K, `[2*hidden,3*hidden)`=V.</summary>
    public required CfmLinearWeight SelfAttnQWeight { get; init; }
    public required CfmLinearWeight SelfAttnKWeight { get; init; }
    public required CfmLinearWeight SelfAttnVWeight { get; init; }
    public required CfmLinearWeight SelfAttnOutProjWeight { get; init; }
    public required float[] Norm1Weight { get; init; }
    public required float[] Norm1Bias { get; init; }

    public required CfmLinearWeight CrossAttnQWeight { get; init; }
    public required CfmLinearWeight CrossAttnKWeight { get; init; }
    public required CfmLinearWeight CrossAttnVWeight { get; init; }
    public required CfmLinearWeight CrossAttnOutProjWeight { get; init; }
    public required float[] NormCrossWeight { get; init; }
    public required float[] NormCrossBias { get; init; }

    public required CfmLinearWeight Linear1Weight { get; init; }
    public required CfmLinearWeight Linear2Weight { get; init; }
    public required float[] Norm2Weight { get; init; }
    public required float[] Norm2Bias { get; init; }
}
