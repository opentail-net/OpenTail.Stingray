using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Diffusion.AceStep.Transformer;

/// <summary>
/// Weight loader for ACE-Step Turbo's 24-layer DiT (`AceStepDiTModel`), from the real
/// `acestep-v15-turbo/model.safetensors` (`decoder.*` prefix). Every tensor name/shape below was
/// confirmed against the real checkpoint's own safetensors header, not assumed from the reference
/// source alone -- see docs/064-acestep-implementation-plan.md.
/// </summary>
public sealed class AceStepDiTWeights
{
    public required float[] ProjInWeight { get; init; } // Conv1d [hidden(2048), inChannels(192), patchSize(2)]
    public required float[] ProjInBias { get; init; }
    public required float[] ProjOutWeight { get; init; } // ConvTranspose1d [hidden(2048), acousticDim(64), patchSize(2)]
    public required float[] ProjOutBias { get; init; }

    public required AceStepTimestepEmbeddingWeights TimeEmbed { get; init; }
    public required AceStepTimestepEmbeddingWeights TimeEmbedR { get; init; }

    public required CfmLinearWeight ConditionEmbedderWeight { get; init; }
    public required float[] ConditionEmbedderBias { get; init; }

    public required float[] NormOutWeight { get; init; }
    public required float[] ScaleShiftTable { get; init; } // [2, hidden] (batch dim 1 dropped)

    public required AceStepDiTLayerWeights[] Layers { get; init; }

    public static AceStepDiTWeights Load(SafetensorsLoader loader)
    {
        int hidden = AceStepConfig.HiddenSize;

        var layers = new AceStepDiTLayerWeights[AceStepConfig.NumHiddenLayers];
        for (int i = 0; i < layers.Length; i++)
            layers[i] = LoadLayer(loader, i);

        return new AceStepDiTWeights
        {
            ProjInWeight = loader.ReadF32("decoder.proj_in.1.weight"),
            ProjInBias = loader.ReadF32("decoder.proj_in.1.bias"),
            ProjOutWeight = loader.ReadF32("decoder.proj_out.1.weight"),
            ProjOutBias = loader.ReadF32("decoder.proj_out.1.bias"),
            TimeEmbed = LoadTimestepEmbedding(loader, "decoder.time_embed"),
            TimeEmbedR = LoadTimestepEmbedding(loader, "decoder.time_embed_r"),
            ConditionEmbedderWeight = CfmLinearWeight.FromF32(loader.ReadF32("decoder.condition_embedder.weight"), outDim: hidden, inDim: hidden),
            ConditionEmbedderBias = loader.ReadF32("decoder.condition_embedder.bias"),
            NormOutWeight = loader.ReadF32("decoder.norm_out.weight"),
            ScaleShiftTable = loader.ReadF32("decoder.scale_shift_table"), // [1,2,hidden] flat == [2,hidden] flat
            Layers = layers,
        };
    }

    private static AceStepTimestepEmbeddingWeights LoadTimestepEmbedding(SafetensorsLoader loader, string p)
    {
        int hidden = AceStepConfig.HiddenSize;
        return new AceStepTimestepEmbeddingWeights
        {
            Linear1Weight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.linear_1.weight"), outDim: hidden, inDim: 256),
            Linear1Bias = loader.ReadF32($"{p}.linear_1.bias"),
            Linear2Weight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.linear_2.weight"), outDim: hidden, inDim: hidden),
            Linear2Bias = loader.ReadF32($"{p}.linear_2.bias"),
            TimeProjWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.time_proj.weight"), outDim: hidden * 6, inDim: hidden),
            TimeProjBias = loader.ReadF32($"{p}.time_proj.bias"),
        };
    }

    private static AceStepDiTLayerWeights LoadLayer(SafetensorsLoader loader, int i)
    {
        int hidden = AceStepConfig.HiddenSize;
        int qDim = AceStepConfig.NumAttentionHeads * AceStepConfig.HeadDim; // 2048
        int kvDim = AceStepConfig.NumKeyValueHeads * AceStepConfig.HeadDim; // 1024
        int ffn = AceStepConfig.IntermediateSize;
        string p = $"decoder.layers.{i}";

        return new AceStepDiTLayerWeights
        {
            SelfAttnNormWeight = loader.ReadF32($"{p}.self_attn_norm.weight"),
            SelfAttn = LoadAttention(loader, $"{p}.self_attn", qDim, kvDim, hidden),

            CrossAttnNormWeight = loader.ReadF32($"{p}.cross_attn_norm.weight"),
            CrossAttn = LoadAttention(loader, $"{p}.cross_attn", qDim, kvDim, hidden),

            MlpNormWeight = loader.ReadF32($"{p}.mlp_norm.weight"),
            MlpGateWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.mlp.gate_proj.weight"), outDim: ffn, inDim: hidden),
            MlpUpWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.mlp.up_proj.weight"), outDim: ffn, inDim: hidden),
            MlpDownWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.mlp.down_proj.weight"), outDim: hidden, inDim: ffn),

            ScaleShiftTable = loader.ReadF32($"{p}.scale_shift_table"), // [1,6,hidden] flat == [6,hidden] flat
        };
    }

    private static AceStepAttentionWeights LoadAttention(SafetensorsLoader loader, string p, int qDim, int kvDim, int hidden) => new()
    {
        QWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.q_proj.weight"), outDim: qDim, inDim: hidden),
        KWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.k_proj.weight"), outDim: kvDim, inDim: hidden),
        VWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.v_proj.weight"), outDim: kvDim, inDim: hidden),
        OWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.o_proj.weight"), outDim: hidden, inDim: qDim),
        QNormWeight = loader.ReadF32($"{p}.q_norm.weight"), // [headDim]
        KNormWeight = loader.ReadF32($"{p}.k_norm.weight"), // [headDim]
    };
}

public sealed class AceStepTimestepEmbeddingWeights
{
    public required CfmLinearWeight Linear1Weight { get; init; }
    public required float[] Linear1Bias { get; init; }
    public required CfmLinearWeight Linear2Weight { get; init; }
    public required float[] Linear2Bias { get; init; }
    public required CfmLinearWeight TimeProjWeight { get; init; }
    public required float[] TimeProjBias { get; init; }
}

public sealed class AceStepAttentionWeights
{
    public required CfmLinearWeight QWeight { get; init; }
    public required CfmLinearWeight KWeight { get; init; }
    public required CfmLinearWeight VWeight { get; init; }
    public required CfmLinearWeight OWeight { get; init; }
    public required float[] QNormWeight { get; init; }
    public required float[] KNormWeight { get; init; }
}

public sealed class AceStepDiTLayerWeights
{
    public required float[] SelfAttnNormWeight { get; init; }
    public required AceStepAttentionWeights SelfAttn { get; init; }

    public required float[] CrossAttnNormWeight { get; init; }
    public required AceStepAttentionWeights CrossAttn { get; init; }

    public required float[] MlpNormWeight { get; init; }
    public required CfmLinearWeight MlpGateWeight { get; init; }
    public required CfmLinearWeight MlpUpWeight { get; init; }
    public required CfmLinearWeight MlpDownWeight { get; init; }

    public required float[] ScaleShiftTable { get; init; }
}
