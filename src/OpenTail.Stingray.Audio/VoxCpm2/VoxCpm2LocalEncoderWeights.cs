namespace OpenTail.Stingray.Audio.VoxCpm2;

public sealed class VoxCpm2MiniCpmLayerWeights
{
    public required float[] InputNorm { get; init; }
    public required float[] QProjWeight { get; init; } // [numHeads*headDim, hiddenDim]
    public required float[] KProjWeight { get; init; } // [numKvHeads*headDim, hiddenDim]
    public required float[] VProjWeight { get; init; } // [numKvHeads*headDim, hiddenDim]
    public required float[] OProjWeight { get; init; } // [hiddenDim, numHeads*headDim]
    public required float[] PostNorm { get; init; }
    public required float[] GateProjWeight { get; init; } // [ffnDim, hiddenDim]
    public required float[] UpProjWeight { get; init; } // [ffnDim, hiddenDim]
    public required float[] DownProjWeight { get; init; } // [hiddenDim, ffnDim]
}

/// <summary>
/// Real weights for VoxCPM2's local encoder -- a small, bidirectional (`is_causal=false`) MiniCPM-
/// architecture transformer (real config `encoder_config`: `hidden_dim=1024`, `ffn_dim=4096`,
/// `num_heads=16`, `num_layers=12`, `kv_channels=128`; `num_key_value_heads=2` inherited from the
/// base `lm_config`, per `minicpm.cpp`'s `local_transformer_config`), ported from
/// `generator.cpp`'s `VoxCPM2LocalEncoderRuntime::Impl::build` and `minicpm_blocks.h`'s
/// `minicpm_layer`/`minicpm_transformer` (not guessed). Real prefix `weights/feat_encoder.*`.
/// </summary>
public sealed class VoxCpm2LocalEncoderWeights
{
    public required float[] InProjWeight { get; init; } // [encoderHiddenDim, featDim]
    public required float[] InProjBias { get; init; }
    public required float[] SpecialToken { get; init; } // [encoderHiddenDim]
    public required VoxCpm2MiniCpmLayerWeights[] Layers { get; init; }
    public required float[] FinalNorm { get; init; }
    public required float[] EncToLmProjWeight { get; init; } // [lmHiddenDim, encoderHiddenDim]
    public required float[] EncToLmProjBias { get; init; }

    public static VoxCpm2LocalEncoderWeights Load(int numLayers, Func<string, float[]> get)
    {
        var layers = new VoxCpm2MiniCpmLayerWeights[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            string p = $"weights/feat_encoder.encoder.layers.{i}.";
            layers[i] = new VoxCpm2MiniCpmLayerWeights
            {
                InputNorm = get(p + "input_layernorm.weight"),
                QProjWeight = get(p + "self_attn.q_proj.weight"),
                KProjWeight = get(p + "self_attn.k_proj.weight"),
                VProjWeight = get(p + "self_attn.v_proj.weight"),
                OProjWeight = get(p + "self_attn.o_proj.weight"),
                PostNorm = get(p + "post_attention_layernorm.weight"),
                GateProjWeight = get(p + "mlp.gate_proj.weight"),
                UpProjWeight = get(p + "mlp.up_proj.weight"),
                DownProjWeight = get(p + "mlp.down_proj.weight"),
            };
        }

        return new VoxCpm2LocalEncoderWeights
        {
            InProjWeight = get("weights/feat_encoder.in_proj.weight"),
            InProjBias = get("weights/feat_encoder.in_proj.bias"),
            SpecialToken = get("weights/feat_encoder.special_token"),
            Layers = layers,
            FinalNorm = get("weights/feat_encoder.encoder.norm.weight"),
            EncToLmProjWeight = get("weights/enc_to_lm_proj.weight"),
            EncToLmProjBias = get("weights/enc_to_lm_proj.bias"),
        };
    }
}
