namespace OpenTail.Stingray.Audio.VoxCpm2;

/// <summary>Real weights for VoxCPM2's DiT estimator (the flow-matching velocity-field network),
/// ported from `generator.cpp`'s `VoxCPM2DiTEstimatorRuntime::Impl::build` (not guessed). Real
/// prefix `weights/feat_decoder.estimator.*`; the transformer decoder shares the exact same
/// config shape (hence weight struct) as the local encoder's stack, see
/// <see cref="VoxCpm2MiniCpmBidirectionalStack"/>.</summary>
public sealed class VoxCpm2DiTEstimatorWeights
{
    public required float[] InProjWeight { get; init; } // [hiddenDim, featDim]
    public required float[] InProjBias { get; init; }
    public required float[] CondProjWeight { get; init; } // [hiddenDim, featDim]
    public required float[] CondProjBias { get; init; }
    public required float[] TimeMlp1Weight { get; init; } // [hiddenDim, hiddenDim]
    public required float[] TimeMlp1Bias { get; init; }
    public required float[] TimeMlp2Weight { get; init; }
    public required float[] TimeMlp2Bias { get; init; }
    public required float[] DeltaTimeMlp1Weight { get; init; }
    public required float[] DeltaTimeMlp1Bias { get; init; }
    public required float[] DeltaTimeMlp2Weight { get; init; }
    public required float[] DeltaTimeMlp2Bias { get; init; }
    public required VoxCpm2MiniCpmLayerWeights[] DecoderLayers { get; init; }
    public required float[] DecoderNorm { get; init; }
    public required float[] OutProjWeight { get; init; } // [featDim, hiddenDim]
    public required float[] OutProjBias { get; init; }

    public static VoxCpm2DiTEstimatorWeights Load(int numLayers, Func<string, float[]> get)
    {
        var layers = new VoxCpm2MiniCpmLayerWeights[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            string p = $"weights/feat_decoder.estimator.decoder.layers.{i}.";
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

        return new VoxCpm2DiTEstimatorWeights
        {
            InProjWeight = get("weights/feat_decoder.estimator.in_proj.weight"),
            InProjBias = get("weights/feat_decoder.estimator.in_proj.bias"),
            CondProjWeight = get("weights/feat_decoder.estimator.cond_proj.weight"),
            CondProjBias = get("weights/feat_decoder.estimator.cond_proj.bias"),
            TimeMlp1Weight = get("weights/feat_decoder.estimator.time_mlp.linear_1.weight"),
            TimeMlp1Bias = get("weights/feat_decoder.estimator.time_mlp.linear_1.bias"),
            TimeMlp2Weight = get("weights/feat_decoder.estimator.time_mlp.linear_2.weight"),
            TimeMlp2Bias = get("weights/feat_decoder.estimator.time_mlp.linear_2.bias"),
            DeltaTimeMlp1Weight = get("weights/feat_decoder.estimator.delta_time_mlp.linear_1.weight"),
            DeltaTimeMlp1Bias = get("weights/feat_decoder.estimator.delta_time_mlp.linear_1.bias"),
            DeltaTimeMlp2Weight = get("weights/feat_decoder.estimator.delta_time_mlp.linear_2.weight"),
            DeltaTimeMlp2Bias = get("weights/feat_decoder.estimator.delta_time_mlp.linear_2.bias"),
            DecoderLayers = layers,
            DecoderNorm = get("weights/feat_decoder.estimator.decoder.norm.weight"),
            OutProjWeight = get("weights/feat_decoder.estimator.out_proj.weight"),
            OutProjBias = get("weights/feat_decoder.estimator.out_proj.bias"),
        };
    }
}
