namespace OpenTail.Stingray.Audio.VoxCpm2;

/// <summary>Real weights for VoxCPM2's per-step fusion/projection layers, ported from
/// `assets.cpp`'s `validate_weight_anchors` shapes and `generator.cpp`'s `VoxCPM2StepProjection
/// Runtime::Impl::build` (not guessed). All plain `Linear` layers; `fsq_in_proj`/`fsq_out_proj`
/// implement the FSQ (finite scalar quantization) residual-LM bottleneck, `fusion_concat_proj`/
/// `lm_to_dit_proj` are each REUSED across two different real call sites (current-frame and
/// FSQ-quantized-frame variants of the same projection, not two separate learned weights).</summary>
public sealed class VoxCpm2StepProjectionWeights
{
    public required float[] FsqInProjWeight { get; init; } // [latentDim, hiddenDim]
    public required float[] FsqInProjBias { get; init; }
    public required float[] FsqOutProjWeight { get; init; } // [hiddenDim, latentDim]
    public required float[] FsqOutProjBias { get; init; }
    public required float[] FusionConcatProjWeight { get; init; } // [hiddenDim, 2*hiddenDim]
    public required float[] FusionConcatProjBias { get; init; }
    public required float[] LmToDitProjWeight { get; init; } // [ditHiddenDim, hiddenDim]
    public required float[] LmToDitProjBias { get; init; }
    public required float[] ResToDitProjWeight { get; init; } // [ditHiddenDim, hiddenDim]
    public required float[] ResToDitProjBias { get; init; }
    public required float[] StopProjWeight { get; init; } // [hiddenDim, hiddenDim]
    public required float[] StopProjBias { get; init; }
    public required float[] StopHeadWeight { get; init; } // [2, hiddenDim], no bias

    public static VoxCpm2StepProjectionWeights Load(Func<string, float[]> get) => new()
    {
        FsqInProjWeight = get("weights/fsq_layer.in_proj.weight"),
        FsqInProjBias = get("weights/fsq_layer.in_proj.bias"),
        FsqOutProjWeight = get("weights/fsq_layer.out_proj.weight"),
        FsqOutProjBias = get("weights/fsq_layer.out_proj.bias"),
        FusionConcatProjWeight = get("weights/fusion_concat_proj.weight"),
        FusionConcatProjBias = get("weights/fusion_concat_proj.bias"),
        LmToDitProjWeight = get("weights/lm_to_dit_proj.weight"),
        LmToDitProjBias = get("weights/lm_to_dit_proj.bias"),
        ResToDitProjWeight = get("weights/res_to_dit_proj.weight"),
        ResToDitProjBias = get("weights/res_to_dit_proj.bias"),
        StopProjWeight = get("weights/stop_proj.weight"),
        StopProjBias = get("weights/stop_proj.bias"),
        StopHeadWeight = get("weights/stop_head.weight"),
    };
}
