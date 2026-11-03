namespace OpenTail.Stingray.Engine;

using OpenTail.Stingray.Core;

internal static class Gemma4Architecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "gemma4",
        ChatProtocolId = "gemma",
        UsesNeoxRope = true,
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Gemma 4 E4B text (`gemma4`)",
        EvidenceDoc = "docs/STATUS.md",
        ThinkingDefaultOff = true,
        SupportsImageInput = true,
        ProjectorFileHints = ["*mmproj*.gguf", "*vision*.gguf"],
        CanBatchPredicate = (hp, tq) => !hp.IsMoE && !tq && hp.LayerHeadDim is null,
        CreateForwardPass = CreateForwardPass,
        ApplyModelSemantics = Gemma4ModelSemantics.Apply,
    };

    /// <summary>The expert tensor the Gemma 4 mixture-of-experts checkpoints (26B-A4B and the DiffusionGemma backbone) carry: gate and up fused in one tensor. Its presence selects the parallel dense+MoE layer (CPU only, <see cref="ModelHyperparams.ParallelDenseMoe"/>).</summary>
    internal const string FusedExpertTensor = "blk.0.ffn_gate_up_exps.weight";

    internal static IForwardPass CreateForwardPass(ArchitectureLoadContext ctx)
    {
        ArgumentNullException.ThrowIfNull(ctx);
        return CommonForwardPassFactory.CreateDense(ctx);
    }
}
