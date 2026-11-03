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
        CheckpointRefusal = FusedExpertRefusal,
        CanBatchPredicate = (hp, tq) => !hp.IsMoE && !tq && hp.LayerHeadDim is null,
        CreateForwardPass = CreateForwardPass,
        ApplyModelSemantics = Gemma4ModelSemantics.Apply,
    };

    /// <summary>The expert tensor the Gemma 4 mixture-of-experts checkpoints (26B-A4B and the DiffusionGemma backbone) carry: gate and up fused in one tensor.</summary>
    internal const string FusedExpertTensor = "blk.0.ffn_gate_up_exps.weight";

    internal static IForwardPass CreateForwardPass(ArchitectureLoadContext ctx)
    {
        ArgumentNullException.ThrowIfNull(ctx);
        RejectFusedExpertLayout(ctx.Probe.TensorSource);
        return CommonForwardPassFactory.CreateDense(ctx);
    }

    /// <summary>
    /// The dense forward pass expects separate <c>ffn_gate_exps</c> / <c>ffn_up_exps</c> tensors and fails on the fused layout with a bare
    /// "Missing tensor"; say what is actually unsupported (the admitted evidence is Gemma 4 E4B text, which is dense). The loader and
    /// static scout both go through this, so neither reports a different answer.
    /// </summary>
    internal static string? FusedExpertRefusal(IModelTensorSource source) =>
        source.FindTensor(FusedExpertTensor) is null ? null :
            "This Gemma 4 checkpoint is a mixture-of-experts model with fused expert tensors (" + FusedExpertTensor + ", as in the 26B-A4B); " +
            "the gemma4 path only runs the dense Gemma 4 text models. See docs/2026-11-01-plan-6.md (Gemma 4 26B-A4B).";

    internal static void RejectFusedExpertLayout(IModelTensorSource source)
    {
        if (FusedExpertRefusal(source) is { } reason) throw new NotSupportedException(reason);
    }
}
