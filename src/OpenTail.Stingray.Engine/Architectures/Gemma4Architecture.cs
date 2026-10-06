namespace OpenTail.Stingray.Engine;

internal static class Gemma4Architecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "gemma4",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Gemma 4 E4B text (`gemma4`)",
        EvidenceDoc = "docs/STATUS.md",
        ThinkingDefaultOff = true,
        SupportsImageInput = true,
        ProjectorFileHints = ["*mmproj*.gguf", "*vision*.gguf"],
        CanBatchPredicate = (hp, tq) => !hp.IsMoE && !tq && hp.LayerHeadDim is null,
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
