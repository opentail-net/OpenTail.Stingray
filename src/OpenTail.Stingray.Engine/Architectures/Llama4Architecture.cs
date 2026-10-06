namespace OpenTail.Stingray.Engine;

internal static class Llama4Architecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "llama4",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Llama 4 Scout (`llama4`)",
        EvidenceDoc = "docs/STATUS.md",
        FallbackChat = FallbackChatFormat.Llama4,
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
