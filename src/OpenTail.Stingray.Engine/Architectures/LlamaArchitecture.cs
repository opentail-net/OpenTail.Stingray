namespace OpenTail.Stingray.Engine;

internal static class LlamaArchitecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "llama",
        Status = AdmissionStatus.Admitted,
        EvidenceDoc = "docs/STATUS.md",
        FallbackChat = FallbackChatFormat.Llama3,
    };
}
