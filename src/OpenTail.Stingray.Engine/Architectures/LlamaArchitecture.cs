namespace OpenTail.Stingray.Engine;

internal static class LlamaArchitecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "llama",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        FallbackChat = FallbackChatFormat.Llama3,
    };
}
