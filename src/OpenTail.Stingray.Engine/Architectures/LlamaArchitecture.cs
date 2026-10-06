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
        DetectFromProbe = probe =>
            probe.HasTensor("token_embd.weight") &&
            probe.HasTensor("blk.0.attn_q.weight") &&
            !probe.HasTensor("blk.0.ssm_in.weight"),
    };
}
