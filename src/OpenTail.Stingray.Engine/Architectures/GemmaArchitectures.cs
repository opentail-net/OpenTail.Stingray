namespace OpenTail.Stingray.Engine;

internal static class GemmaArchitectures
{
    public static readonly ArchitectureDescriptor Gemma = new()
    {
        Id = "gemma",
        ChatProtocolId = "gemma",
        UsesNeoxRope = true,
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // gemma2 — verified 2026-10-11 against golden gemma2.golden.json: llama-server 10306 (6b5c2efb4), sha256 e0aee85060f168f0f2d8473d7ea41ce2f3230c1bc1374847505ea599288a7787, NearTie. CPU only.
    public static readonly ArchitectureDescriptor Gemma2 = new()
    {
        Id = "gemma2",
        ChatProtocolId = "gemma",
        UsesNeoxRope = true,
        ApplyModelSemantics = FamilyModelSemantics.Gemma2,
        // Attention-logit softcapping (50) is implemented on the CPU attention paths only; the Vulkan and CUDA attention kernels do not apply it.
        SupportedBackends = SupportedBackends.Cpu,
        BackendLimitation = "Gemma 2's attention-logit softcapping is implemented on the CPU attention paths only (2026-10-11).",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Gemma 2 text (`gemma2`)",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // Shares the evidence comment and receipt for 'gemma'.
    public static readonly ArchitectureDescriptor Gemma3 = new()
    {
        Id = "gemma3",
        ChatProtocolId = "gemma",
        UsesNeoxRope = true,
        ApplyModelSemantics = FamilyModelSemantics.Gemma3,
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Gemma 3 text (`gemma3`)",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // Shares the evidence comment and receipt for 'gemma'.
    public static readonly ArchitectureDescriptor Gemma3n = new()
    {
        Id = "gemma3n",
        UsesNeoxRope = true,
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
