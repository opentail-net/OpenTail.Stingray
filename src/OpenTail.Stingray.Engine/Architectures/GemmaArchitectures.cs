namespace OpenTail.Stingray.Engine;

internal static class GemmaArchitectures
{
    public static readonly ArchitectureDescriptor Gemma = new()
    {
        Id = "gemma",
        UsesNeoxRope = true,
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // Shares the evidence comment and receipt for 'gemma'.
    public static readonly ArchitectureDescriptor Gemma2 = new()
    {
        Id = "gemma2",
        UsesNeoxRope = true,
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // Shares the evidence comment and receipt for 'gemma'.
    public static readonly ArchitectureDescriptor Gemma3 = new()
    {
        Id = "gemma3",
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
