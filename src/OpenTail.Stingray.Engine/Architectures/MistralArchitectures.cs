namespace OpenTail.Stingray.Engine;

// Part of OtherAdmittedArchitectures (partial), grouped by family. Descriptors are independent of one another.
internal static partial class OtherAdmittedArchitectures
{
    public static readonly ArchitectureDescriptor Mistral3 = new()
    {
        Id = "mistral3",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        RecognizeRelabelledFile = (arch, probe) =>
            arch.Equals("llama", StringComparison.OrdinalIgnoreCase) &&
            (probe.GetMetadataString("general.name")?.Contains("mistral-3", StringComparison.OrdinalIgnoreCase) == true ||
             probe.GetMetadataString("general.name")?.Contains("mistral3", StringComparison.OrdinalIgnoreCase) == true),
        RelabelledFileDescription = "Mistral 3 models labeled with general.architecture 'llama'",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    public static readonly ArchitectureDescriptor Ministral = new()
    {
        Id = "ministral",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
