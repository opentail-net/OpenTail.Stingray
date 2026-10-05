namespace OpenTail.Stingray.Engine;

internal static class Gemma4Architecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "gemma4",
        Status = AdmissionStatus.Admitted,
        EvidenceDoc = "docs/STATUS.md",
        // Not trained for the engine's <|channel> thought split: enabling it makes the model ramble
        // and go out-of-distribution on multimodal input.
        ThinkingDefaultOff = true,
    };
}
