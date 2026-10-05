namespace OpenTail.Stingray.Engine;

internal static class DeepSeek32Architecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "deepseek32",
        Status = AdmissionStatus.NotAdmitted,
        EvidenceDoc = "docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md",
        // Ported, not verified (CLAUDE.md rule 14); the table in EvidenceDoc is the internal record.
        RefusalReason = "Structurally complete alpha MLA + DSA forward pass, never run on a real DeepSeek-V3.2 GGUF; Hadamard rotation on the indexer and MTP are not implemented.",
    };
}
