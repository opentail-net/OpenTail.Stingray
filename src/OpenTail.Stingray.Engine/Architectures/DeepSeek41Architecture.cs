namespace OpenTail.Stingray.Engine;

internal static class DeepSeek41Architecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "deepseek41",
        Status = AdmissionStatus.NotAdmitted,
        EvidenceDoc = "docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md",
        // Ported, not verified (CLAUDE.md rule 14); the table in EvidenceDoc is the internal record.
        RefusalReason = "Partial port (raw-attention trunk only; refuses compression ratios 1/2 and YaRN, no indexer, F32-only Engram); checkpoints exceed 335 GB and cannot be verified here.",
    };
}
