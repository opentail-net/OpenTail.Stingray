namespace OpenTail.Stingray.Engine;

internal static class Qwen4ExpArchitecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "qwen4exp",
        Status = AdmissionStatus.NotAdmitted,
        EvidenceDoc = "docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md",
        // Ported, not verified (CLAUDE.md rule 14); the table in EvidenceDoc is the internal record.
        RefusalReason = "Alpha forward pass ported 2026-10-03 (GDN+QSA, hyper-connections, 512-expert MoE); PLE n-gram table, QSA RoPE and indexer missing, and the 72.5 GB checkpoint does not fit this machine's RAM.",
    };
}
