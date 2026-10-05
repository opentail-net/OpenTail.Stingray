namespace OpenTail.Stingray.Engine;

internal static class Glm5NextArchitecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "glm5next",
        Status = AdmissionStatus.NotAdmitted,
        EvidenceDoc = "docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md",
        // Ported, not verified (CLAUDE.md rule 14); the table in EvidenceDoc is the internal record.
        RefusalReason = "Hybrid KDA + MLA trunk ported 2026-10-03 (mHC, K-pool indexer, 288-expert MoE); smoke tests only, K-pool sparse selection missing, no numeric oracle run.",
    };
}
