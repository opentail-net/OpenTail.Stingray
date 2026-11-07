namespace OpenTail.Stingray.Engine;

internal static class Glm5NextArchitecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "glm5next",
        Status = AdmissionStatus.NotAdmitted,
        EvidenceDoc = "docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md",
        // Ported, not verified (CLAUDE.md rule 14); the table in EvidenceDoc is the internal record.
        RefusalReason = "Hybrid KDA + MLA trunk ported 2026-10-03 (mHC, K-pool indexer, 288-expert MoE); smoke tests only, K-pool sparse selection missing, no numeric oracle run. Also carries the same unguarded Float32-cast risk confirmed on a real checkpoint for qwen4exp/deepseek4/deepseek41 (2026-11-07) -- this file's norm/scale/conv weight casts are scattered across many individual call sites rather than one helper, so the same fix wasn't ported here yet.",
    };
}
