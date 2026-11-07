namespace OpenTail.Stingray.Engine;

internal static class GlmDsaArchitecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "glm-dsa",
        Status = AdmissionStatus.NotAdmitted,
        EvidenceDoc = "docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md",
        // Ported, not verified (CLAUDE.md rule 14); the table in EvidenceDoc is the internal record.
        RefusalReason = "Alpha forward pass ported 2026-10-03 from llama.cpp glm-dsa.cpp (MLA absorption, DSA lightning indexer); no real checkpoint has been run, so no parity receipt exists. Also carries the same unguarded Float32-cast risk confirmed on a real checkpoint for qwen4exp/deepseek4/deepseek41 (2026-11-07) -- this file's norm/scale/conv weight casts are scattered across many individual call sites rather than one helper, so the same fix wasn't ported here yet.",
    };
}
