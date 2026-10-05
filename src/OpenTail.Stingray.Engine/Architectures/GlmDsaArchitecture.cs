namespace OpenTail.Stingray.Engine;

internal static class GlmDsaArchitecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "glm-dsa",
        Status = AdmissionStatus.NotAdmitted,
        EvidenceDoc = "docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md",
        // Ported, not verified (CLAUDE.md rule 14); the table in EvidenceDoc is the internal record.
        RefusalReason = "Alpha forward pass ported 2026-10-03 from llama.cpp glm-dsa.cpp (MLA absorption, DSA lightning indexer); no real checkpoint has been run, so no parity receipt exists.",
    };
}
