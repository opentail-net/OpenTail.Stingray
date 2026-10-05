namespace OpenTail.Stingray.Engine;

internal static class DiffusionGemmaArchitecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "diffusion-gemma",
        Status = AdmissionStatus.NotAdmitted,
        EvidenceDoc = "docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md",
        // Ported, not verified (CLAUDE.md rule 14); the table in EvidenceDoc is the internal record.
        RefusalReason = "Block text-diffusion model, not autoregressive: it must run through DiffusionGemmaPipeline, never Forward(token). The port also does not yet implement the real checkpoint's Gemma-4 MoE layout.",
    };
}
