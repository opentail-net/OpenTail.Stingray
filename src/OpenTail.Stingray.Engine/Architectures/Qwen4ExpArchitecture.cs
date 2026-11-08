namespace OpenTail.Stingray.Engine;

internal static class Qwen4ExpArchitecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "qwen4exp",
        Status = AdmissionStatus.NotAdmitted,
        EvidenceDoc = "docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md",
        // Ported, not verified (CLAUDE.md rule 14); the table in EvidenceDoc is the internal record.
        // Revised 2026-10-04 (EvidenceDoc): full structural implementation landed (GatedResidual HC,
        // FP32 GDN, fused/separate-tensor MoE, 16-head n-gram PLE with disk mmap, QSA four-section
        // IMRoPE, indexer cache, K-pool sparse selection) -- this comment previously described an
        // earlier, incomplete state and was stale.
        RefusalReason = "Ported, not verified (CLAUDE.md rule 14). Structurally complete forward pass (GatedResidual HC, FP32 GDN, 512-expert MoE, n-gram PLE, QSA IMRoPE + indexer, K-pool selection). On the UD-IQ1_S checkpoint, the text-path measurements sit inside the noise envelope (2026-11-08); the image-path logit comparison does not (TV 0.178, logit gap 0.81), and the FFN injection gate (Phase 0.3) is open. Status: docs/2-coverage/2026-11-08-qwen38-remaining-work-plan.md. Admission needs Phases 0-7 closed or waived by the owner (D4).",
    };
}
