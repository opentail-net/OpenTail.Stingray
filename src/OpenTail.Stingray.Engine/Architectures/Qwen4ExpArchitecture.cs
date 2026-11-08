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
        CreateForwardPass = BuildForwardPass,
    };

    /// <summary>Loads the qwen4exp trunk for the server. Local investigation only until admission (CLAUDE.md rule 14).</summary>
    private static OpenTail.Stingray.Core.IForwardPass BuildForwardPass(ArchitectureLoadContext ctx)
    {
        var model = ctx.Probe.Gguf ?? throw new InvalidOperationException("qwen4exp needs a GGUF model.");
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        ctx.TrackDisposable(pass);
        return pass;
    }
}
