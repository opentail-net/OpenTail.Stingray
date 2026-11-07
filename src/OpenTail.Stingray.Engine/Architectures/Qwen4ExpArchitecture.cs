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
        RefusalReason = "Structurally complete alpha forward pass (GatedResidual HC, FP32 GDN, 512-expert MoE, n-gram PLE, QSA IMRoPE + indexer); real-weight run 2026-11-07 against unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S -- fixed two wrong tensor names, a dtype crash (Q8_0 dequant), the GDN output gate (sigmoid not SiLU), and a missing SiLU on the GDN conv1d output (confirmed against both llama.cpp qwen4exp.cpp:1160 and TensorSharp's own doc comment; found via real-weight layer-by-layer sum diffing against a local llama.cpp build's eval-callback dump on the identical checkpoint+prompt). llama.cpp's own build on this exact checkpoint produces coherent output (\"The capital of France is Paris.\"), proving the remaining divergence is a real bug in this port, not the checkpoint. After the SiLU fix, layer-0's post-GDN residual sum moved from 19.17 (vs llama's 10.16, ~89% off) to 12.96 (~27% off) -- real, measured progress, not yet convergence. Final logits are still off by roughly 10x in magnitude, so at least one more bug remains, most likely in the FFN/MoE path (layer-0's full post-FFN residual barely moved despite the GDN-side fix). Not narrowed down further yet.",
    };
}
