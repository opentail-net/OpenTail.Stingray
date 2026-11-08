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
        RefusalReason = "Structurally complete alpha forward pass (GatedResidual HC, FP32 GDN, 512-expert MoE, n-gram PLE, QSA IMRoPE + indexer); real-weight run 2026-11-07 against unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S -- fixed two wrong tensor names, a dtype crash (Q8_0 dequant), the ssm_conv1d weight transpose (stored [channel][kernel]), the GDN output gate (sigmoid not SiLU), a missing SiLU on the GDN conv1d output, and a missing per-head L2 norm on Q/K before the recurrence (all three confirmed against llama.cpp qwen4exp.cpp/models.h and, for the first two, TensorSharp's doc comments too; found via real-weight layer-by-layer sum diffing against a local llama.cpp build's eval-callback dump on the identical checkpoint+prompt). llama.cpp's own build on this exact checkpoint produces coherent output (\"The capital of France is Paris.\"), proving the remaining divergence is a real bug in this port, not the checkpoint. Layer-0's post-GDN-and-FFN residual sum (l_last-0) has moved 73% off -> 77% off (noise) -> 46% off across these fixes, vs llama's reference value -- real, measured, monotonic-ish progress on the earlier layers, not yet convergence across all 48. Final logits are still off (wrong order of magnitude), so at least one more bug remains, not yet isolated to a specific component. 2026-11-07, later: checked the FFN-stage injection gate specifically (the component most sensitive to upstream error, since it is a per-stream dot product with no cross-stream averaging unlike `mixed`) -- the hc_ffn_inject weight tensor loads the full expected length (40960) with statistically sane values (no corruption, no truncation), but the resulting per-stream gate values diverge from llama's reference well beyond the already-known upstream GDN imprecision (one stream even flips sign). Inconclusive: could be a real remaining bug in this specific path, or compounding/amplified error from GDN math that is closer but not yet fully converged. Not resolved this round; the full-stack `l_last` trace (layers 0-47) still grows markedly faster than llama's reference throughout, so convergence is not close.",
    };
}
