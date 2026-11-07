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
        RefusalReason = "Structurally complete alpha forward pass (GatedResidual HC, FP32 GDN, 512-expert MoE, n-gram PLE, QSA IMRoPE + indexer); real-weight run 2026-11-07 against unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S -- fixed two wrong tensor names and a crash (AsFloatSpan cast quantized Q8_0 weights as Float32; now dequantizes via Dequantize.ToFloat32, cached per tensor), and prefill+decode now run to completion with finite logits (Qwen4ExpRealWeightsSmokeTests). That is NOT evidence of correctness: a 12-token manual greedy continuation of \"The capital of France is\" produced incoherent mixed-script garbage, not plausible text, so a real numerical bug remains somewhere in this alpha pass (GDN, QSA, PLE, MoE routing, or the HC mixing math are all independently flagged unverified in the source -- not narrowed down). MTP and vision remain deferred.",
    };
}
