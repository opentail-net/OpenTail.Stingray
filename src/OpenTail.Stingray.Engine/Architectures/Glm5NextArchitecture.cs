namespace OpenTail.Stingray.Engine;

internal static class Glm5NextArchitecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "glm5next",
        // Confirmed 2026-11-07 against a real checkpoint (unsloth/GLM-5.3-Flash-GGUF): its
        // general.architecture is "glm5-next" (hyphenated), not "glm5next". ArchitectureRegistry.Find
        // does an exact string match with no normalization, so without this alias the real GGUF
        // would never match this descriptor at all -- it would hit the generic "architecture not
        // supported" message instead of this file's specific (and more informative) RefusalReason.
        Aliases = ["glm5-next"],
        Status = AdmissionStatus.NotAdmitted,
        EvidenceDoc = "docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md",
        // Ported, not verified (CLAUDE.md rule 14); the table in EvidenceDoc is the internal record.
        RefusalReason = "Hybrid KDA + MLA trunk ported 2026-10-03 (mHC, K-pool indexer, 288-expert MoE); K-pool sparse selection missing, no numeric oracle run. 2026-11-07: real-weight load attempted against unsloth/GLM-5.3-Flash-GGUF UD-IQ1_S -- fixed the real architecture-string mismatch (\"glm5-next\" vs this id) and four tensor-naming bugs (ssm_a's missing .weight suffix, indexer.* dot-vs-underscore naming, indexer_kpool_gate/_ape's wrong name). Prefill on a short prompt succeeds with finite logits; the scattered Float32-cast risk flagged earlier did not materialize for this checkpoint (its norm/scale/conv tensors are genuinely F32). A manual 12-token greedy continuation produced degenerate repeated-token output (\"!!!!!!!!!!!!!\"), so a real decode-step bug remains, not narrowed down. Not a parity receipt: no reference output compared, and K-pool selection still blocks any context longer than indexer.top_k.",
    };
}
