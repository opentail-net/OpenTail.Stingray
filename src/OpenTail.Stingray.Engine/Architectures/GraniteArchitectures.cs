namespace OpenTail.Stingray.Engine;

internal static class GraniteArchitectures
{
    // granitehybrid — IBM Granite 4.0-H (Mamba-2 + NoPE attention hybrid, llama.cpp granite-hybrid.cpp +
    // mamba-base.cpp build_mamba2_layer). ADMITTED 2026-09-27 with the new Mamba-2 mixer (ForwardPass.Mamba2.cs,
    // CPU, token-by-token). Evidence, wikitext second-half [1024,+) PPL at -c 2048 vs llama-perplexity --chunks 1:
    // granite-4.0-h-350m Q8_0 17.9578 vs 17.9258; granite-4.0-h-1b Q8_0 8.7833 vs 8.7563 (the recurrence moves this
    // metric ~0.3% with summation order alone: the scalar scan gave 17.9003 / 8.7639). Tokenisation and the
    // first greedy tokens match llama-server; a longer greedy run diverges at a near-tie (" explained" 0.140 vs
    // " showed" 0.136 in llama.cpp).
    public static readonly ArchitectureDescriptor Granitehybrid = new()
    {
        Id = "granitehybrid",
        ChatProtocolId = "granite",
        ForwardPassFamily = ForwardPassFamily.HybridGdn, // the hybrid-recurrent factory; it delegates Mamba-2 layouts to the dense passes
        ApplyModelSemantics = ctx => FamilyModelSemantics.Granite(ctx, miniCpm: false, hybrid: true) with { NormalizeMoeTopKWeights = true },
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "IBM Granite 4.0-H (`granitehybrid`)",
        EvidenceDoc = "docs/STATUS.md",
        SupportsContinuousBatching = false,
        CreateForwardPass = CommonForwardPassFactory.CreateHybridGdn,
    };

    // granitemoe — admitted 2026-08-09, FULL 24-of-24-token exact greedy match, bucket-1
    // (genuinely Apache-2.0), essentially a free admission. llama_model_granite_moe::graph is
    // a type alias for llama_model_granite::graph (confirmed in models.h before writing any
    // code) — the SAME graph as dense Granite (already admitted), which already branches on
    // n_expert==0 internally. This engine's generic MoE dispatch and the Granite-family scale
    // block (ResidualScale/EmbeddingScale/LogitScale, isGraniteFamily in ModelGraph.cs) already
    // explicitly checked arch=="granitemoe" from when the dense receipt was built. Standard
    // softmax gating, no shared expert, standard GQA — every mechanism this checkpoint
    // exercises was already correct on the first real attempt (the only failure along the way
    // was a wrong test assertion, not an engine defect — LogitScale already carries the
    // reciprocal of the raw metadata value, documented but momentarily forgotten while writing
    // the test). See GraniteMoeGreedyParityTests and docs/done/01-gguf-model-coverage-plan.md §1r.
    public static readonly ArchitectureDescriptor Granitemoe = new()
    {
        Id = "granitemoe",
        ChatProtocolId = "granite",
        ApplyModelSemantics = ctx => FamilyModelSemantics.Granite(ctx, miniCpm: false) with { NormalizeMoeTopKWeights = true },
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
