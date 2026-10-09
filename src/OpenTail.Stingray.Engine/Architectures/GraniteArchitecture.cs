namespace OpenTail.Stingray.Engine;

internal static class GraniteArchitecture
{
    // Admitted 2026-08-08 on a FULL 24-token exact greedy match against llama.cpp (stronger than the
    // olmoe receipt, which only reaches a 2-token prefix). Needs a "scale trio" + attention-scale
    // override beyond the plain llama trunk, read from GGUF metadata (ModelHyperparams.ResidualScale /
    // AttentionScaleOverride / LogitScale, generalized EmbeddingScale). See Goldens/granite.golden.json and
    // docs/done/01-gguf-model-coverage-plan.md §1d for the receipt and for what is NOT yet wired
    // (TurboQuant prefill, continuous-batching admission, CUDA/Vulkan). MiniCPM (not MiniCPM3, which is
    // MLA) shares this exact graph in llama.cpp and reuses the same implementation.
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "granite",
        ChatProtocolId = "granite",
        ApplyModelSemantics = ctx => FamilyModelSemantics.Granite(ctx, miniCpm: false),
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
        // IBM's embedded Jinja injects a default system prompt and indents user content, which makes
        // vision decoders emit <|end_of_text|> at token 0; use llama.cpp's canonical GRANITE_4_0 layout.
        FallbackChat = FallbackChatFormat.Granite,
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
