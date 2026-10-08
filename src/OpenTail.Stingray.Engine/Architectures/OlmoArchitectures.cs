namespace OpenTail.Stingray.Engine;

// Part of OtherAdmittedArchitectures (partial), grouped by family. Descriptors are independent of one another.
internal static partial class OtherAdmittedArchitectures
{
    // olmoe — admitted 2026-08-08 on perplexity parity, NOT on token-for-token greedy parity,
    // which it does not achieve. On wikitext at a matched 2048-token context llama.cpp b8585
    // scores 7.4868 and this engine scores 7.3889 (1.3%). The greedy divergence is at a flat
    // position where the top five candidates span 1.55 logits, i.e. where a differently
    // quantised matmul reorders candidates. Evidence and the argument for accepting it:
    // docs/done/01-gguf-model-coverage-plan.md §1b. Note `olmo2` is deliberately NOT here — it
    // shares neither a fixture nor a receipt.
    public static readonly ArchitectureDescriptor Olmoe = new()
    {
        Id = "olmoe",
        UsesNeoxRope = true,
        ApplyModelSemantics = ctx => ctx.Baseline with { NormalizeMoeTopKWeights = false },
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; the MoE partial-offload row is a backend capability, not a family verification row.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // olmo2 — a THIRD residual pattern, distinct from both the ordinary pre-norm trunk and
    // gptneox/falcon's parallel residual: post-norm sandwiching. No attn_norm/ffn_norm tensor
    // exists in the GGUF at all — attention and FFN both read the RAW residual directly, and
    // the norm (RMSNorm, no bias) is applied to each sublayer's OUTPUT via attn_post_norm/
    // ffn_post_norm, immediately before the residual add (confirmed against
    // src/models/olmo2.cpp: x1 = x + PostNorm(Attn(x)); x2 = x1 + PostNorm(FFN(x1))).
    // ForwardPass's constructor leaves _attnNorm[i]/_ffnNorm[i] at their default (DataPtr
    // null) when absent — the same tensor-presence sentinel Apertus/GPT-NeoX already use for
    // "no ffn_gate" — and RunTrunk/PrefillCore's pre-norm steps copy the raw residual through
    // unmodified when that sentinel is set, instead of normalizing. The post-norm application
    // itself reuses Gemma 4's existing _postAttnNorm/_postFfwNorm mechanism unchanged (same
    // llama.cpp tensor names, LLM_TENSOR_ATTN_POST_NORM/FFN_POST_NORM) — generalized in
    // ModelGraph.cs to detect from tensor presence for any architecture, not just gemma4.
    // Because that mechanism was never wired into PrefillCore's batched loop (documented
    // there as Gemma-4-only in MoeBatchedPrefillSupported's doc comment), PrefillDispatch now
    // also falls back to sequential per-token Forward() for ANY post-norm model, not just
    // per-layer-head-dim ones — the same fallback pattern Gemma 4 already uses, just widened.
    // QK-norm reuses the OLMoE whole-vector-RMS fix unchanged (same convention, same code).
    // See Olmo2GreedyParityTests and docs/done/01-gguf-model-coverage-plan.md for the receipt.
    public static readonly ArchitectureDescriptor Olmo2 = new()
    {
        Id = "olmo2",
        UsesNeoxRope = true,
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // olmo (v1) — admitted 2026-08-09, FULL 24-of-24-token exact greedy match, bucket-1
    // (genuinely Apache-2.0, AI2). One genuinely new mechanism: LayerNorm with NEITHER a
    // learned scale NOR a bias at all — confirmed against olmo.cpp: every build_norm call
    // passes both weight and bias as NULL, and no attn_norm/ffn_norm/output_norm tensor
    // exists in the GGUF at all. A third norm shape distinct from weighted LayerNorm-with-bias
    // (gptneox/falcon/gpt2/starcoder2) and bias-less-but-still-weighted LayerNorm (cohere2). A
    // missing norm tensor already meant something specific here (OLMo2's "skip normalizing
    // here entirely, sandwich-normed on the output instead"), so this needed a genuine
    // arch-string check (ModelHyperparams.UsesUnweightedNorm) to disambiguate from that, not a
    // generalized tensor-presence rule. Added SimdKernels.PureLayerNorm (mean-subtract +
    // variance-normalize, no weight/bias parameter) and wired it into RunTrunk's three norm
    // points ahead of the existing null-DataPtr-means-skip check; PrefillCore's batched norm
    // steps were NOT taught this third mode, routed to the sequential path instead via a new
    // unweightedNormUnsupported flag in PrefillDispatch's fallback gate (same pattern
    // OLMo2/cohere2/Gemma-4 already use for their own PrefillCore gaps). Everything else
    // (plain MHA, standard interleaved RoPE, SiLU-gated FFN, tied embeddings) was already
    // generic. Full exact match on the first real attempt. See OlmoGreedyParityTests and
    // docs/done/01-gguf-model-coverage-plan.md §1s.
    public static readonly ArchitectureDescriptor Olmo = new()
    {
        Id = "olmo",
        // olmo.cpp: build_norm with NULL weight AND bias -> normalise anyway, with no parameters (OLMo2 instead skips pre-norm).
        ApplyModelSemantics = ctx => ctx.Baseline with { UsesUnweightedNorm = true },
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; the MoE partial-offload row is a backend capability, not a family verification row.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
