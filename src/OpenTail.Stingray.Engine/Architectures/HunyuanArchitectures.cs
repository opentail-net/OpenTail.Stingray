namespace OpenTail.Stingray.Engine;

// Part of OtherAdmittedArchitectures (partial), grouped by family. Descriptors are independent of one another.
internal static partial class OtherAdmittedArchitectures
{
    // hunyuan-dense — admitted 2026-08-09 on a FULL 24-of-24-token exact greedy match
    // (deterministic, though the reference itself is a degenerate repeated-token loop, since
    // the checkpoint is Instruct-tuned and the receipt used a bare, un-templated prompt — still
    // valid token-for-token parity evidence, just not a coherent completion). NOT
    // llama_model_hunyuan_vl (the actual multimodal class) — hunyuan-dense INHERITS its
    // load_arch_hparams/load_arch_tensors/graph wholesale from hunyuan-vl.cpp with no override,
    // confirmed by reading models.h before writing any code, so this receipt's evidence covers
    // both. Ordinary pre-norm RMSNorm trunk, standard GQA (head_count=16, head_count_kv=8),
    // SiLU-gated FFN, no biases anywhere, no MoE, no MRoPE for the text-only dense checkpoint
    // (rope.dimension_sections absent) — none of that is new. The one genuinely new mechanism:
    // weighted QK-norm (a learned per-head RMSNorm, attn_q_norm/attn_k_norm, shape [128] =
    // headDim, not per-channel) applied AFTER RoPE rather than before — confirmed directly
    // against hunyuan-vl.cpp's graph (rope first, then build_norm on the already-rotated Q/K).
    // This engine had two existing QK-norm timings (Qwen3: weighted, before RoPE; Llama-4:
    // unweighted L2, after RoPE) but no "weighted, after RoPE" combination — added
    // ModelHyperparams.QkNormAfterRope and wired it into PrefillCore and RunTrunk (the two
    // paths a plain Prefill()/Forward() receipt exercises; PrefillCoreTq/BatchVerify/
    // BatchForwardMulti's own QK-norm blocks are untouched and would need the identical fix if
    // hunyuan-dense is ever run through those paths). Also needed a new pre-tokenizer cascade:
    // this checkpoint declares tokenizer.ggml.pre=hunyuan-dense, a DISTINCT llama.cpp pre-type
    // (LLAMA_VOCAB_PRE_TYPE_HUNYUAN_DENSE) from the plain "hunyuan" already in this engine's
    // table (which is actually the Qwen-2 cascade) — confirmed via llama-vocab.cpp; added as a
    // 3-stage cascade (PreTokenizerPatterns.DigitRun3/Cjk/HunyuanDenseTail) shared with the
    // deepseek3-llm/joyai-llm pre-types llama.cpp folds onto the same case, verified by
    // checking tokenizer.Encode against llama-tokenize before writing any forward-pass code.
    //
    // NO AUTOMATED TEST FOR THIS ARCHITECTURE, FOR LICENCE REASONS. Checkpoint:
    // `tencent/Hunyuan-0.5B-Instruct` (bartowski GGUF, Q8_0, 578 MB), Tencent Hunyuan Community
    // License (MAU threshold, territorial exclusions) — not MIT/Apache-2.0/BSD/MPL. Transient
    // local download, never vendored, deleted immediately after this receipt.
    //
    // Verification evidence (2026-08-09, hunyuan-0.5b-instruct Q8_0, llama.cpp b8585-cad2d3884):
    // prompt "The capital of France is" -> ids [628, 6801, 279, 9391, 316] (confirms the new
    // pre-tokenizer cascade matches the reference exactly). Raw (no chat template) greedy
    // completion degenerates to token 478 repeated 24 times — this engine reproduces that
    // EXACT degenerate sequence for all 24 tokens, a full deterministic match.
    //
    // DO NOT MODIFY THE QkNormAfterRope TIMING FOR hunyuan-dense WITHOUT GOOD REASON — there is
    // no regression test to catch a mistake, and this receipt is currently the only thing in
    // the codebase exercising that combination.
    public static readonly ArchitectureDescriptor Hunyuandense = new()
    {
        Id = "hunyuan-dense",
        UsesNeoxRope = true,
        // Weighted QK-norm applied AFTER RoPE (hunyuan-vl.cpp), unlike every other weighted-QK-norm family.
        ApplyModelSemantics = ctx => ctx.Baseline with { QkNormAfterRope = true },
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // hunyuan-moe (Hunyuan-A13B-Instruct) — admitted 2026-10-04 on a real checkpoint (DevQuasar Q3_K_S only), CPU:
    // HunyuanMoeGreedyParityTests teacher-forced against llama-server (same GGUF): all 39 confident positions match
    // (27 of 32 on a 196-token prompt, 12 of 22 on a 5-token one), 0 near-tie differences. Needed in ModelGraph: QK-norm
    // after RoPE (as hunyuan-dense) and renormalised top-k expert weights. Not on any GPU path yet.
    public static readonly ArchitectureDescriptor Hunyuanmoe = new()
    {
        Id = "hunyuan-moe",
        UsesNeoxRope = true,
        // QK-norm after RoPE (HF HunYuanAttention); norm_w = true, softmax gating.
        ApplyModelSemantics = ctx => ctx.Baseline with { QkNormAfterRope = true, NormalizeMoeTopKWeights = true },
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Hunyuan-A13B-Instruct (`hunyuan-moe`)",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
