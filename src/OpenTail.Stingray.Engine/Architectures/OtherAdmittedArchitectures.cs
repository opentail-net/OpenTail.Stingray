namespace OpenTail.Stingray.Engine;

// Admitted architectures that do not belong to a larger family file. The class is partial: see the *Architectures.cs files beside it.
internal static partial class OtherAdmittedArchitectures
{
    // smollm3 — one twist over the plain llama trunk: NoPE every 4th layer, gated the same way
    // as llama4's noRopeStep. See tests/OpenTail.Stingray.Tests.ForwardPass/Goldens/smollm3.golden.json for the full 24-token greedy receipt.
    public static readonly ArchitectureDescriptor Smollm3 = new()
    {
        Id = "smollm3",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
        ApplyModelSemantics = ctx => ctx.Baseline with { NoRopeLayerStep = 4 },
    };

    // apertus — admitted 2026-08-08 on an 11-token EXACT prefix match (one full sentence)
    // against llama.cpp, diverging afterward into a different but still coherent, on-topic
    // completion (not degenerate output). The first "new-kernel" architecture admitted this
    // session: no ffn_gate tensor at all (plain up -> xIELU -> down, ModelHyperparams.Xielu*,
    // SimdKernels.XieluInPlace), detected from tensor inventory rather than architecture
    // string. See ApertusGreedyParityTests and docs/done/01-gguf-model-coverage-plan.md §1f for the
    // receipt, including a real defect found and fixed in the xIELU parameter transform
    // (GGUF stores pre-softplus values; llama.cpp's ggml_xielu() wrapper — not the compute
    // kernel — applies softplus before use, easy to miss by reading only the kernel).
    public static readonly ArchitectureDescriptor Apertus = new()
    {
        Id = "apertus",
        UsesNeoxRope = true,
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // falcon (7B only — 40B's second attn_norm_2 tensor is NOT implemented, no small 40B
    // checkpoint to validate against) — reuses every gptneox mechanism (LayerNorm, biased
    // non-gated GELU FFN [though Falcon carries no biases at all], fused attn_qkv,
    // UseParallelResidual's 3-way sum) plus one new wrinkle: Falcon-7B has NO separate
    // ffn_norm tensor at all — attention and FFN read the SAME LayerNorm output (confirmed
    // against src/models/falcon.cpp: "use the attn norm, not the result"). ForwardPass's
    // constructor falls _ffnNorm/_bFfnNorm back to _attnNorm/_bAttnNorm's own TensorRef/
    // pointer when blk.*.ffn_norm.{weight,bias} is absent — Dispose() guards the aliased
    // bias pointer so it isn't double-freed. use_parallel_residual is never a metadata key
    // for this arch (llama.cpp hardcodes it in the graph), so ModelGraph.cs hardcodes it too
    // for arch=="falcon" rather than reading a key that doesn't exist. Also exercises MQA
    // (head_count=71, head_count_kv=1) through the existing GQA-parametrized fused-QKV
    // split for the first time on this profile. See FalconGreedyParityTests and
    // docs/done/01-gguf-model-coverage-plan.md for the receipt.
    public static readonly ArchitectureDescriptor Falcon = new()
    {
        Id = "falcon",
        UsesNeoxRope = true,
        ApplyModelSemantics = ctx => ctx.Baseline with { UseParallelResidual = true },
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // cohere2 (Command-R7B) — ADMITTED 2026-08-09, 1-token exact match plus a documented
    // near-tie, bucket-2. Reuses gptneox/falcon's shared-attn/ffn-norm fallback (no separate
    // ffn_norm tensor) and UseParallelResidual's 3-way sum, but needed THREE genuinely new
    // mechanisms, all confirmed against examples/llama.cpp/llama.cpp/src/models/cohere2.cpp
    // before writing any code: (1) LayerNorm WITHOUT a learned bias — SimdKernels.LayerNorm's
    // bias param is now null-safe (skips the bias-add step), and the new
    // ModelHyperparams.UsesLayerNorm decouples "use LayerNorm math" from HasNormBias ("has a
    // bias tensor"), since a weight-only norm tensor looks identical on disk whether the
    // architecture means RMSNorm or bias-less LayerNorm — this is an arch-string fact, not a
    // tensor-presence one. (2) Generic (non-Gemma4-gated) sliding-window attention — 3 local +
    // 1 global layers (swaPeriod=4, hardcoded default even when the metadata key is absent),
    // computed in ModelGraph.cs outside the isGemma4 block via the exact formula in
    // llama-hparams.cpp's set_swa_pattern (dense_first=false: is_swa[il] = il%period <
    // period-1) — NOT Gemma 4's literal-bool-array convention, since cohere2's own metadata
    // key is a plain period scalar. (3) RoPE applied ONLY on SWA layers, none at all on global
    // ones (ModelHyperparams.RopeOnlySwaLayers) — cohere2.cpp's attention block has no `else`
    // branch on its `if (is_swa)` rope application, the opposite selection rule from
    // Llama-4/SmolLM3's period-based NoRopeLayerStep. Also: PrefillCoreAttention has NO
    // windowSize parameter at all (only ever needed by Gemma 4, which never reaches it — see
    // perLayerHdUnsupported), so PrefillDispatch now also falls back to sequential Forward()
    // for any SWA model without per-layer head dims, reusing RunTrunk's Attention() call
    // (proven correct by every Gemma 4 receipt) instead of teaching PrefillCore SWA masking.
    // logit_scale is read with the OPPOSITE convention from Granite's (direct multiply, not
    // reciprocal — cohere2.cpp does ggml_scale(cur, f_logit_scale) unconditionally, not
    // 1/f_logit_scale).
    //
    // NO AUTOMATED TEST FOR THIS ARCHITECTURE, FOR LICENCE REASONS. Checkpoint:
    // `CohereLabs/c4ai-command-r7b-12-2024` (bartowski GGUF, Q4_K_M), CC-BY-NC-4.0 — not
    // MIT/Apache-2.0/BSD/MPL. Transient local download, never vendored, deleted immediately
    // after this receipt.
    //
    // Verification evidence (2026-08-09, c4ai-command-r7b-12-2024 Q4_K_M, llama.cpp
    // b8585-cad2d3884): prompt "The capital of France is" -> ids [2162, 7784, 1719, 5334, 1801].
    // First generated token matches exactly (id 1690). Second diverges: this engine picks
    // token 19 (",", logit 13.7218) where llama.cpp's reference implies 1671 (" a", this
    // engine's own logit for it: 13.6563) — a 0.0655-logit gap, tighter than every other
    // near-tie accepted this session, on a Q4_K_M checkpoint. Ruled out before accepting:
    // re-ran with STINGRAY_CPU_PREFILL_Q8=0 (same result — not an int8-prefill artifact);
    // confirmed this checkpoint declares no rope_freqs.weight tensor (not a missing-mechanism
    // gap); confirmed via list-tensors that blk.*.attn_norm has no .bias tensor and no
    // separate ffn_norm tensor (matches the bias-less/shared-norm design exactly, not a
    // loading defect). Reads as ordinary Q4_K accumulation-order sensitivity at a
    // closely-contested position, the same category of evidence the OLMoE/Apertus/GPT-NeoX
    // receipts were accepted on — measured directly via a top-5 logit dump before accepting,
    // not assumed.
    //
    // DO NOT MODIFY THIS ARCHITECTURE'S CODE PATH WITHOUT GOOD REASON — there is no regression
    // test to catch a mistake. In particular the SWA/RopeOnlySwaLayers/UsesLayerNorm additions
    // above are new, cohere2-only code paths nothing else in the codebase exercises.
    public static readonly ArchitectureDescriptor Cohere2 = new()
    {
        Id = "cohere2",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
        ApplyModelSemantics = Cohere2Semantics,
    };

    // stablelm — admitted 2026-08-09. Smallest code change of any new-kernel architecture this
    // session: LayerNorm-with-bias, non-gated-FFN plumbing, and NEOX partial rope were all
    // already generic (built for gptneox/falcon/glm4), so nothing new was needed for any of
    // those. The one real finding: this checkpoint's GGUF carries a stale
    // `stablelm.use_parallel_residual=true` metadata key that stablelm.cpp's graph builder
    // never actually reads — the real sequential-vs-parallel choice is made by branching on
    // whether the per-layer `ffn_norm` TENSOR exists, and this checkpoint has real ffn_norm
    // tensors on every layer (i.e. genuinely sequential, despite the metadata saying true).
    // Reusing the pre-existing GetBool(metadata, "{arch}.use_parallel_residual") fallback
    // (written for gptneox, where the key genuinely is consulted) would have silently taken
    // the wrong branch. Fixed in ModelGraph.cs: stablelm now derives UseParallelResidual from
    // blk.0.ffn_norm.weight tensor presence instead of the metadata key.
    //
    // NO AUTOMATED TEST FOR THIS ARCHITECTURE, FOR LICENCE REASONS. Checkpoint:
    // `stabilityai/stablelm-2-zephyr-1_6b` (afrideva GGUF, Q8_0), Stability AI "other" license
    // (non-commercial, gated) — not MIT/Apache-2.0/BSD/MPL. Transient local download, never
    // vendored, deleted immediately after this receipt.
    //
    // Verification evidence (2026-08-09, stablelm-2-zephyr-1_6b Q8_0, llama.cpp
    // b8585-cad2d3884): prompt "The capital of France is" -> ids [791, 6864, 315, 9822, 374].
    // First 4 generated tokens match exactly (" Paris, and it"). Diverges at position 4: this
    // engine picks token 596 ("'s", logit 24.6648) where llama.cpp's reference implies 374
    // (" is", this engine's own logit: 24.6441) — a 0.0207-logit gap, on a near-lossless Q8_0
    // checkpoint (every other near-tie accepted this session was Q4_K/Q4_K_M). Ruled out
    // before accepting: re-ran with STINGRAY_CPU_PREFILL_Q8=0 (identical result); confirmed
    // via list-tensors that no rope_freqs.weight or attn_q_norm/attn_k_norm tensor is silently
    // missing; confirmed the post-divergence continuation stays fully coherent English, not
    // degenerate. Reads as ordinary Q8_0 accumulation-order sensitivity at a closely-contested
    // position, the same evidentiary category as every other near-tie receipt this session.
    //
    // DO NOT MODIFY THE UseParallelResidual TENSOR-PRESENCE BRANCH FOR stablelm WITHOUT GOOD
    // REASON — there is no regression test to catch a mistake, and reverting to the
    // metadata-key-only computation would silently break it again.
    public static readonly ArchitectureDescriptor Stablelm = new()
    {
        Id = "stablelm",
        UsesNeoxRope = true,
        // stablelm.cpp ignores the use_parallel_residual key and branches on whether the per-layer ffn_norm TENSOR exists
        // (2-1.6B ships the key true yet has ffn_norm; only the 12B variant, without it, is truly parallel).
        ApplyModelSemantics = ctx => ctx.Baseline with { UseParallelResidual = ctx.TensorSource.FindTensor("blk.0.ffn_norm.weight") is null },
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // afmoe (Arcee Trinity Mini) — admitted 2026-10-04 on a real checkpoint (arcee-ai Q4_K_M only), CPU, contexts below the 2048-token
    // sliding window only (window masking beyond it is not verified): AfmoeGreedyParityTests teacher-forced against llama-server, all 31
    // confident positions match (13 of 32 on a 183-token prompt, 18 of 22 on a 5-token one), 3 near-tie differences. Needed: the per-layer
    // attention output gate and 3:1 sliding/global pattern with RoPE only on sliding layers (shared with Muse-Glimmer), muP embedding scale
    // sqrt(n_embd), and the afmoe pre-tokenizer (right-aligned digit groups). Not on any GPU path yet.
    public static readonly ArchitectureDescriptor Afmoe = new()
    {
        Id = "afmoe",
        UsesNeoxRope = true,
        ApplyModelSemantics = ctx => FamilyModelSemantics.GatedSwa(ctx, muse: false),
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Arcee Trinity Mini (`afmoe`)",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // maincoder — admitted 2026-08-09, FULL 24-of-24-token exact greedy match, bucket-1
    // (genuinely Apache-2.0), zero new code. Confirmed against maincoder.cpp before writing
    // any code: a literal Qwen3-shaped architecture — RMSNorm, biasless GQA with weighted
    // per-head QK-norm (AFTER RoPE — corrected 2026-09-26, see QkNormAfterRope), standard SiLU-gated FFN,
    // standard interleaved (non-NEOX) RoPE (confirmed via llama_model_rope_type() returning
    // NORM for LLM_ARCH_MAINCODER, matching the default). tokenizer.ggml.pre=qwen2 with real
    // merges — already covered. Every mechanism this checkpoint exercises predates this
    // session. See MaincoderGreedyParityTests and docs/done/01-gguf-model-coverage-plan.md §1w.
    public static readonly ArchitectureDescriptor Maincoder = new()
    {
        Id = "maincoder",
        // src/models/maincoder.cpp applies the weighted QK-norm after RoPE (PPL 12.61 -> 11.90 vs llama.cpp 12.01).
        ApplyModelSemantics = ctx => ctx.Baseline with { QkNormAfterRope = true },
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
