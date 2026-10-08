namespace OpenTail.Stingray.Engine;

// Part of OtherAdmittedArchitectures (partial), grouped by family. Descriptors are independent of one another.
internal static partial class OtherAdmittedArchitectures
{
    // gpt-oss — admitted 2026-09-26. Runs on its own GptOssForwardPass (CPU only; attention
    // sinks, 1:1 SWA/full alternation, biased MoE, OAI SwiGLU, YaRN factor 32 on both layer
    // kinds), routed by RunCommand/InferenceEngineLoader. Receipt vs llama-server (vendored
    // tools/llama.cpp) on gpt-oss-20b-MXFP4: "The capital of France is" teacher-forced over
    // llama-server's 24 greedy tokens, 22/24 exact argmax, worst gap 0.063 logits
    // (GptOssRealWeightSmokeTests; step 1 is a 0.02-logit tie inside llama.cpp itself); a 188-token Paris-history prompt (past the 128-token window) 16/32 exact, then
    // a flip at a 0.105-logit near-tie (220 vs 5030). First-token top-5 in identical order,
    // gaps to the top logit within 0.002-0.065 of llama.cpp's -fa off run — smaller than
    // llama.cpp's own -fa on/off shift (up to 0.13). Plain RoPE instead of YaRN diverged at
    // token 14 with a different top-5, so the YaRN wiring is load-bearing. See
    // docs/done/101-work-queue-after-coverage-plan.md.
    public static readonly ArchitectureDescriptor GptOss = new()
    {
        Id = "gpt-oss",
        UsesNeoxRope = true,
        Status = AdmissionStatus.Admitted,
        // CPU or full Vulkan only; CUDA and partial GPU requests fall back to CPU.
        // Enforced by RunCommand.cs:2035-2054 and InferenceEngineLoader.cs:620-634.
        ForwardPassFamily = ForwardPassFamily.GptOss,
        SupportedBackends = SupportedBackends.Cpu | SupportedBackends.Vulkan,
        BackendLimitation = "gpt-oss supports CPU or full Vulkan offload only; CUDA and partial offload use CPU.",
        StatusAnchor = "gpt-oss (`gpt-oss`)",
        EvidenceDoc = "docs/done/101-work-queue-after-coverage-plan.md",
        SupportsContinuousBatching = false,
        CreateForwardPass = ctx =>
        {
            var gptOssHp = GptOssHyperparams.FromModel(ctx.Probe.Gguf!);
            if (ctx.Decision.Kind == ForwardPassKind.GptOssVulkan)
            {
                var vk = ctx.VulkanBackend ?? new OpenTail.Stingray.Vulkan.VulkanBackend();
                if (ctx.VulkanBackend is null) ctx.TrackDisposable(vk);
                var gptGpu = new GptOssGpuForwardPass(ctx.Probe.Gguf!, vk, gptOssHp, maxContextLength: ctx.ContextSize);
                ctx.TrackDisposable(gptGpu);
                return gptGpu;
            }
            var gptPass = new GptOssForwardPass(ctx.Probe.Gguf!, gptOssHp);
            ctx.TrackDisposable(gptPass);
            return gptPass;
        },
    };

    // gptneox (Pythia) — LayerNorm (mean/variance + learned bias, not RMSNorm), a biased
    // non-gated GELU FFN, a fused blk.*.attn_qkv.weight/bias tensor pair (split by
    // contiguous row offset in ForwardPass's constructor — Q rows, then K rows, then V
    // rows; confirmed against examples/llama.cpp/llama.cpp/conversion/gptneox.py and
    // src/models/gptneox.cpp, NOT the interleaved per-head layout an earlier draft
    // assumed), and the metadata-driven parallel-residual graph (x + attn(ln1(x)) +
    // ffn(ln2(x)), both norms reading the SAME incoming residual — ModelHyperparams.
    // HasNormBias/HasFfnBias/UseParallelResidual). See GptNeoxGreedyParityTests and
    // docs/done/01-gguf-model-coverage-plan.md for the receipt. TurboQuant prefill, continuous-
    // batching admission, and CUDA/Vulkan are not wired to this profile.
    public static readonly ArchitectureDescriptor GptNeoX = new()
    {
        Id = "gptneox",
        UsesNeoxRope = true,
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // starcoder2 — ADMITTED 2026-08-09, full 24-of-24-token exact match. Reuses gptneox/
    // falcon's LayerNorm-with-bias + non-gated biased-GELU FFN infrastructure exactly (same
    // SimdKernels.LayerNorm/GeluInPlace, same HasNormBias/HasFfnBias/HasAttnBias/
    // HasAttnOutputBias tensor-presence detection), but with the ORDINARY sequential residual
    // (x1 = x + attn(LN(x)); x2 = x1 + ffn(LN(x1))) — confirmed against
    // examples/llama.cpp/llama.cpp/src/models/starcoder2.cpp — not gptneox/falcon's parallel
    // 3-way sum. UseParallelResidual is false here (no metadata key, no arch-string hardcode).
    //
    // ONE REAL DEFECT FOUND AND FIXED — a latent bug this receipt was the first thing to
    // exercise. RunTrunk's sequential (non-parallel-residual) FFN pre-norm still called
    // FastRmsNorm directly instead of the bias-aware FastNorm dispatcher, because no
    // previously-admitted architecture had BOTH HasNormBias=true AND UseParallelResidual=false
    // at the same time (gptneox/falcon are always parallel-residual) — the sequential+LayerNorm
    // combination was unreachable code until starcoder2. Symptom: greedy continuation matched
    // llama.cpp for exactly 1 token then diverged completely, and the prefill/decode
    // consistency check disagreed with ITSELF (maxDiff 30.4, argmax mismatch) — a strong signal
    // of a structural bug, not a numerical approximation. Fixed by routing that call through
    // FastNorm with the layer's ffn-norm bias, matching what PrefillCore's equivalent branch
    // already did correctly.
    //
    // NO AUTOMATED TEST FOR THIS ARCHITECTURE, FOR LICENCE REASONS. Checkpoint license:
    // "bigcode-openrail-m" (BigCode OpenRAIL-M — a restricted-use RAIL license: use-based
    // restrictions, e.g. malicious-code generation), not MIT/Apache-2.0/BSD/MPL. Verified once
    // against `bigcode/starcoder2-3b` (via QuantFactory/starcoder2-3b-GGUF, Q8_0), a transient
    // local download, never vendored, deleted immediately after this receipt. Per the license
    // policy in docs/done/01-gguf-model-coverage-plan.md, no permanent test persists.
    //
    // Verification evidence (2026-08-09, starcoder2-3b Q8_0, llama.cpp b8585-cad2d3884): prompt
    // "The capital of France is" -> ids [1338, 18972, 451, 45569, 458]. Reference 24-token
    // greedy continuation (--temp 0 --top-k 1 --seed 0, no-bos): " Paris.\n\n```\n\nI want to
    // get the value of the attribute `value` of the `span" -> ids [2736, 316, 51, 222, 222, 932,
    // 222, 222, 78, 2660, 391, 640, 341, 804, 451, 341, 3895, 548, 872, 101, 451, 341, 548, 681].
    // This engine matched ALL 24 tokens exactly (after the FastNorm fix above), and the
    // prefill/decode stepwise-consistency check agreed.
    //
    // DO NOT MODIFY THIS ARCHITECTURE'S CODE PATH WITHOUT GOOD REASON — there is no regression
    // test to catch a mistake. In particular, the RunTrunk sequential-FFN-norm fix above is
    // currently the ONLY thing exercising that exact code combination; a future change there
    // could silently reintroduce the bug this receipt just fixed.
    public static readonly ArchitectureDescriptor Starcoder2 = new()
    {
        Id = "starcoder2",
        UsesNeoxRope = true,
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // gpt2 — admitted 2026-08-09, FULL 22-of-22-token exact greedy match, bucket-1 (genuinely
    // MIT). The first architecture this session without RoPE at all: GPT-2 encodes position
    // via a learned absolute position-embedding table (`position_embd.weight`) added to the
    // token embedding once, before the trunk starts, not via rotary embeddings inside
    // attention. New: ForwardPass._posEmbdTensor (loaded only when the tensor exists) and a
    // `position` parameter threaded through EmbedTokenInto/EmbedToken and all 8 call sites
    // (every prefill/decode dispatch path). Disabling RoPE needed no new field at all —
    // ModelHyperparams.NoRopeLayerStep = 1 makes the EXISTING Llama-4/SmolLM3 periodic-skip
    // formula ((layer+1) % step != 0) evaluate to "never" for every layer, reusing dispatch
    // every call site already had. Everything else (LayerNorm-with-bias, fused
    // attn_qkv.weight/.bias, non-gated biased-GELU FFN) was already generic from
    // gptneox/falcon. A Q6_K quant tried first diverged at position 5 on only a 0.106-logit
    // gap — read as ordinary quantization sensitivity for a genuinely small/weak 124M model
    // (more sensitive than larger checkpoints, not less) and confirmed by re-running against
    // a near-lossless F16 checkpoint, which matches exactly with no near-tie at all. See
    // Gpt2GreedyParityTests and docs/done/01-gguf-model-coverage-plan.md §1q for the receipt.
    public static readonly ArchitectureDescriptor Gpt2 = new()
    {
        Id = "gpt2",
        // Learned absolute position table, no RoPE anywhere: step 1 makes (layer+1)%step != 0 false for every layer.
        ApplyModelSemantics = ctx => ctx.Baseline with { NoRopeLayerStep = 1 },
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; the GPT-2 SafeTensors row describes a separate loader path.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // starcoder (v1) — admitted 2026-08-09, FULL 23-of-23-token exact greedy match, bucket-2,
    // near-zero code change. Confirmed against starcoder.cpp before writing any code: SAME
    // shape as gpt2 (this session's earlier admission) — learned absolute position embeddings
    // (ggml_get_rows(pos_embd, inp_pos), no RoPE anywhere), LayerNorm-with-bias, fused
    // attn_qkv.weight/.bias, non-gated biased-GELU FFN. Also exercises MQA (head_count=16,
    // head_count_kv=1) through the already-generic GQA-parametrized fused-QKV split (first
    // proven on falcon's identical head_count_kv=1 shape). The only change: extended
    // ModelGraph.cs's NoRopeLayerStep=1 gate (built for gpt2) from a single-arch check to
    // arch is "gpt2" or "starcoder". Full exact match on the first real attempt.
    //
    // NO AUTOMATED TEST FOR THIS ARCHITECTURE, FOR LICENCE REASONS. Checkpoint:
    // `bigcode/starcoderbase-1b` (mradermacher GGUF, Q8_0), BigCode OpenRAIL-M — a restricted-
    // use RAIL license (e.g. malicious-code-generation restrictions), not MIT/Apache-2.0/
    // BSD/MPL. Transient local download, never vendored, deleted immediately after this
    // receipt.
    //
    // Verification evidence (2026-08-09, starcoderbase-1b Q8_0, llama.cpp b8585-cad2d3884):
    // prompt "The capital of France is" -> ids [1318, 18926, 432, 45600, 438]. Full 23-of-23
    // token exact match against the reference continuation, no near-tie, no divergence.
    //
    // DO NOT MODIFY THIS ARCHITECTURE'S CODE PATH WITHOUT GOOD REASON — there is no regression
    // test to catch a mistake.
    public static readonly ArchitectureDescriptor Starcoder = new()
    {
        Id = "starcoder",
        // Same absolute-position-embedding shape as GPT-2 (starcoder.cpp has no RoPE call).
        ApplyModelSemantics = ctx => ctx.Baseline with { NoRopeLayerStep = 1 },
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // codeshell — admitted 2026-08-09, FULL 24-of-24-token exact greedy match, bucket-2,
    // genuinely zero new production code. Confirmed against codeshell.cpp before writing any
    // test: LayerNorm-with-bias, fused attn_qkv.weight/.bias, non-gated biased-GELU FFN — same
    // shapes as gptneox/falcon/starcoder — but REAL RoPE (ggml_rope_ext calls present, NEOX
    // convention per llama_model_rope_type(), and "codeshell" was already in this engine's
    // isNeoxRope list from an earlier session pass), not gpt2/starcoder's absolute position
    // embeddings — so it didn't even need the NoRopeLayerStep widening those two used. The
    // only failure along the way was a wrong test assertion (assumed NORM rope by misreading
    // llama-model.cpp's rope-type switch; codeshell is genuinely in the NEOX case block), not
    // an engine defect.
    //
    // NO AUTOMATED TEST FOR THIS ARCHITECTURE, FOR LICENCE REASONS. Checkpoint:
    // `WisdomShell/CodeShell-7B` (mradermacher GGUF, Q4_K_M), custom WisdomShell/CodeShell
    // license (no SPDX permissive tag found) — not MIT/Apache-2.0/BSD/MPL. Transient local
    // download, never vendored, deleted immediately after this receipt.
    //
    // Verification evidence (2026-08-09, CodeShell-7B Q4_K_M, llama.cpp b8585-cad2d3884):
    // prompt "The capital of France is" -> ids [46479, 53434, 15979, 48944, 19206, 55391].
    // Full 24-of-24 token exact match against the reference continuation, no near-tie, no
    // divergence.
    //
    // DO NOT MODIFY THIS ARCHITECTURE'S CODE PATH WITHOUT GOOD REASON — there is no regression
    // test to catch a mistake.
    public static readonly ArchitectureDescriptor Codeshell = new()
    {
        Id = "codeshell",
        UsesNeoxRope = true,
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
