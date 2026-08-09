using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Declares the GGUF model profiles that the text-generation forward passes implement.
/// GGUF is a container, not a promise of interchangeable model mathematics: accepting an
/// unfamiliar architecture merely because it happens to expose familiar tensor names can
/// produce plausible but incorrect tokens. Keep this gate deliberately conservative.
/// </summary>
public static class ModelCompatibility
{
    private static readonly HashSet<string> s_textGenerationArchitectures = new(StringComparer.OrdinalIgnoreCase)
    {
        // Decoder-only transformer profiles exercised by OpenTail's forward passes.
        "llama", "llama4",
        "qwen", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "qwen35", "qwen35moe",
        "gemma", "gemma2", "gemma3", "gemma3n", "gemma4",
        "phi2", "phi3", "phimoe",
        // olmoe — admitted 2026-08-08 on perplexity parity, NOT on token-for-token greedy parity,
        // which it does not achieve. On wikitext at a matched 2048-token context llama.cpp b8585
        // scores 7.4868 and this engine scores 7.3889 (1.3%). The greedy divergence is at a flat
        // position where the top five candidates span 1.55 logits, i.e. where a differently
        // quantised matmul reorders candidates. Evidence and the argument for accepting it:
        // docs/01-gguf-model-coverage-plan.md §1b. Note `olmo2` is deliberately NOT here — it
        // shares neither a fixture nor a receipt.
        "olmoe",
        // granite — admitted 2026-08-08 on FULL 24-token exact greedy match against llama.cpp
        // (stronger than the olmoe receipt above, which only reaches a 2-token prefix). Needs a
        // "scale trio" + attention-scale override beyond the plain llama trunk, read from GGUF
        // metadata (ModelHyperparams.ResidualScale/AttentionScaleOverride/LogitScale, generalized
        // EmbeddingScale) — see GraniteGreedyParityTests and docs/01-gguf-model-coverage-plan.md
        // §1d for the receipt and for what is NOT yet wired (TurboQuant prefill, continuous-batching
        // admission, CUDA/Vulkan). MiniCPM (not MiniCPM3 — that's MLA, a different architecture)
        // shares this exact graph in llama.cpp and reuses the same implementation, unvalidated here
        // pending a permissively-licensed checkpoint on a llama.cpp build that can serve as an oracle.
        "granite",
        // smollm3 — one twist over the plain llama trunk: NoPE every 4th layer, gated the same way
        // as llama4's noRopeStep. See SmolLm3GreedyParityTests for the full 24-token greedy receipt.
        "smollm3",
        // apertus — admitted 2026-08-08 on an 11-token EXACT prefix match (one full sentence)
        // against llama.cpp, diverging afterward into a different but still coherent, on-topic
        // completion (not degenerate output). The first "new-kernel" architecture admitted this
        // session: no ffn_gate tensor at all (plain up -> xIELU -> down, ModelHyperparams.Xielu*,
        // SimdKernels.XieluInPlace), detected from tensor inventory rather than architecture
        // string. See ApertusGreedyParityTests and docs/01-gguf-model-coverage-plan.md §1f for the
        // receipt, including a real defect found and fixed in the xIELU parameter transform
        // (GGUF stores pre-softplus values; llama.cpp's ggml_xielu() wrapper — not the compute
        // kernel — applies softplus before use, easy to miss by reading only the kernel).
        "apertus",
        // gptneox (Pythia) — LayerNorm (mean/variance + learned bias, not RMSNorm), a biased
        // non-gated GELU FFN, a fused blk.*.attn_qkv.weight/bias tensor pair (split by
        // contiguous row offset in ForwardPass's constructor — Q rows, then K rows, then V
        // rows; confirmed against examples/llama.cpp/llama.cpp/conversion/gptneox.py and
        // src/models/gptneox.cpp, NOT the interleaved per-head layout an earlier draft
        // assumed), and the metadata-driven parallel-residual graph (x + attn(ln1(x)) +
        // ffn(ln2(x)), both norms reading the SAME incoming residual — ModelHyperparams.
        // HasNormBias/HasFfnBias/UseParallelResidual). See GptNeoxGreedyParityTests and
        // docs/01-gguf-model-coverage-plan.md for the receipt. TurboQuant prefill, continuous-
        // batching admission, and CUDA/Vulkan are not wired to this profile.
        "gptneox",
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
        // docs/01-gguf-model-coverage-plan.md for the receipt.
        "falcon",
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
        // See Olmo2GreedyParityTests and docs/01-gguf-model-coverage-plan.md for the receipt.
        "olmo2",
    };
    // minicpm — NOT admitted. The forward-pass scale trio (reusing Granite's graph, see
    // GraniteGreedyParityTests) is implemented and presumed correct, but MiniCPM4-0.5B — the only
    // Apache-2.0 checkpoint tried (2026-08-08) — declares tokenizer.ggml.model=llama with a
    // `scores` array and NO `merges` array: Unigram-LM SentencePiece (Viterbi segmentation), not
    // the BPE-order SPM (explicit merges list) that Llama/Gemma use and this engine implements.
    // Measured: our tokenizer produces unrelated single-token-per-fragment ids for a 5-token
    // reference prompt. A different, unimplemented tokenization algorithm, not a scale-trio bug —
    // see docs/01-gguf-model-coverage-plan.md §1d.

    /// <summary>Whether the architecture has an implemented text-generation forward profile.</summary>
    public static bool IsTextGenerationArchitectureSupported(string architecture) =>
        s_textGenerationArchitectures.Contains(architecture);

    /// <summary>
    /// Matrix weight formats implemented by the portable CPU path. CUDA/Vulkan routes share
    /// this conservative baseline at model-load time, so a model cannot defer a missing CPU
    /// fallback/dequantizer error until its first request.
    /// </summary>
    public static bool IsSupportedWeightDType(DType dtype) => dtype is
        DType.Float32 or DType.Float16 or DType.BFloat16 or
        DType.Q4_0 or DType.Q4_1 or DType.Q5_0 or DType.Q5_1 or DType.Q8_0 or DType.Q8_1 or
        DType.Q2_K or DType.Q3_K or DType.Q4_K or DType.Q5_K or DType.Q6_K or
        DType.IQ4_NL or DType.MXFP4 or DType.NVFP4 or DType.Q1_0 or DType.Q2_0;

    /// <summary>
    /// Validates that a GGUF can be served by OpenTail's text-generation engine. Call this
    /// after reading metadata and before selecting a backend or constructing a forward pass.
    /// </summary>
    public static void ValidateForTextGeneration(GgufModel model)
    {
        ArgumentNullException.ThrowIfNull(model);

        string architecture = model.Metadata.TryGetValue("general.architecture", out var value)
            ? Convert.ToString(value) ?? ""
            : "llama";

        if (!IsTextGenerationArchitectureSupported(architecture))
        {
            throw new NotSupportedException(
                $"GGUF architecture '{architecture}' is not supported for text generation by OpenTail.Stingray. " +
                "The model was rejected before inference because GGUF tensor naming alone does not establish " +
                "compatible attention, RoPE, normalization, and FFN semantics. Supported profiles: " +
                $"{string.Join(", ", s_textGenerationArchitectures.Order())}.");
        }

        var unsupported = model.Tensors
            .Where(t => !IsSupportedWeightDType(t.DType))
            .Select(t => $"{t.Name} ({t.DType})")
            .Take(4)
            .ToArray();
        if (unsupported.Length > 0)
        {
            throw new NotSupportedException(
                "This GGUF uses tensor storage formats that OpenTail.Stingray cannot execute on its portable " +
                "text-generation path: " + string.Join(", ", unsupported) + ". " +
                "Use a model quantized as Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1, Q2_K–Q6_K, IQ4_NL, " +
                "MXFP4, NVFP4, Q1_0, Q2_0, F16, BF16, or F32.");
        }
    }
}
