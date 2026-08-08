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
        // gptneox — admitted 2026-08-08 on a 22-of-24-token EXACT match against llama.cpp
        // (stronger than the apertus/olmoe receipts above), diverging at token 23 into a genuine
        // near-tie (0.007 apart on logits ~830, confirmed via a top-5 logit dump, not assumed).
        // Pythia (EleutherAI) is the reference family. Needs three things new to this engine:
        // LayerNorm with learned bias (SimdKernels.LayerNorm / ForwardPass.FastNorm, gated on
        // ModelHyperparams.HasNormBias, tensor-inventory-detected like HasAttnBias), non-gated GELU
        // FFN with learned up/down biases (SimdKernels.GeluInPlace, ModelHyperparams.HasFfnBias),
        // and true GPT-NeoX parallel residual (ModelHyperparams.UseParallelResidual: attention and
        // FFN both read the SAME pre-attention layer input, output is a 3-way sum, implemented as
        // an isolated branch in both RunTrunk and PrefillCore). Also the first architecture this
        // engine loads with a FUSED attn_qkv.weight/attn_qkv.bias tensor pair on the dense CPU
        // path (2304-wide bias split by element offset in ForwardPass's constructor).
        // See GptNeoxGreedyParityTests for the receipt, including two real defects found and fixed
        // while building it (neither in the "obvious" formula, both in refactored plumbing): (1) a
        // flipped Copy() direction in PrefillCore's per-token norm setup that fed layer 0's
        // LayerNorm an all-zero input instead of the token embedding (invisible on layers 1+, since
        // both buffers already agreed there) — attn_norm's learned bias made the zeroed-out result
        // still look like a plausible norm output; (2) the fused-QKV-bias split aliased three
        // pointers into ONE allocation, which Dispose() then freed independently — corrupting the
        // native heap with a deferred STATUS_HEAP_CORRUPTION crash on model teardown, long after
        // prefill/decode had already completed and looked fine.
        "gptneox",
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
