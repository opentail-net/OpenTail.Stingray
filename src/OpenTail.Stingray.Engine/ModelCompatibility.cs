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
        "granite",
    };

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
