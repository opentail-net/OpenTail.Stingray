namespace OpenTail.Stingray.Engine;

/// <summary>
/// Declares the GGUF model profiles that the text-generation forward passes implement.
/// GGUF is a container, not a promise of interchangeable model mathematics: accepting an
/// unfamiliar architecture merely because it happens to expose familiar tensor names can
/// produce plausible but incorrect tokens. Keep this gate deliberately conservative.
/// </summary>
public static class ModelCompatibility
{
    /// <summary>Whether the architecture has an implemented text-generation forward profile.</summary>
    public static bool IsTextGenerationArchitectureSupported(string architecture) =>
        ArchitectureRegistry.Find(architecture)?.IsUsable() ?? false;

    /// <summary>
    /// Matrix weight formats implemented by the portable CPU path. CUDA/Vulkan routes share
    /// this conservative baseline at model-load time, so a model cannot defer a missing CPU
    /// fallback/dequantizer error until its first request.
    /// </summary>
    public static bool IsSupportedWeightDType(DType dtype) => dtype is
        DType.Float32 or DType.Float16 or DType.BFloat16 or
        DType.Q4_0 or DType.Q4_1 or DType.Q5_0 or DType.Q5_1 or DType.Q8_0 or DType.Q8_1 or
        DType.Q2_K or DType.Q3_K or DType.Q4_K or DType.Q5_K or DType.Q6_K or
        DType.IQ4_NL or DType.IQ2_S or DType.IQ2_XS or DType.IQ2_XXS or DType.IQ3_XXS or DType.IQ3_S or DType.IQ4_XS or
        DType.IQ1_S or DType.IQ1_M or
        DType.MXFP4 or DType.NVFP4 or DType.Q1_0 or DType.Q2_0 or
        DType.TQ2_0 or DType.TQ1_0;

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

        if (!IsTextGenerationArchitectureSupported(architecture)
            && Environment.GetEnvironmentVariable("STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH") != "1")
        {
            var known = ArchitectureRegistry.Find(architecture);
            if (known is not null)
                throw new NotSupportedException(
                    $"GGUF architecture '{architecture}' is not admitted by OpenTail.Stingray: {known.RefusalReason} " +
                    $"(status {known.Status}; record: {known.EvidenceDoc}).");
            throw new NotSupportedException(
                $"GGUF architecture '{architecture}' is not supported for text generation by OpenTail.Stingray. " +
                "The model was rejected before inference because GGUF tensor naming alone does not establish " +
                "compatible attention, RoPE, normalization, and FFN semantics. Supported profiles: " +
                $"{string.Join(", ", ArchitectureRegistry.All.Where(d => d.Status == AdmissionStatus.Admitted).SelectMany(d => d.Aliases.Prepend(d.Id)).Distinct().Order())}.");
        }

        // Bonsai2 PRISM (PQ2_0/PTQ1_0 + prism.hadamard.* transforms) — NOT admitted. Ported 2026-10-02
        // (BonsaiQuant, PrismHadamard, PrismHadamardMetadata, HybridGdnForwardPass; spec from TensorSharp's
        // BSD-3 port) but not verified against the publisher reference (PrismML-Eng/llama.cpp, branch prism):
        // "ported, not verified" (CLAUDE.md rule 14; docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md
        // §3b). Loadable only with STINGRAY_EXPERIMENTAL_PRISM=1, on the CPU hybrid-GDN pass.
        bool prismTensors = model.Tensors.Any(t => Cpu.BonsaiQuant.IsBonsaiType(t.DType));
        if (prismTensors && Environment.GetEnvironmentVariable("STINGRAY_EXPERIMENTAL_PRISM") != "1")
            throw new NotSupportedException(
                "This GGUF uses Bonsai2 PRISM weights (PQ2_0/PTQ1_0 with Hadamard transforms). Support is ported " +
                "but not yet verified against the publisher's reference; set STINGRAY_EXPERIMENTAL_PRISM=1 to try it " +
                "(CPU only, outputs unverified).");

        var unsupported = model.Tensors
            .Where(t => !IsSupportedWeightDType(t.DType) && !(prismTensors && Cpu.BonsaiQuant.IsBonsaiType(t.DType)))
            .Select(t => $"{t.Name} ({t.DType})")
            .Take(4)
            .ToArray();
        if (unsupported.Length > 0)
        {
            throw new NotSupportedException(
                "This GGUF uses tensor storage formats that OpenTail.Stingray cannot execute on its portable " +
                "text-generation path: " + string.Join(", ", unsupported) + ". " +
                "Use a model quantized as Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1, Q2_K–Q6_K, IQ4_NL, " +
                "IQ2_S, IQ2_XS, IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_XS, IQ1_S, IQ1_M, MXFP4, NVFP4, Q1_0, Q2_0, TQ2_0, TQ1_0, F16, BF16, or F32.");
        }
    }
}
