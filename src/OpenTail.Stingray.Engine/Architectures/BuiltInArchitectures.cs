namespace OpenTail.Stingray.Engine;

/// <summary>
/// The manifest of migrated architectures. Explicit (no reflection) so NativeAOT/trim stay clean.
/// Anything not listed here is still handled by ModelCompatibility's legacy allowlist.
/// </summary>
internal static class BuiltInArchitectures
{
    public static IEnumerable<ArchitectureDescriptor> Create()
    {
        yield return Gemma4Architecture.Descriptor;
        yield return GraniteArchitecture.Descriptor;
        yield return LlamaArchitecture.Descriptor;
        yield return Llama4Architecture.Descriptor;
        yield return MuseGlimmerArchitecture.Descriptor;

        // Ported, not verified (CLAUDE.md rule 14): refused until a real checkpoint is verified.
        yield return GlmDsaArchitecture.Descriptor;
        yield return Glm5NextArchitecture.Descriptor;
        yield return DiffusionGemmaArchitecture.Descriptor;
        yield return Qwen4ExpArchitecture.Descriptor;
        yield return DeepSeek41Architecture.Descriptor;
        yield return DeepSeek4Architecture.Descriptor;
        yield return DeepSeek32Architecture.Descriptor;
    }
}
