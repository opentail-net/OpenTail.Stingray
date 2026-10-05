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
    }
}
