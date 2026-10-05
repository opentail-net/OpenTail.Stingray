namespace OpenTail.Stingray.Engine;

internal static class GraniteArchitecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "granite",
        Status = AdmissionStatus.Admitted,
        EvidenceDoc = "docs/STATUS.md",
        // IBM's embedded Jinja injects a default system prompt and indents user content, which makes
        // vision decoders emit <|end_of_text|> at token 0; use llama.cpp's canonical GRANITE_4_0 layout.
        FallbackChat = FallbackChatFormat.Granite,
    };
}
