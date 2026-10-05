namespace OpenTail.Stingray.Engine;

/// <summary>
/// The manifest of registered architectures. Explicit (no reflection) so NativeAOT/trim stay clean.
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

        yield return QwenArchitectures.Qwen;
        yield return QwenArchitectures.Qwen2;
        yield return QwenArchitectures.Qwen2moe;
        yield return QwenArchitectures.Qwen3;
        yield return QwenArchitectures.Qwen3moe;
        yield return QwenArchitectures.Qwen2vl;
        yield return QwenArchitectures.Qwen35;
        yield return QwenArchitectures.Qwen35moe;
        yield return QwenArchitectures.Qwen3vl;
        yield return GemmaArchitectures.Gemma;
        yield return GemmaArchitectures.Gemma2;
        yield return GemmaArchitectures.Gemma3;
        yield return GemmaArchitectures.Gemma3n;
        yield return PhiArchitectures.Phi2;
        yield return PhiArchitectures.Phi3;
        yield return PhiArchitectures.Phimoe;
        yield return RwkvArchitectures.Rwkv7;
        yield return RwkvArchitectures.Rwkv6;
        yield return DeepSeek2Architectures.DeepSeek2;
        yield return DeepSeek2Architectures.DeepSeek2Ocr;
        yield return GraniteArchitectures.Granitehybrid;
        yield return GraniteArchitectures.Granitemoe;
        yield return OtherAdmittedArchitectures.Mimo;
        yield return OtherAdmittedArchitectures.Mimo2;
        yield return OtherAdmittedArchitectures.Olmoe;
        yield return OtherAdmittedArchitectures.GptOss;
        yield return OtherAdmittedArchitectures.Smollm3;
        yield return OtherAdmittedArchitectures.Apertus;
        yield return OtherAdmittedArchitectures.GptNeoX;
        yield return OtherAdmittedArchitectures.Falcon;
        yield return OtherAdmittedArchitectures.Olmo2;
        yield return OtherAdmittedArchitectures.Exaone;
        yield return OtherAdmittedArchitectures.Orion;
        yield return OtherAdmittedArchitectures.Ernie45;
        yield return OtherAdmittedArchitectures.Paddleocr;
        yield return OtherAdmittedArchitectures.Nemotronh;
        yield return OtherAdmittedArchitectures.Lfm2;
        yield return OtherAdmittedArchitectures.Lfm2moe;
        yield return OtherAdmittedArchitectures.Internlm2;
        yield return OtherAdmittedArchitectures.Starcoder2;
        yield return OtherAdmittedArchitectures.Cohere2;
        yield return OtherAdmittedArchitectures.Glm4;
        yield return OtherAdmittedArchitectures.Glm4moe;
        yield return OtherAdmittedArchitectures.Stablelm;
        yield return OtherAdmittedArchitectures.Hunyuandense;
        yield return OtherAdmittedArchitectures.Hunyuanmoe;
        yield return OtherAdmittedArchitectures.Afmoe;
        yield return OtherAdmittedArchitectures.Gpt2;
        yield return OtherAdmittedArchitectures.Olmo;
        yield return OtherAdmittedArchitectures.Starcoder;
        yield return OtherAdmittedArchitectures.Codeshell;
        yield return OtherAdmittedArchitectures.Jais2;
        yield return OtherAdmittedArchitectures.Jais;
        yield return OtherAdmittedArchitectures.Maincoder;
        yield return OtherAdmittedArchitectures.Exaone4;
        yield return OtherAdmittedArchitectures.Mistral3;
        yield return OtherAdmittedArchitectures.Ministral;
        yield return OtherAdmittedArchitectures.Xverse;
        yield return OtherAdmittedArchitectures.Minicpm;

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
