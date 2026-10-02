namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// <see cref="RwkvTokenizer"/> (tokenizer.ggml.model = rwkv). The golden ids come from
/// <c>tools/llama.cpp/llama-tokenize.exe -f &lt;utf8 file&gt; --ids --no-escape</c> on
/// RWKV7-Goose-World3-1.5B-HF Q8_0 (2026-10-01; file input because Windows argv mangles non-ASCII);
/// the real-vocab cases skip visibly when that GGUF is not on this machine.
/// </summary>
public sealed class RwkvTokenizerTests
{
    private const string ModelFile = "RWKV7-Goose-World3-1.5B-HF.Q8_0.gguf";

    [Theory]
    [InlineData(@"\n", new byte[] { 10 })]
    [InlineData(@"\t\r", new byte[] { 9, 13 })]
    [InlineData(@"\x00\xff\x7f", new byte[] { 0, 255, 127 })]
    [InlineData(@"\\a\'", new byte[] { (byte)'\\', (byte)'a', (byte)'\'' })]
    [InlineData("é", new byte[] { 0xC3, 0xA9 })]
    public void Unescape_MatchesLlamaCppRules(string escaped, byte[] expected) =>
        Assert.Equal(expected, RwkvTokenizer.Unescape(escaped));

    [Fact]
    public void Encode_TakesLongestMatch_AndFallsBackToUnknown()
    {
        var tok = new RwkvTokenizer(["<unk>", "a", "ab", "abc", "b", @"\n"], unknownTokenId: 0);
        Assert.Equal([3, 2, 5, 1], tok.Encode("abcab\na"));
        Assert.Equal([1, 0, 4], tok.Encode("azb"));
    }

    public static TheoryData<string, int[]> Golden => new()
    {
        { "User: The capital of France is\n\nAssistant: Hello\t世界 x\\y",
            [24281, 59, 20996, 51128, 4706, 44312, 4600, 261, 5585, 41693, 59, 36786, 10, 10267, 14610, 355, 93, 122] },
        { "Hello world! 123 \U0001F600 café", [33155, 40213, 34, 3485, 52, 33, 3319, 153, 129, 37946] },
        { "\tindented\r\nline\n", [10, 41621, 1843, 263, 26150, 11] },
        // Not a special token in this GGUF (id 0 is "<s>", the only control token), so llama.cpp
        // tokenizes the chat template's opening marker as plain text, and so must we.
        { "<|rwkv_tokenizer_end_of_text|>User: hi",
            [61, 125, 2172, 2003, 96, 58464, 96, 7463, 96, 2090, 96, 27139, 125, 63, 24281, 59, 4571] },
    };

    [Theory]
    [MemberData(nameof(Golden))]
    public void Encode_MatchesLlamaTokenize_OnRealVocab(string text, int[] expected)
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for the llama-tokenize golden.");
        using var model = GgufModel.Open(path!);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        Assert.False(tokenizer.AddBosToken);
        Assert.Equal(expected, tokenizer.Encode(text));
        Assert.Equal(text, tokenizer.Decode(expected));
    }

    [Fact]
    public void NamedRwkvWorldTemplate_RendersLikeLlamaCppBuiltin()
    {
        // rwkv-6-world GGUFs store tokenizer.chat_template = "rwkv-world", the NAME of a llama.cpp
        // built-in. Expected strings are llama-server --no-jinja /apply-template output (2026-10-01).
        var source = new TokenizerSource { Tokens = ["<s>", "a"], ChatTemplate = "rwkv-world" };
        var template = GgufTokenizer.FromSource(source).ChatTemplate;
        Assert.NotNull(template);
        static Dictionary<string, object?> Msg(string role, string content) => new() { ["role"] = role, ["content"] = content };

        Assert.Equal("User: Write a short story about a lighthouse keeper.\n\nAssistant:",
            template.Render(new Dictionary<string, object?>
            {
                ["messages"] = new List<object?> { Msg("user", "  Write a short story about a lighthouse keeper.  ") },
                ["add_generation_prompt"] = true,
            }));
        Assert.Equal("System: Be brief.\n\nUser: Hi\n\nAssistant: Hello.\n\nUser: Bye\n\nAssistant:",
            template.Render(new Dictionary<string, object?>
            {
                ["messages"] = new List<object?> { Msg("system", "Be brief."), Msg("user", "Hi"), Msg("assistant", "Hello."), Msg("user", "Bye") },
                ["add_generation_prompt"] = true,
            }));
    }

    [Theory]
    [InlineData("chatml", "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello.<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n")]
    [InlineData("llama2", "[INST] Be brief.\nHi [/INST]Hello.</s>[INST] Bye [/INST]")]
    [InlineData("llama2-sys", "[INST] <<SYS>>\nBe brief.\n<</SYS>>\n\nHi [/INST]Hello.</s>[INST] Bye [/INST]")]
    [InlineData("mistral-v7", "[SYSTEM_PROMPT] Be brief.[/SYSTEM_PROMPT][INST] Hi[/INST] Hello.</s>[INST] Bye[/INST]")]
    [InlineData("phi3", "<|system|>\nBe brief.<|end|>\n<|user|>\nHi<|end|>\n<|assistant|>\nHello.<|end|>\n<|user|>\nBye<|end|>\n<|assistant|>\n")]
    [InlineData("zephyr", "<|system|>\nBe brief.<|endoftext|>\n<|user|>\nHi<|endoftext|>\n<|assistant|>\nHello.<|endoftext|>\n<|user|>\nBye<|endoftext|>\n<|assistant|>\n")]
    [InlineData("gemma", "<start_of_turn>user\nBe brief.\n\nHi<end_of_turn>\n<start_of_turn>model\nHello.<end_of_turn>\n<start_of_turn>user\nBye<end_of_turn>\n<start_of_turn>model\n")]
    [InlineData("falcon3", "<|system|>\nBe brief.\n<|user|>\nHi\n<|assistant|>\nHello.\n<|user|>\nBye\n<|assistant|>\n")]
    [InlineData("openchat", "Be brief.<|end_of_turn|>GPT4 Correct User: Hi<|end_of_turn|>GPT4 Correct Assistant: Hello.<|end_of_turn|>GPT4 Correct User: Bye<|end_of_turn|>GPT4 Correct Assistant:")]
    [InlineData("vicuna", "Be brief.\n\nUSER: Hi\nASSISTANT: Hello.</s>\nUSER: Bye\nASSISTANT:")]
    [InlineData("vicuna-orca", "SYSTEM: Be brief.\nUSER: Hi\nASSISTANT: Hello.</s>\nUSER: Bye\nASSISTANT:")]
    public void NamedTemplates_RenderLikeLlamaServerNoJinja(string templateName, string expected)
    {
        var source = new TokenizerSource { Tokens = ["<s>", "a"], ChatTemplate = templateName };
        var template = GgufTokenizer.FromSource(source).ChatTemplate;
        Assert.NotNull(template);
        static Dictionary<string, object?> Msg(string role, string content) => new() { ["role"] = role, ["content"] = content };

        string actual = template.Render(new Dictionary<string, object?>
        {
            ["messages"] = new List<object?>
            {
                Msg("system", "Be brief."),
                Msg("user", "Hi"),
                Msg("assistant", "Hello."),
                Msg("user", "Bye")
            },
            ["add_generation_prompt"] = true,
        });

        Assert.Equal(expected, actual);
    }

    private static string? FindModel()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            foreach (var sub in new[] { "models", Path.Combine("models", "_models") })
            {
                var candidate = Path.Combine(dir, sub, ModelFile);
                if (File.Exists(candidate)) return candidate;
            }
            if (Directory.GetParent(dir) is not { } parent) break;
            dir = parent.FullName;
        }
        var external = Path.Combine(@"E:\_models\rwkv7-goose-world3-1b5", ModelFile);
        return File.Exists(external) ? external : null;
    }
}
