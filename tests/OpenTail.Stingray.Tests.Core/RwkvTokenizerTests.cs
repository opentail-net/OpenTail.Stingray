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
