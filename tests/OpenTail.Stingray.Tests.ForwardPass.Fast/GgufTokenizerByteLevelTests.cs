namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// GPT-2 byte-level BPE: byte 0xAD (soft hyphen) is NOT in bytes_to_unicode's printable set and
/// must map to U+0143. It used to pass through unchanged, so any character whose UTF-8 contains
/// 0xAD — "í" is C3 AD — tokenized into byte garbage (found 2026-09-26 diffing wikitext against
/// llama-tokenize: Qwen2.5 and Pythia diverged at "jídàchéng"). Reference ids are llama.cpp's
/// <c>llama-tokenize -p</c> output for the same string on the same GGUF.
/// </summary>
public sealed class GgufTokenizerByteLevelTests
{
    private const string Text = "Chéngjì (jídàchéng-) soft­hyphen";

    private static string? FindModel(string file)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            foreach (var p in new[] { Path.Combine(dir.FullName, "models", file), Path.Combine(dir.FullName, "models", "_models", file) })
                if (File.Exists(p)) return p;
            dir = dir.Parent;
        }
        return null;
    }

    [Fact]
    public void Qwen25_SoftHyphenBytes_MatchLlamaTokenize()
    {
        string? path = FindModel("qwen2.5-0.5b-instruct-q4_k_m.gguf");
        Assert.SkipUnless(path is not null, "qwen2.5-0.5b-instruct-q4_k_m.gguf not present");
        using var model = GgufModel.Open(path!);
        var tok = GgufTokenizer.FromGgufModel(model);

        int[] expected = [1143, 963, 968, 73, 23531, 320, 73, 86020, 6362, 74017, 968, 61996, 8413, 5760, 8503, 14769];
        int[] ids = tok.Encode(Text).ToArray();
        Assert.Equal(expected, ids);
        Assert.Equal(Text, tok.Decode(ids));
    }
}
