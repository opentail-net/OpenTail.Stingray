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

    /// <summary>
    /// A byte-level BPE GGUF with no <c>tokenizer.ggml.pre</c> key (StableLM-zephyr's conversion)
    /// must use llama.cpp's LLAMA_VOCAB_PRE_TYPE_DEFAULT cascade — punctuation/symbol runs split out
    /// first, then GPT-2, then digit runs — not plain GPT-2. With plain GPT-2 wikitext diverged from
    /// token 0 (" = " came out as different pieces); the reference is llama-tokenize's output.
    /// </summary>
    [Fact]
    public void StableLm_MissingPreTokenizer_UsesLlamaCppDefaultCascade()
    {
        string? path = FindModel("stablelm-zephyr-3b.Q4_K_M.gguf");
        Assert.SkipUnless(path is not null, "stablelm-zephyr-3b.Q4_K_M.gguf not present");
        using var model = GgufModel.Open(path!);
        var tok = GgufTokenizer.FromGgufModel(model);

        int[] expected = [209, 30, 6911, 378, 3941, 350, 209, 30, 275, 209, 1518, 17, 13, 247, 12, 67, 44072, 68, 209, 95, 536];
        Assert.Equal(expected, tok.Encode(" = Robert Boulter = in 2000, a+b<=c ~ok").ToArray());
    }

    /// <summary>
    /// tokenizer.ggml.model=gemma4 is merge-RANK BPE in llama.cpp (not score-based SPM), with the
    /// text split into newline / non-newline runs first. Score-based merging gave "▁Heron"+"s"
    /// instead of "▁Her"+"ons". Reference ids: llama-tokenize --no-bos on the same GGUF.
    /// </summary>
    [Fact]
    public void Gemma4_RankBpeAndNewlineRuns_MatchLlamaTokenize()
    {
        string? path = FindModel("gemma-4-E4B-it-Q4_K_M.gguf");
        Assert.SkipUnless(path is not null, "gemma-4-E4B-it-Q4_K_M.gguf not present");
        using var model = GgufModel.Open(path!);
        var tok = GgufTokenizer.FromGgufModel(model);

        int[] expected = [528, 506, 1441, 5165, 1190, 109, 25217, 684, 138, 63862, 107];
        Assert.Equal(expected, tok.Encode(" in the play Herons\n\n\nwritten by  Simon\n").ToArray());
    }
}
