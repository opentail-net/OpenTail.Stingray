namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// <see cref="EncoderTokenizer"/> built from real <c>tokenizer.json</c> files in <c>models/_models/hf</c>.
/// The WordPiece case reruns llama.cpp's BERT/BGE vocab golden (<c>ggml-vocab-bert-bge.gguf.inp/.out</c>)
/// through the tokenizer.json loader for every checkpoint that ships the 30522-entry uncased BERT vocab,
/// so the independent oracle covers the JSON path too. Template/pair tests pin the post_processor
/// layouts (BERT TemplateProcessing, XLM-R TemplateProcessing, MPNet RobertaProcessing) and HF's
/// longest-first truncation. Each case skips visibly when its files are missing.
/// </summary>
public sealed class EncoderTokenizerTests
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static EncoderTokenizer Load(string repoDir)
    {
        string? path = FindRepoFile($"models/_models/hf/{repoDir}/tokenizer.json");
        Assert.SkipUnless(path != null, $"{repoDir}/tokenizer.json not found");
        return EncoderTokenizer.FromTokenizerJson(path!);
    }

    [Theory]
    [InlineData("BAAI__bge-small-en-v1.5")]
    [InlineData("BAAI__bge-large-en-v1.5")]
    [InlineData("sentence-transformers__all-MiniLM-L6-v2")]
    [InlineData("cross-encoder__ms-marco-MiniLM-L6-v2")]
    [InlineData("google-bert__bert-base-uncased")]
    [InlineData("google__electra-base-discriminator")]
    [InlineData("nomic-ai__nomic-embed-text-v1.5")]
    public void WordPieceJson_MatchesLlamaCppBgeVocabGolden(string repoDir)
    {
        string? inp = FindRepoFile("examples/llama.cpp/llama.cpp/models/ggml-vocab-bert-bge.gguf.inp");
        string? outp = FindRepoFile("examples/llama.cpp/llama.cpp/models/ggml-vocab-bert-bge.gguf.out");
        Assert.SkipUnless(inp != null && outp != null, "llama.cpp BGE vocab golden not found");
        var tok = Load(repoDir);

        var inputs = File.ReadAllText(inp!).Replace("\r\n", "\n").Split("\n__ggml_vocab_test__\n");
        var expected = File.ReadAllLines(outp!);
        int n = Math.Min(inputs.Length, expected.Length), mismatches = 0;
        var report = new StringBuilder();
        for (int i = 0; i < n; i++)
        {
            var enc = tok.Encode(inputs[i]);
            Assert.Equal(101, enc.Ids[0]);
            Assert.Equal(102, enc.Ids[^1]);
            string ours = string.Join(" ", enc.Ids[1..^1]);
            if (ours == expected[i].Trim()) continue;
            if (++mismatches <= 5) report.AppendLine($"#{i}\n  want: {expected[i].Trim()}\n  ours: {ours}");
        }
        Console.WriteLine($"[EncoderTokenizer] {repoDir}: {n - mismatches}/{n} WordPiece golden cases match");
        Assert.True(mismatches == 0, $"{mismatches}/{n} cases differ:\n{report}");
    }

    [Fact]
    public void BertPair_UsesClsSepTemplateWithSegmentIds()
    {
        var tok = Load("cross-encoder__ms-marco-MiniLM-L6-v2");
        var a = tok.Encode("how many people live in berlin?");
        var b = tok.Encode("berlin has a population of 3,520,031.");
        var pair = tok.EncodePair("how many people live in berlin?", "berlin has a population of 3,520,031.");

        int[] expectedIds = [.. a.Ids[..^1], 102, .. b.Ids[1..]];
        Assert.Equal(expectedIds, pair.Ids);
        int lenA = a.Ids.Length;
        Assert.All(pair.TypeIds[..lenA], t => Assert.Equal(0, t));
        Assert.All(pair.TypeIds[lenA..], t => Assert.Equal(1, t));
        Assert.Equal(0, tok.PadTokenId);
    }

    [Fact]
    public void XlmRobertaPair_UsesDoubleSeparator()
    {
        var tok = Load("BAAI__bge-reranker-v2-m3");
        var single = tok.Encode("what is panda?");
        Assert.Equal(0, single.Ids[0]);
        Assert.Equal(2, single.Ids[^1]);

        var pair = tok.EncodePair("what is panda?", "The giant panda is a bear species endemic to China.");
        var b = tok.Encode("The giant panda is a bear species endemic to China.");
        int[] expectedIds = [.. single.Ids, 2, .. b.Ids[1..]];
        Assert.Equal(expectedIds, pair.Ids);
        Assert.All(pair.TypeIds, t => Assert.Equal(0, t));
        Assert.Equal(1, tok.PadTokenId);
    }

    [Fact]
    public void MpnetRobertaProcessing_WrapsWithSAndSlashS()
    {
        var tok = Load("sentence-transformers__all-mpnet-base-v2");
        var enc = tok.Encode("Hello world");
        Assert.Equal([0, 7596, 2092, 2], enc.Ids);
        Assert.Equal(1, tok.PadTokenId);
    }

    [Fact]
    public void Truncation_IsLongestFirstLikeHfTokenizers()
    {
        var tok = Load("cross-encoder__ms-marco-MiniLM-L6-v2");
        string longA = string.Join(" ", Enumerable.Repeat("apple", 20));
        string longB = string.Join(" ", Enumerable.Repeat("banana", 30));

        Assert.Equal(16, tok.Encode(longA, maxLength: 16).Ids.Length);
        Assert.Equal(102, tok.Encode(longA, maxLength: 16).Ids[^1]);

        // Budget 15 after the 3 specials: both exceed half, so the shorter (A) keeps floor(15/2)=7 and
        // the longer (B) keeps 8.
        var pair = tok.EncodePair(longA, longB, maxLength: 18);
        Assert.Equal(18, pair.Ids.Length);
        Assert.Equal(1 + 7 + 1, pair.TypeIds.Count(t => t == 0));
        Assert.Equal(8 + 1, pair.TypeIds.Count(t => t == 1));

        // Short A (3 tokens) stays whole; B gets the rest of the budget.
        var pair2 = tok.EncodePair("apple apple apple", longB, maxLength: 18);
        Assert.Equal(1 + 3 + 1, pair2.TypeIds.Count(t => t == 0));
        Assert.Equal(12 + 1, pair2.TypeIds.Count(t => t == 1));
    }
}
