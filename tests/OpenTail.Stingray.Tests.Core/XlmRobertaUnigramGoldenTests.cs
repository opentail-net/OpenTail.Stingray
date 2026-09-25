namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Golden id parity for <see cref="UnigramTokenizer"/> on the XLM-RoBERTa SentencePiece vocab (250002
/// pieces, shared by xlm-roberta, multilingual-e5, bge-m3/bge-reranker-v2-m3 and
/// paraphrase-multilingual-MiniLM). Expected ids come from llama.cpp's UGM tokenizer
/// (<c>tools/llama.cpp/llama-tokenize.exe --ids --no-escape</c> on gpustack/bge-reranker-v2-m3-GGUF
/// Q8_0, 2026-09-25), stored as <c>Fixtures/xlm-roberta-ugm.inp/.out</c> in llama.cpp's own vocab-test
/// format, with the leading &lt;s&gt; and trailing &lt;/s&gt; removed. Inputs are llama.cpp's
/// <c>ggml-vocab-llama-spm.gguf.inp</c> set plus multilingual cases that exercise the precompiled charsmap
/// (fullwidth, circled digits, ligatures, NBSP/em space, zero-width chars, CJK/Hangul/RTL). Each
/// checkpoint's tokenizer.json is loaded from <c>models/_models/hf</c> and the test skips visibly
/// when it is missing.
/// </summary>
public sealed class XlmRobertaUnigramGoldenTests
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

    [Theory]
    [InlineData("BAAI__bge-reranker-v2-m3")]
    [InlineData("FacebookAI__xlm-roberta-base")]
    [InlineData("sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2")]
    [InlineData("intfloat__multilingual-e5-small")]
    public void Encode_MatchesLlamaCppUgmGolden(string repoDir)
    {
        string? inp = FindRepoFile("tests/OpenTail.Stingray.Tests.Core/Fixtures/xlm-roberta-ugm.inp");
        string? outp = FindRepoFile("tests/OpenTail.Stingray.Tests.Core/Fixtures/xlm-roberta-ugm.out");
        string? tokJson = FindRepoFile($"models/_models/hf/{repoDir}/tokenizer.json");
        Assert.SkipUnless(inp != null && outp != null && tokJson != null, $"golden fixture or {repoDir}/tokenizer.json not found");

        var inputs = File.ReadAllText(inp!).Replace("\r\n", "\n").Split("\n__ggml_vocab_test__\n");
        var expected = File.ReadAllLines(outp!);
        Assert.Equal(expected.Length, inputs.Length);
        var tok = UnigramTokenizer.FromTokenizerJson(tokJson!);

        int mismatches = 0;
        var report = new StringBuilder();
        for (int i = 0; i < inputs.Length; i++)
        {
            // The e5 config (Replace " {2,}" -> " ", no Strip/WhitespaceSplit) keeps trailing whitespace as
            // its own "▁" piece, where llama.cpp drops it; compare that model on right-trimmed input.
            string text = repoDir == "intfloat__multilingual-e5-small" ? inputs[i].TrimEnd() : inputs[i];
            string ours = string.Join(" ", tok.Encode(text));
            string want = expected[i].Trim();
            if (ours == want) continue;
            mismatches++;
            if (mismatches <= 10)
                report.AppendLine($"#{i} input=\"{inputs[i].Replace("\n", "\\n").Replace("\t", "\\t")}\"\n  want: {want}\n  ours: {ours}");
        }
        Console.WriteLine($"[XlmrUgmGolden] {repoDir}: {inputs.Length - mismatches}/{inputs.Length} cases match");
        Assert.True(mismatches == 0, $"{mismatches}/{inputs.Length} cases differ:\n{report}");
    }
}
