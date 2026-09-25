namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Golden id parity for <see cref="OpenTail.Stingray.Core.BertWordPieceTokenizer"/> against llama.cpp's
/// own tokenizer test vectors for the BERT/BGE vocab (`examples/llama.cpp/llama.cpp/models/
/// ggml-vocab-bert-bge.gguf.inp` / `.out`: inputs separated by `__ggml_vocab_test__`, expected ids
/// without [CLS]/[SEP]), using the real bge-small-en-v1.5 `vocab.txt` (the same 30522-entry uncased
/// vocab). Covers whitespace edge cases, accents, punctuation, emoji, CJK and code-ish text.
/// Skips visibly when either file is missing.
/// </summary>
public sealed class BertWordPieceTokenizerGoldenTests
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

    [Fact]
    public void Encode_MatchesLlamaCppBgeVocabGolden()
    {
        string? inp = FindRepoFile("examples/llama.cpp/llama.cpp/models/ggml-vocab-bert-bge.gguf.inp");
        string? outp = FindRepoFile("examples/llama.cpp/llama.cpp/models/ggml-vocab-bert-bge.gguf.out");
        string? vocab = FindRepoFile("models/_models/hf/BAAI__bge-small-en-v1.5/vocab.txt");
        Assert.SkipUnless(inp != null && outp != null && vocab != null, "llama.cpp BGE vocab golden or bge-small vocab.txt not found");

        // The upstream file is LF-only; a Windows checkout may have turned it into CRLF.
        var inputs = File.ReadAllText(inp!).Replace("\r\n", "\n").Split("\n__ggml_vocab_test__\n");
        var expected = File.ReadAllLines(outp!);
        var tok = OpenTail.Stingray.Core.BertWordPieceTokenizer.LoadVocabFile(vocab!, doLowerCase: true);

        int n = Math.Min(inputs.Length, expected.Length), mismatches = 0;
        var report = new System.Text.StringBuilder();
        for (int i = 0; i < n; i++)
        {
            var ids = tok.Encode(inputs[i]);
            string ours = string.Join(" ", ids[1..^1]);
            string want = expected[i].Trim();
            if (ours != want)
            {
                mismatches++;
                string shown = inputs[i].Replace("\n", "\\n").Replace("\t", "\\t");
                if (mismatches <= 10) report.AppendLine($"#{i} input=\"{shown}\"\n  want: {want}\n  ours: {ours}");
            }
        }
        Console.WriteLine($"[WordPieceGolden] {n - mismatches}/{n} cases match");
        Assert.True(mismatches == 0, $"{mismatches}/{n} cases differ:\n{report}");
    }
}
