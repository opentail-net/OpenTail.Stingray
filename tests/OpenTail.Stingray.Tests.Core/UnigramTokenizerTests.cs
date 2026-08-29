
namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Golden verification for <see cref="UnigramTokenizer"/> against the real Hugging Face
/// `tokenizers` Python package (`Tokenizer.from_file(...).encode(text, add_special_tokens=False)
/// .ids`), run on Parler-TTS's real `tokenizer.json`. See `UnigramTokenizer`'s class doc for the
/// real algorithm's source derivation and the documented `precompiled_charsmap` gap (this test
/// set is plain-ASCII by design, where that gap does not apply).
/// </summary>
public sealed class UnigramTokenizerTests
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Encode_RealParlerTokenizer_MatchesGoldenIds()
    {
        string? tokenizerPath = FindRepoFile("scratch-llamacpp-ref/parler-tokenizer/tokenizer.json");
        string? goldenPath = FindRepoFile("scratch-llamacpp-ref/parler-tokenizer/unigram_golden.json");
        Assert.SkipUnless(tokenizerPath != null && goldenPath != null,
            "Parler-TTS real tokenizer.json / golden fixture not found");

        var tok = UnigramTokenizer.FromTokenizerJson(tokenizerPath!);

        using var doc = JsonDocument.Parse(File.ReadAllBytes(goldenPath!));
        foreach (var entry in doc.RootElement.EnumerateArray())
        {
            string text = entry.GetProperty("text").GetString()!;
            var expected = new List<int>();
            foreach (var idEl in entry.GetProperty("ids").EnumerateArray()) expected.Add(idEl.GetInt32());
            // Golden was captured with add_special_tokens=True, which appends T5's real EOS (id 1);
            // UnigramTokenizer.Encode is segmentation-only, so drop the trailing EOS before comparing.
            expected.RemoveAt(expected.Count - 1);

            var actual = tok.Encode(text);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void Encode_RealParlerTokenizer_MatchesGoldenIds_HarderCases()
    {
        string? tokenizerPath = FindRepoFile("scratch-llamacpp-ref/parler-tokenizer/tokenizer.json");
        string? goldenPath = FindRepoFile("scratch-llamacpp-ref/parler-tokenizer/unigram_golden2.json");
        Assert.SkipUnless(tokenizerPath != null && goldenPath != null,
            "Parler-TTS real tokenizer.json / harder golden fixture not found");

        var tok = UnigramTokenizer.FromTokenizerJson(tokenizerPath!);

        using var doc = JsonDocument.Parse(File.ReadAllBytes(goldenPath!));
        foreach (var entry in doc.RootElement.EnumerateArray())
        {
            string text = entry.GetProperty("text").GetString()!;
            var expected = new List<int>();
            foreach (var idEl in entry.GetProperty("ids").EnumerateArray()) expected.Add(idEl.GetInt32());

            var actual = tok.Encode(text);
            Assert.Equal(expected, actual);
        }
    }
}
