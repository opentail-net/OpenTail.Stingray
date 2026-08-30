using OpenTail.Stingray.Audio.Xtts;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Exact-match golden verification for <see cref="XttsBpeTokenizer"/> against the real
/// `tokenizers.Tokenizer.from_file("models/xtts-v2/vocab.json").encode(text).ids` output, confirmed
/// directly via the real Python `tokenizers` library (see the commit message / progress doc for the
/// exact verification script). Text is pre-normalized (already what `VoiceBpeTokenizer.encode`
/// would pass in after its `[lang]` prefix + `" "`-&gt;`"[SPACE]"` substitution) to isolate the
/// tokenizer algorithm itself from the separate multilingual text-cleaning pass.</summary>
public sealed class XttsBpeTokenizerTests : HeavyTestBase
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

    [Theory]
    [InlineData("[en]hello world[SPACE]this is a test.", new[] { 259, 62, 84, 28, 179, 79, 2, 147, 54, 14, 136, 63, 9 })]
    [InlineData("[en]the quick brown fox jumps over the lazy dog.", new[] { 259, 42, 194, 91, 24, 243, 190, 182, 37, 1081, 26, 1093, 5555, 44, 42, 494, 2457, 6064, 9 })]
    [InlineData("[fr]bonjour le monde, comment allez-vous?", new[] { 262, 871, 818, 64, 1031, 7, 174, 426, 33, 14, 84, 933, 8, 955, 32, 13 })]
    [InlineData("[en]a", new[] { 259, 14 })]
    [InlineData("[en]", new[] { 259 })]
    public void Encode_RealSentences_MatchesGoldenOracle(string text, int[] expected)
    {
        string? vocabPath = FindRepoFile("models/xtts-v2/vocab.json");
        Assert.SkipUnless(vocabPath != null, "models/xtts-v2/vocab.json not found");

        var tokenizer = new XttsBpeTokenizer(vocabPath!);
        var ids = tokenizer.Encode(text);

        Assert.Equal(expected, ids);
    }
}
