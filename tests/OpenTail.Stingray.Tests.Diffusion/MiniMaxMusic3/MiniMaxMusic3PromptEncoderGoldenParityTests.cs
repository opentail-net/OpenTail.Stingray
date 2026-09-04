using System.Text.Json;
using OpenTail.Stingray.Diffusion.MiniMaxMusic3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real golden-parity check for <see cref="MiniMaxMusic3PromptEncoder"/> against the real
/// `diffusers.modular_pipelines.minimax_music3.encoders._clean_caption`/`_normalize_lyrics` plus
/// the real `transformers.Qwen2Tokenizer` loaded from this checkpoint's own real
/// `tokenizer/tokenizer.json`. See docs/066-minimax-music3-future-plan.md.
/// </summary>
public sealed class MiniMaxMusic3PromptEncoderGoldenParityTests
{
    private sealed record Case(string Prompt, string Lyrics, string Cleaned, string Normalized, string Text, int[] Ids);

    private static string? FindRepoDir(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static List<Case>? LoadCases()
    {
        string path = @"C:\Users\Dmitri\AppData\Local\Temp\claude\C--Git-Public-OpenTail-Stingray\6cb31b57-ce45-49d6-9926-8736cdcfcfa9\scratchpad\mm3_tokenizer_ref_cases.json";
        if (!File.Exists(path)) return null;
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var cases = new List<Case>();
        foreach (var el in doc.RootElement.EnumerateArray())
        {
            var ids = el.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            cases.Add(new Case(
                el.GetProperty("prompt").GetString()!,
                el.GetProperty("lyrics").GetString()!,
                el.GetProperty("cleaned").GetString()!,
                el.GetProperty("normalized").GetString()!,
                el.GetProperty("text").GetString()!,
                ids));
        }
        return cases;
    }

    [Fact]
    public void CleanCaption_And_NormalizeLyrics_MatchRealDiffusersReference()
    {
        var cases = LoadCases();
        Assert.SkipUnless(cases != null, "mm3_tokenizer_ref_cases.json reference dump not found");

        foreach (var c in cases!)
        {
            Assert.Equal(c.Cleaned, MiniMaxMusic3PromptEncoder.CleanCaption(c.Prompt));
            Assert.Equal(c.Normalized, MiniMaxMusic3PromptEncoder.NormalizeLyrics(c.Lyrics));
        }
    }

    [Fact]
    public void BuildConditionalPrompt_RealTokenizer_MatchesRealQwen2TokenizerReference()
    {
        string? tokenizerDir = FindRepoDir("models/minimax-music3/tokenizer");
        Assert.SkipUnless(tokenizerDir != null, "models/minimax-music3/tokenizer/ not found");
        var cases = LoadCases();
        Assert.SkipUnless(cases != null, "mm3_tokenizer_ref_cases.json reference dump not found");

        var encoder = MiniMaxMusic3PromptEncoder.Load(tokenizerDir!);
        foreach (var c in cases!)
        {
            var actual = encoder.BuildConditionalPrompt(c.Prompt, c.Lyrics);
            Assert.Equal(c.Ids, actual);
        }
    }
}
