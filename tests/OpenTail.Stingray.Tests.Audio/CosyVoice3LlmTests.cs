
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real structural test for <see cref="CosyVoice3Llm.GenerateSpeechTokens"/> -- the real
/// autoregressive generation loop over CosyVoice3's LLM backbone, real prompt composition
/// (instruction prefix + endofprompt + synthesis text + sos/task speech tokens) transcribed from
/// `examples/cosyvoice.cpp`'s `cosyvoice-llm-job.cpp`. Not yet golden-verified against a numeric
/// oracle (no real Python CosyVoice3 reference confirmed runnable locally) -- checks the loop
/// runs to completion on real weights and produces a real, in-range, non-degenerate token
/// sequence, matching this doc's established first-pass bar for a from-scratch generation loop.
/// </summary>
public sealed class CosyVoice3LlmTests : HeavyTestBase
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
    public void GenerateSpeechTokens_RealWeights_ProducesInRangeNonDegenerateTokenSequence()
    {
        string? path = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(path != null, "models/cosyvoice3/CosyVoice3-2512_F16.gguf not found");

        using var rawModel = GgufModel.Open(path!);
        using var source = new CosyVoice3LlmTensorSource(rawModel);
        source.EnableSpeechGenerationMode();

        var tokens = CosyVoice3Llm.GenerateSpeechTokens(rawModel, source, "Hello there, this is a test.", maxNewTokens: 20);

        Assert.True(tokens.Length > 0, "generation produced zero speech tokens");
        Assert.True(tokens.Length <= 20);

        var stopTokenIds = rawModel.Metadata.TryGetValue("stop_token_ids", out var raw) && raw is object[] arr
            ? new HashSet<int>(Array.ConvertAll(arr, System.Convert.ToInt32))
            : [];

        foreach (var t in tokens)
        {
            Assert.InRange(t, 0, source.SpeechVocabSize - 1);
            Assert.False(stopTokenIds.Contains(t), $"a stop token ({t}) leaked into the returned speech token sequence");
        }

        // Non-degenerate: real weights over real varied input shouldn't collapse to one repeated id.
        Assert.True(new HashSet<int>(tokens).Count > 1 || tokens.Length == 1, "generated sequence looks degenerate (all-identical tokens)");
    }
}
