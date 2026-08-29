
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weights sanity coverage for CosyVoice2's LLM speech-token generation loop (see
/// docs/audio-review-progress.md's CosyVoice section). NOT yet golden-verified against a real
/// oracle -- confirms the real prompt composition (sos/task_id via the separate `llm_embedding`
/// table, real Qwen2 tokenizer) runs end-to-end through the real Engine `ForwardPass` and
/// produces finite, in-range speech token ids.
/// </summary>
public sealed class CosyVoiceLlmGenerationTests : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void GenerateSpeechTokens_RealWeights_ProducesInRangeFiniteTokens()
    {
        string? llmPath = FindRepoFile("models/cosyvoice2_llm.safetensors");
        string? tokenizerDir = FindRepoFile("models/cosyvoice2_tokenizer");
        Assert.SkipUnless(llmPath != null, "models/cosyvoice2_llm.safetensors not found");
        Assert.SkipUnless(tokenizerDir != null, "models/cosyvoice2_tokenizer not found");

        using var source = new CosyVoiceLlmTensorSource(
            llmPath!,
            numLayers: 24, hiddenDim: 896, numHeads: 14, numKvHeads: 2, headDim: 64,
            ffDim: 4864, vocabSize: 151936, ropeTheta: 1_000_000f, rmsNormEps: 1e-6f);

        var tokens = CosyVoiceLlmGeneration.GenerateSpeechTokens(source, tokenizerDir!, "Hello world.", maxNewTokens: 20);

        Assert.True(tokens.Length > 0, "Expected at least one generated speech token.");
        foreach (var t in tokens)
        {
            Assert.InRange(t, 0, 6563);
        }
    }
}
