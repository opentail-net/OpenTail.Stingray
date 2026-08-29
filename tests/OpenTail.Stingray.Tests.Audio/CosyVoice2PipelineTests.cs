
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real end-to-end proof for the full CosyVoice2 pipeline (see docs/audio-review-progress.md's
/// CosyVoice section): text -&gt; LLM speech tokens -&gt; flow encoder -&gt; CFM decoder -&gt; HiFT vocoder,
/// all on real weights. NOT yet golden-verified against a real oracle -- structurally complete,
/// same bar every other pipeline's first end-to-end test passed before golden verification
/// followed.
/// </summary>
public sealed class CosyVoice2PipelineTests : HeavyTestBase
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
    public void Generate_RealWeights_ProducesFiniteNonSilentAudio()
    {
        string? llmPath = FindRepoFile("models/cosyvoice2_llm.safetensors");
        string? tokenizerDir = FindRepoFile("models/cosyvoice2_tokenizer");
        string? flowPath = FindRepoFile("models/cosyvoice2_flow.safetensors");
        string? hiftPath = FindRepoFile("models/cosyvoice2_hift.safetensors");
        Assert.SkipUnless(llmPath != null, "models/cosyvoice2_llm.safetensors not found");
        Assert.SkipUnless(tokenizerDir != null, "models/cosyvoice2_tokenizer not found");
        Assert.SkipUnless(flowPath != null, "models/cosyvoice2_flow.safetensors not found");
        Assert.SkipUnless(hiftPath != null, "models/cosyvoice2_hift.safetensors not found");

        using var pipeline = CosyVoice2Pipeline.Load(llmPath!, tokenizerDir!, flowPath!, hiftPath!);

        var audio = pipeline.Generate("Hello world, this is a real test.", maxNewSpeechTokens: 30, odeSteps: 10, seed: 1234);

        Assert.True(audio.Length > 0, "Expected non-empty generated audio.");
        foreach (var s in audio) Assert.False(float.IsNaN(s) || float.IsInfinity(s));

        bool anyNonZero = false;
        foreach (var s in audio) if (MathF.Abs(s) > 1e-6f) { anyNonZero = true; break; }
        Assert.True(anyNonZero, "Generated audio looks degenerate (all zero).");
    }
}
