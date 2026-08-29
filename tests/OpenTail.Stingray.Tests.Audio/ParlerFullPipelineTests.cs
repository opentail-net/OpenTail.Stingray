
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// End-to-end wiring smoke test for <see cref="ParlerFullPipeline"/>. Every component (tokenizer,
/// T5 encoder, decoder forward pass, KV cache, delay pattern, EOS logits processor, DAC decoder)
/// already has its own real-oracle golden test elsewhere in this project -- this test verifies
/// the PLUMBING that chains them into one real autoregressive generation loop produces a
/// non-empty, finite, non-silent PCM waveform, not new model math. Deliberately allows a large
/// `maxNewTokens` budget since Parler-TTS's real `min_new_tokens=10` default means the model must
/// run for a while before it's even permitted to stop.
/// </summary>
public sealed class ParlerFullPipelineTests : HeavyTestBase
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
    public void Synthesize_RealWeights_ProducesFinitePcm()
    {
        string? modelPath = FindRepoFile("models/parler-tts-mini-v1.safetensors");
        string? tokenizerPath = FindRepoFile("scratch-llamacpp-ref/parler-tokenizer/tokenizer.json");
        Assert.SkipUnless(modelPath != null && tokenizerPath != null,
            "models/parler-tts-mini-v1.safetensors or the real Parler tokenizer.json not found");

        using var loader = SafetensorsLoader.Open(modelPath!);
        using var pipeline = new ParlerFullPipeline(tokenizerPath!, loader);

        var pcm = pipeline.Synthesize("Hello there.", maxNewTokens: 40, minNewTokens: 10);

        Assert.NotEmpty(pcm);
        foreach (var s in pcm)
            Assert.True(float.IsFinite(s), "PCM sample was not finite");

        double sumSq = 0;
        foreach (var s in pcm) sumSq += s * s;
        double rms = Math.Sqrt(sumSq / pcm.Length);
        Assert.True(rms > 1e-6, $"PCM output appears silent (rms={rms})");
    }
}
