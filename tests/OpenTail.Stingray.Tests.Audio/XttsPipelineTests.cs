using OpenTail.Stingray.Audio.Xtts;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>End-to-end smoke test for <see cref="XttsPipeline"/> wiring every real, individually
/// golden-verified XTTS-v2 stage together (see `docs/audio-review-progress.md`'s XTTS-v2 entries
/// for each stage's own numeric golden verification). This test checks the FULL pipeline runs on
/// real weights + a real reference clip and produces finite, non-trivial audio -- it does not
/// itself re-verify per-stage numerics (already covered by each stage's own golden test).</summary>
public sealed class XttsPipelineTests : HeavyTestBase
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
    public void Generate_RealWeightsRealReference_ProducesFiniteBoundedWaveform()
    {
        string? checkpointDir = FindRepoFile("models/xtts-v2/model.safetensors") is { } p ? Path.GetDirectoryName(p) : null;
        Assert.SkipUnless(checkpointDir != null, "models/xtts-v2/ checkpoint not found");
        string? refWav = FindRepoFile("docs/audio-samples/fishspeech-lunch-REFERENCE.wav");
        Assert.SkipUnless(refWav != null, "reference audio clip not found");

        var pipeline = XttsPipeline.Load(checkpointDir!);
        float[] waveform = pipeline.Generate("This is a short test.", refWav!, "en", seed: 42);

        Assert.True(waveform.Length > 1000, $"expected non-trivial audio output, got {waveform.Length} samples");
        foreach (float v in waveform)
        {
            Assert.False(float.IsNaN(v), "waveform must not contain NaN");
            Assert.False(float.IsInfinity(v), "waveform must not contain Infinity");
            Assert.InRange(v, -1.5f, 1.5f);
        }
    }
}
