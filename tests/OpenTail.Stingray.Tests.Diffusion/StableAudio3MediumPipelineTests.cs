using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.StableAudio;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// First real, genuine end-to-end Stable Audio 3 Medium smoke test: real T5Gemma text encoder
/// (shared, identical checkpoint to Small), real differential-attention DiT, real SAME-L VAE
/// decoder -- all wired through <see cref="StableAudioMediumPipeline"/>. Real weights, short
/// duration to keep wall-clock low (Medium is a real, much larger 24-layer/1536-dim model, plus
/// real CFG runs the DiT twice per step). Non-degeneracy receipt, not yet a numeric golden-parity
/// test -- see docs/057-stable-audio-3-implementation-plan.md.
/// </summary>
public sealed class StableAudio3MediumPipelineTests
{
    private static string? FindRepoDir(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath.Replace('/', Path.DirectorySeparatorChar));
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Generate_RealMediumWeights_ProducesFiniteNonSilentAudio()
    {
        string? ditDir = FindRepoDir("models/stable-audio-3-medium-base");
        string? t5gemmaDir = FindRepoDir("models/stable-audio-3-t5gemma");
        Assert.SkipUnless(ditDir != null, "models/stable-audio-3-medium-base not found");
        Assert.SkipUnless(t5gemmaDir != null, "models/stable-audio-3-t5gemma not found");

        using var ditWeights = SafetensorsLoader.OpenDirectory(ditDir!);
        using var textEncoderWeights = SafetensorsLoader.OpenDirectory(t5gemmaDir!);
        using var pipeline = new StableAudioMediumPipeline(ditWeights, textEncoderWeights, t5gemmaDir);

        var pcm = pipeline.Generate(new StableAudioRequest
        {
            Prompt = "a beautiful piano arpeggio",
            DurationSeconds = 1f,
            Steps = 4, // short on purpose -- real smoke test, not a quality benchmark
            CfgScale = 6.0f,
            Seed = 1234,
            OutputPath = "",
        });

        Assert.True(pcm.Length > 0, "generated zero samples");
        foreach (var v in pcm) Assert.True(float.IsFinite(v), "PCM contains NaN/Inf");

        double sumSq = 0;
        foreach (var v in pcm) sumSq += (double)v * v;
        double rms = Math.Sqrt(sumSq / pcm.Length);
        Assert.True(rms > 1e-6, $"generated audio RMS ({rms}) is near-silent -- likely a wiring bug");
    }
}
