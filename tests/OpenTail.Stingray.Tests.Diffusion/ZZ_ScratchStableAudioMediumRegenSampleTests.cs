using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.StableAudio;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Scratch: regenerates the Medium "piano arpeggio" sample after the `sinusoidal_blocks` FeedForward
/// fix (see SameLargeVae.cs), same prompt/duration/steps/seed as the original
/// docs/diffusion-samples/sa3_medium_piano-arpeggio_4s.wav for a direct before/after listening
/// comparison. Delete once the comparison is done.
/// </summary>
public sealed class ZZ_ScratchStableAudioMediumRegenSampleTests
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

    private static string? FindRepoRoot()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            if (Directory.Exists(Path.Combine(dir, "docs"))) return dir;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Regen_PianoArpeggio_AfterSinusoidalFfFix()
    {
        string? ditDir = FindRepoDir("models/stable-audio-3-medium-base");
        string? t5gemmaDir = FindRepoDir("models/stable-audio-3-t5gemma");
        string? repoRoot = FindRepoRoot();
        Assert.SkipUnless(ditDir != null, "models/stable-audio-3-medium-base not found");
        Assert.SkipUnless(t5gemmaDir != null, "models/stable-audio-3-t5gemma not found");
        Assert.SkipUnless(repoRoot != null, "repo root not found");

        using var ditWeights = SafetensorsLoader.OpenDirectory(ditDir!);
        using var textEncoderWeights = SafetensorsLoader.OpenDirectory(t5gemmaDir!);
        using var pipeline = new StableAudioMediumPipeline(ditWeights, textEncoderWeights, t5gemmaDir!);

        string outDir = Path.Combine(repoRoot!, "docs", "diffusion-samples");
        Directory.CreateDirectory(outDir);
        string outPath = Path.Combine(outDir, "sa3_medium_piano-arpeggio_4s_v2.wav");

        var pcm = pipeline.Generate(new StableAudioRequest
        {
            Prompt = "a beautiful piano arpeggio",
            DurationSeconds = 4f,
            Steps = 25,
            CfgScale = 6.0f,
            Seed = 1234,
            OutputPath = outPath,
        });

        Assert.True(pcm.Length > 0, "generated zero samples");
        foreach (var v in pcm) Assert.True(float.IsFinite(v), "PCM contains NaN/Inf");
        Assert.True(File.Exists(outPath));
    }
}
