using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.StableAudio;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Scratch: regenerates the SFX "glass shatter" and a Small Music sample after implementing the real
/// duration-padding + distribution-shift timestep schedule (see StableAudioScheduleKernels.cs), same
/// prompt/duration/steps/seed as the originals for a direct before/after listening comparison.
/// Delete once the comparison is done.
/// </summary>
public sealed class ZZ_ScratchStableAudioScheduleFixRegenTests
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
    public void Regen_SfxGlassShatter_AfterScheduleFix()
    {
        string? ditDir = FindRepoDir("models/stable-audio-3-small-sfx-base");
        string? t5gemmaDir = FindRepoDir("models/stable-audio-3-t5gemma");
        string? repoRoot = FindRepoRoot();
        Assert.SkipUnless(ditDir != null, "models/stable-audio-3-small-sfx-base not found");
        Assert.SkipUnless(t5gemmaDir != null, "models/stable-audio-3-t5gemma not found");
        Assert.SkipUnless(repoRoot != null, "repo root not found");

        using var ditWeights = SafetensorsLoader.OpenDirectory(ditDir!);
        using var textEncoderWeights = SafetensorsLoader.OpenDirectory(t5gemmaDir!);
        using var pipeline = new StableAudioPipeline(ditWeights, textEncoderWeights, t5gemmaDir!);

        string outDir = Path.Combine(repoRoot!, "docs", "diffusion-samples");
        Directory.CreateDirectory(outDir);
        string outPath = Path.Combine(outDir, "sa3_small-sfx_glass-shatter_3s_v2.wav");

        var pcm = pipeline.Generate(new StableAudioRequest
        {
            Prompt = "a glass bottle shattering on a hard floor",
            DurationSeconds = 3f,
            Steps = 25,
            CfgScale = 6.0f,
            Seed = 1234,
            OutputPath = outPath,
        });

        Assert.True(pcm.Length > 0, "generated zero samples");
        foreach (var v in pcm) Assert.True(float.IsFinite(v), "PCM contains NaN/Inf");
        Assert.True(File.Exists(outPath));
    }

    /// <summary>Real official model-card defaults (`stabilityai/stable-audio-3-small-sfx-base`
    /// README: `steps=50, cfg_scale=7.0`) instead of this project's `steps=25, cfg_scale=6.0`
    /// defaults -- tests whether under-stepped/under-guided generation, not an architecture bug,
    /// explains why SFX sounds worse than Music (which tolerated the same deviation).</summary>
    [Fact]
    public void Regen_SfxGlassShatter_OfficialRecommendedParams()
    {
        string? ditDir = FindRepoDir("models/stable-audio-3-small-sfx-base");
        string? t5gemmaDir = FindRepoDir("models/stable-audio-3-t5gemma");
        string? repoRoot = FindRepoRoot();
        Assert.SkipUnless(ditDir != null, "models/stable-audio-3-small-sfx-base not found");
        Assert.SkipUnless(t5gemmaDir != null, "models/stable-audio-3-t5gemma not found");
        Assert.SkipUnless(repoRoot != null, "repo root not found");

        using var ditWeights = SafetensorsLoader.OpenDirectory(ditDir!);
        using var textEncoderWeights = SafetensorsLoader.OpenDirectory(t5gemmaDir!);
        using var pipeline = new StableAudioPipeline(ditWeights, textEncoderWeights, t5gemmaDir!);

        string outDir = Path.Combine(repoRoot!, "docs", "diffusion-samples");
        Directory.CreateDirectory(outDir);
        string outPath = Path.Combine(outDir, "sa3_small-sfx_glass-shatter_3s_v3_official-params.wav");

        var pcm = pipeline.Generate(new StableAudioRequest
        {
            Prompt = "a glass bottle shattering on a hard floor",
            DurationSeconds = 3f,
            Steps = 50,
            CfgScale = 7.0f,
            Seed = 1234,
            OutputPath = outPath,
        });

        Assert.True(pcm.Length > 0, "generated zero samples");
        foreach (var v in pcm) Assert.True(float.IsFinite(v), "PCM contains NaN/Inf");
        Assert.True(File.Exists(outPath));
    }

    [Fact]
    public void Regen_MusicPianoArpeggio_AfterScheduleFix()
    {
        string? ditDir = FindRepoDir("models/stable-audio-3-small-music-base");
        string? t5gemmaDir = FindRepoDir("models/stable-audio-3-t5gemma");
        string? repoRoot = FindRepoRoot();
        Assert.SkipUnless(ditDir != null, "models/stable-audio-3-small-music-base not found");
        Assert.SkipUnless(t5gemmaDir != null, "models/stable-audio-3-t5gemma not found");
        Assert.SkipUnless(repoRoot != null, "repo root not found");

        using var ditWeights = SafetensorsLoader.OpenDirectory(ditDir!);
        using var textEncoderWeights = SafetensorsLoader.OpenDirectory(t5gemmaDir!);
        using var pipeline = new StableAudioPipeline(ditWeights, textEncoderWeights, t5gemmaDir!);

        string outDir = Path.Combine(repoRoot!, "docs", "diffusion-samples");
        Directory.CreateDirectory(outDir);
        string outPath = Path.Combine(outDir, "sa3_small-music_piano-arpeggio_6s_v3.wav");

        var pcm = pipeline.Generate(new StableAudioRequest
        {
            Prompt = "A beautiful piano arpeggio grows into a grand cinematic climax",
            DurationSeconds = 6f,
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
