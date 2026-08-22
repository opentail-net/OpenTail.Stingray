using System;
using System.IO;
using OpenTail.Stingray.Audio.FishSpeech;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// End-to-end wiring smoke test for <see cref="FishSpeechFullPipeline"/>. Each stage (slow-AR,
/// fast-AR, codec) already has its own real-oracle golden test (cosine similarity) elsewhere in
/// this project -- this test only verifies the PLUMBING that chains them together produces a
/// non-empty, finite, non-silent PCM waveform end-to-end, not new model math.
/// </summary>
public sealed class FishSpeechFullPipelineTests : HeavyTestBase
{
    private static string? FindModelPath(string fileName)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

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

    [Fact]
    public void Synthesize_RealWeights_ProducesFinitePcm()
    {
        string? talkerPath = FindModelPath("s2-pro-q4_k_m.gguf");
        string? codecPath = FindModelPath("s2-pro-q4_k_m.gguf");
        string? tokDir = FindRepoDir("examples/s2.cpp");
        Assert.SkipUnless(talkerPath != null && codecPath != null && tokDir != null,
            "S2 Pro GGUF or examples/s2.cpp not found");

        using var pipeline = new FishSpeechFullPipeline(talkerPath!, tokDir!, codecPath!);
        var pcm = pipeline.Synthesize("Hello, this is a test.", maxTokens: 20);

        Assert.NotEmpty(pcm);
        foreach (var s in pcm)
            Assert.True(float.IsFinite(s), "PCM sample was not finite");

        double sumSq = 0;
        foreach (var s in pcm) sumSq += s * s;
        double rms = Math.Sqrt(sumSq / pcm.Length);
        Assert.True(rms > 1e-6, $"PCM output appears silent (rms={rms})");
    }
}
