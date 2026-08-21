using System;
using System.IO;
using OpenTail.Stingray.Audio.Chatterbox;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Vocoder-only perf hotloop: real S3Gen weights, but a synthetic mel input at a fixed, realistic
/// scale (matching what a real ~5-10s sentence produces: ~250 generated mel frames) instead of
/// running the flow encoder + CFM decoder to get there. This isolates ChatterboxVocoder.Generate's
/// own cost (~50-75s in the full pipeline) from T3/encoder/CFM (~65s combined), so iterating on
/// vocoder performance doesn't need a ~2-minute full-pipeline run each time. Correctness of the
/// mel *content* doesn't matter here (it's random noise, not real conditioning) -- this is purely
/// for timing. See ChatterboxVocoderTests for the real correctness check.
/// </summary>
public sealed class ChatterboxVocoderBenchmarkTests : HeavyTestBase
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
    public void ChatterboxVocoder_Benchmark_SyntheticMelAtRealisticScale()
    {
        string? s3GenPath = FindRepoFile("models/chatterbox-turbo-s3gen-q4_k.gguf");
        if (s3GenPath is null) return;

        using var w = new ChatterboxS3GenWeights(s3GenPath);

        // ~250 mel frames matches a real few-second sentence's generated (non-prompt) tail --
        // see the "generated mel frames" figures logged by STINGRAY_AUDIO_DIAGNOSTIC_DUMP=1 runs.
        const int melFrames = 250;
        var mel = new float[w.MelChannels * melFrames];
        var rng = new Random(42);
        for (int i = 0; i < mel.Length; i++) mel[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var waveform = ChatterboxVocoder.Generate(w, mel, melFrames, rng);
        sw.Stop();

        Console.Error.WriteLine($"[VocoderBenchmark] {melFrames} mel frames -> {waveform.Length} samples in {sw.ElapsedMilliseconds}ms");
        try
        {
            File.AppendAllText(
                Path.Combine(Path.GetTempPath(), "stingray-chatterbox-diag.log"),
                $"[VocoderBenchmark] {melFrames} mel frames -> {waveform.Length} samples in {sw.ElapsedMilliseconds}ms{Environment.NewLine}");
        }
        catch { /* best-effort */ }

        Assert.NotEmpty(waveform);
    }
}
