using System;
using System.IO;
using OpenTail.Stingray.Audio.QwenASR;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Perf hotloop for QwenASR's real-weight AuT audio encoder, mirroring
/// <see cref="ChatterboxVocoderBenchmarkTests"/>/<see cref="CosyVoiceBenchmarkTests"/>'s shape:
/// real weights, synthetic-but-realistically-scaled mel input, timing logged to a temp diag
/// file for before/after comparison across optimization passes. Correctness of the mel
/// *content* doesn't matter here -- see QwenAsrAudioEncoderTests for the real correctness check.
/// </summary>
public sealed class QwenAsrBenchmarkTests : HeavyTestBase
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

    private static void LogTiming(string label, System.Diagnostics.Stopwatch sw)
    {
        Console.Error.WriteLine($"[QwenAsrBenchmark] {label}: {sw.ElapsedMilliseconds}ms");
        try
        {
            File.AppendAllText(
                Path.Combine(Path.GetTempPath(), "stingray-qwenasr-diag.log"),
                $"[QwenAsrBenchmark] {label}: {sw.ElapsedMilliseconds}ms{Environment.NewLine}");
        }
        catch { /* best-effort */ }
    }

    [Fact]
    public void AudioEncoder_Benchmark_RealisticClipLength()
    {
        string? path = FindRepoFile("models/qwen3-asr-0.6b-q4_k.gguf");
        if (path is null) return;

        using var w = new QwenAsrWeights(path);
        var config = new QwenAsrEncoderConfig
        {
            EncoderDim = w.AudioDim,
            NumLayers = w.AudioLayers,
            NumHeads = w.AudioHeads,
            QwenHiddenDim = w.LlmDim,
        };
        var encoder = new QwenAsrAudioEncoder(config, w);

        // ~10s of audio at a 10ms mel hop -- realistic single-utterance scale.
        const int melFrames = 1000;
        var mel = new float[w.NMels * melFrames];
        var rng = new Random(42);
        for (int i = 0; i < mel.Length; i++) mel[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var (tokens, numTokens) = encoder.Forward(mel, melFrames);
        sw.Stop();

        LogTiming($"AudioEncoder.Forward {melFrames} mel frames -> {numTokens} audio tokens", sw);
        Assert.NotEmpty(tokens);
    }
}
