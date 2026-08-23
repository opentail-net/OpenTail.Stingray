using System;
using System.Diagnostics;
using System.IO;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Whisper;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMPORARY, throwaway baseline timing bench across every locally-available Whisper GGML
/// checkpoint size (Tiny/Base/Small/Medium/Large-v3), requested explicitly because of Whisper's
/// outsized real-world usage -- see docs/audio-review-progress.md. Fixed ~12s synthetic audio
/// input (VAD disabled, single chunk, greedy decode) per model, reporting wall time, real-time
/// factor (RTF = wallSeconds / audioSeconds; RTF &lt; 1 is faster than real time), and generated
/// token count so throughput is comparable across model sizes. Not part of the permanent suite,
/// delete after use.
/// </summary>
public sealed class WhisperFullPipelinePerfBenchTests : HeavyTestBase
{
    private static readonly (string Label, string FileName)[] Models =
    [
        ("Tiny", "ggml-tiny.bin"),
        ("Base", "ggml-base.bin"),
        ("Small", "ggml-small.bin"),
        ("Medium", "ggml-medium.bin"),
        ("Large-v3", "ggml-large-v3.bin"),
    ];

    [Fact]
    public void Bench_Transcribe_FixedAudio_AcrossModelSizes()
    {
        var report = new System.Text.StringBuilder();
        report.AppendLine($"CPU_THREADS_ENV={Environment.GetEnvironmentVariable("STINGRAY_CPU_THREADS")} ProcessorCount={Environment.ProcessorCount}");

        const int seconds = 12;
        int numSamples = 16000 * seconds;
        float[] audio = new float[numSamples];
        var rng = new Random(42);
        for (int i = 0; i < numSamples; i++)
            audio[i] = MathF.Sin(2.0f * MathF.PI * 220.0f * i / 16000.0f) * 0.3f
                     + MathF.Sin(2.0f * MathF.PI * 880.0f * i / 16000.0f) * 0.1f
                     + (float)(rng.NextDouble() - 0.5) * 0.02f;

        bool anyRan = false;

        foreach (var (label, fileName) in Models)
        {
            string? modelPath = FindModelPath(fileName);
            if (modelPath is null)
            {
                report.AppendLine($"{label}: SKIPPED (models/{fileName} not found)");
                continue;
            }
            anyRan = true;

            using var pipeline = WhisperPipeline.Load(modelPath);

            var request = new SpeechToTextRequest
            {
                AudioSamples = audio,
                SampleRate = 16000,
                Language = "en",
                UseVad = false,
            };

            // Warmup (JIT, thread pool spin-up, weight prefault).
            var warmup = pipeline.Transcribe(request);

            const int n = 3;
            var times = new double[n];
            int tokenCount = 0;
            for (int i = 0; i < n; i++)
            {
                var sw = Stopwatch.StartNew();
                var result = pipeline.Transcribe(request);
                sw.Stop();
                times[i] = sw.Elapsed.TotalMilliseconds;
                tokenCount = result.Text.Length; // proxy; real token count not exposed on result
            }

            Array.Sort(times);
            double mean = 0; foreach (var t in times) mean += t; mean /= n;
            double rtf = (mean / 1000.0) / seconds;

            report.AppendLine($"{label} ({fileName}): audio={seconds}s samples_ms=[{string.Join(", ", Array.ConvertAll(times, t => t.ToF1()))}] " +
                               $"mean_ms={mean:F1} median_ms={times[n / 2]:F1} RTF={rtf:F3} (chars_out={tokenCount})");
        }

        Assert.SkipUnless(anyRan, "No Whisper ggml models found under models/");

        var reportText = report.ToString();
        Console.Error.WriteLine(reportText);
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "whisper_full_bench_result.txt"), reportText);
    }

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
}

file static class DoubleExt
{
    public static string ToF1(this double d) => d.ToString("F1");
}
