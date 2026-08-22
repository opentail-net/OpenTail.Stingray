using System;
using System.Diagnostics;
using System.IO;
using OpenTail.Stingray.Audio.Orpheus;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY, throwaway baseline timing bench for CLAUDE.md rule 7's pre-optimization performance pass -- not part of the permanent suite, delete after use.</summary>
public sealed class OrpheusFullPipelinePerfBenchTests : HeavyTestBase
{
    [Fact]
    public void Bench_Synthesize_ShortText()
    {
        string? talkerPath = FindModelPath("orpheus-3b-0.1-ft.Q4_K_M.gguf");
        string? snacPath = FindModelPath("snac-24khz.gguf");
        Assert.SkipUnless(talkerPath != null && snacPath != null, "Orpheus/SNAC GGUFs not found");

        using var pipeline = new OrpheusPipeline(talkerPath!, snacPath!);

        pipeline.Synthesize("Hello there.", maxTokens: 140); // warmup, ~20 real superframes

        const int n = 5;
        var times = new double[n];
        for (int i = 0; i < n; i++)
        {
            var sw = Stopwatch.StartNew();
            pipeline.Synthesize("Hello there.", maxTokens: 140);
            sw.Stop();
            times[i] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(times);
        double mean = 0; foreach (var t in times) mean += t; mean /= n;
        var report = $"CPU_THREADS_ENV={Environment.GetEnvironmentVariable("STINGRAY_CPU_THREADS")} ProcessorCount={Environment.ProcessorCount}\n" +
                     $"maxTokens=140\n" +
                     $"samples_ms=[{string.Join(", ", Array.ConvertAll(times, t => t.ToString("F1")))}]\n" +
                     $"mean_ms={mean:F2} median_ms={times[n / 2]:F2}\n";
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "orpheus_full_bench_result.txt"), report);
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
