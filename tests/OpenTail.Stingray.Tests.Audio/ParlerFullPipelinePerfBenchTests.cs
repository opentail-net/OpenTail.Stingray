
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY, throwaway baseline timing bench for CLAUDE.md rule 7's pre-optimization performance pass -- not part of the permanent suite, delete after use.</summary>
public sealed class ParlerFullPipelinePerfBenchTests : HeavyTestBase
{
    [Fact]
    public void Bench_Synthesize_ShortText()
    {
        string? modelPath = FindModelPath("parler-tts-mini-v1.safetensors");
        string? tokenizerPath = FindRepoFile("scratch-llamacpp-ref/parler-tokenizer/tokenizer.json");
        Assert.SkipUnless(modelPath != null && tokenizerPath != null,
            "models/parler-tts-mini-v1.safetensors or the real Parler tokenizer.json not found");

        using var loader = SafetensorsLoader.Open(modelPath!);
        using var pipeline = new ParlerFullPipeline(tokenizerPath!, loader);

        pipeline.Synthesize("Hello there.", maxNewTokens: 30, minNewTokens: 10); // warmup

        const int n = 5;
        var times = new double[n];
        for (int i = 0; i < n; i++)
        {
            var sw = Stopwatch.StartNew();
            pipeline.Synthesize("Hello there.", maxNewTokens: 30, minNewTokens: 10);
            sw.Stop();
            times[i] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(times);
        double mean = 0; foreach (var t in times) mean += t; mean /= n;
        var report = $"CPU_THREADS_ENV={Environment.GetEnvironmentVariable("STINGRAY_CPU_THREADS")} ProcessorCount={Environment.ProcessorCount}\n" +
                     $"maxNewTokens=30 minNewTokens=10\n" +
                     $"samples_ms=[{string.Join(", ", Array.ConvertAll(times, t => t.ToString("F1")))}]\n" +
                     $"mean_ms={mean:F2} median_ms={times[n / 2]:F2}\n";
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "parler_full_bench_result.txt"), report);
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
}
