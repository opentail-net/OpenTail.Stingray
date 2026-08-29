
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY, throwaway baseline timing bench for CLAUDE.md rule 7's pre-optimization performance pass -- not part of the permanent suite, delete after use. Small n/maxTokens deliberately, since the full autoregressive generation loop is expensive per call (each token re-runs the fast-AR from scratch, see FishSpeechFastAr's own doc comment).</summary>
public sealed class FishSpeechFullPipelinePerfBenchTests : HeavyTestBase
{
    [Fact]
    public void Bench_Synthesize_ShortText()
    {
        string? talkerPath = FindModelPath("s2-pro-q4_k_m.gguf");
        string? tokDir = FindRepoDir("examples/s2.cpp");
        Assert.SkipUnless(talkerPath != null && tokDir != null, "S2 Pro GGUF or examples/s2.cpp not found");

        using var pipeline = new FishSpeechFullPipeline(talkerPath!, tokDir!, talkerPath!);

        pipeline.Synthesize("Hello there.", maxTokens: 15); // warmup

        const int n = 3;
        var times = new double[n];
        for (int i = 0; i < n; i++)
        {
            var sw = Stopwatch.StartNew();
            pipeline.Synthesize("Hello there.", maxTokens: 15);
            sw.Stop();
            times[i] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(times);
        double mean = 0; foreach (var t in times) mean += t; mean /= n;
        var report = $"CPU_THREADS_ENV={Environment.GetEnvironmentVariable("STINGRAY_CPU_THREADS")} ProcessorCount={Environment.ProcessorCount}\n" +
                     $"maxTokens=15\n" +
                     $"samples_ms=[{string.Join(", ", Array.ConvertAll(times, t => t.ToString("F1")))}]\n" +
                     $"mean_ms={mean:F2} median_ms={times[n / 2]:F2}\n";
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "fishspeech_full_bench_result.txt"), report);
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
}
