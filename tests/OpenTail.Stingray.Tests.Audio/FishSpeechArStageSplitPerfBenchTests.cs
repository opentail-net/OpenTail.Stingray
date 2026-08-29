
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY, throwaway perf-split bench (CLAUDE.md rule 7): isolates the fast-AR's own
/// per-frame cost (10 sequential ForwardStep calls, real weights, no slow-AR involved) against
/// the combined slow-AR+fast-AR per-frame cost from GenerateFrames, to find where the AR
/// generation loop's dominant time actually goes before optimizing blindly.</summary>
public sealed class FishSpeechArStageSplitPerfBenchTests : HeavyTestBase
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
    public void Bench_FastArAlone_PerFrame()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        Assert.SkipUnless(modelPath != null, "S2 Pro GGUF not found");

        using var weights = new FishSpeechWeights(modelPath!);
        var cache = new FishSpeechFastArCache(weights);
        var rnd = new Random(42);
        var hidden = new float[weights.EmbeddingDim];
        for (int i = 0; i < hidden.Length; i++) hidden[i] = (float)(rnd.NextDouble() * 2 - 1);

        void OneFrame()
        {
            cache.Reset();
            var stepLogits = FishSpeechFastAr.ForwardStep(weights, cache, hidden);
            for (int cb = 1; cb < weights.NumCodebooks; cb++)
            {
                int tok = cb; // fixed, deterministic -- only timing matters here
                if (cb < weights.NumCodebooks - 1)
                    stepLogits = FishSpeechFastAr.ForwardStep(weights, cache, FishSpeechFastAr.EmbedFastToken(weights, tok));
            }
        }

        OneFrame(); // warmup (JIT)

        const int n = 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < n; i++) OneFrame();
        sw.Stop();

        var report = $"fastArOnly: n={n} totalMs={sw.Elapsed.TotalMilliseconds:F1} msPerFrame={sw.Elapsed.TotalMilliseconds / n:F2}\n";
        Console.WriteLine(report);
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "fishspeech_fastar_only_bench_result.txt"), report);
    }

    [Fact]
    public void Bench_FullGenerateFrames_PerFrame()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        string? tokDir = FindRepoDir("examples/s2.cpp");
        Assert.SkipUnless(modelPath != null && tokDir != null, "S2 Pro GGUF or examples/s2.cpp not found");

        using var pipeline = new FishSpeechPipeline(modelPath!, tokDir!);
        pipeline.GenerateFrames("Warmup.", maxTokens: 5); // warmup

        const int n = 20;
        var sw = Stopwatch.StartNew();
        pipeline.GenerateFrames("This is a test of speech synthesis timing for the benchmark run today.", maxTokens: n);
        sw.Stop();

        var report = $"slowArPlusFastAr: n={n} totalMs={sw.Elapsed.TotalMilliseconds:F1} msPerFrame={sw.Elapsed.TotalMilliseconds / n:F2}\n";
        Console.WriteLine(report);
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "fishspeech_full_ar_bench_result.txt"), report);
    }
}
