
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY, throwaway baseline timing bench for CLAUDE.md rule 7's pre-optimization performance pass -- not part of the permanent suite, delete after use.</summary>
public sealed class SileroVadPerfBenchTests : HeavyTestBase
{
    [Fact]
    public void Bench_DetectSegments_LongerAudio()
    {
        string? modelPath = FindModelPath("silero_vad.onnx");
        Assert.SkipUnless(modelPath != null, "models/silero_vad.onnx not found");

        var vad = SileroVad.Load(modelPath!);

        int numSamples = 16000 * 12;
        float[] audio = new float[numSamples];
        var rng = new Random(42);
        for (int i = 0; i < numSamples; i++)
            audio[i] = MathF.Sin(2.0f * MathF.PI * 220.0f * i / 16000.0f) * 0.3f
                     + MathF.Sin(2.0f * MathF.PI * 880.0f * i / 16000.0f) * 0.1f
                     + (float)(rng.NextDouble() - 0.5) * 0.02f;

        vad.DetectSegments(audio); // warmup

        const int n = 8;
        var times = new double[n];
        for (int i = 0; i < n; i++)
        {
            var sw = Stopwatch.StartNew();
            vad.DetectSegments(audio);
            sw.Stop();
            times[i] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(times);
        double mean = 0; foreach (var t in times) mean += t; mean /= n;
        var report = $"CPU_THREADS_ENV={Environment.GetEnvironmentVariable("STINGRAY_CPU_THREADS")} ProcessorCount={Environment.ProcessorCount}\n" +
                     $"samples_ms=[{string.Join(", ", Array.ConvertAll(times, t => t.ToString("F1")))}]\n" +
                     $"mean_ms={mean:F2} median_ms={times[n / 2]:F2}\n";
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "silerovad_bench_result.txt"), report);
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
