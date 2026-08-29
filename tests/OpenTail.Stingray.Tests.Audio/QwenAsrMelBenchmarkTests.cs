
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Perf hotloop for QwenAsrMelExtractor.ExtractMel (no real weights needed -- pure DSP, unlike
/// the other *BenchmarkTests files). Mirrors their timing-to-temp-file pattern for before/after
/// comparison across optimization passes.
/// </summary>
public sealed class QwenAsrMelBenchmarkTests
{
    private static void LogTiming(string label, System.Diagnostics.Stopwatch sw)
    {
        Console.Error.WriteLine($"[QwenAsrMelBenchmark] {label}: {sw.ElapsedMilliseconds}ms");
        try
        {
            System.IO.File.AppendAllText(
                System.IO.Path.Combine(System.IO.Path.GetTempPath(), "stingray-qwenasr-mel-diag.log"),
                $"[QwenAsrMelBenchmark] {label}: {sw.ElapsedMilliseconds}ms{Environment.NewLine}");
        }
        catch { /* best-effort */ }
    }

    [Fact]
    public void ExtractMel_Benchmark_RealisticClipLength()
    {
        var extractor = new QwenAsrMelExtractor();

        // 10s of 16kHz audio -- realistic single-utterance scale.
        const int sampleRate = 16000;
        const int durationSec = 10;
        var pcm = new float[sampleRate * durationSec];
        var rng = new Random(42);
        for (int i = 0; i < pcm.Length; i++) pcm[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var mel = extractor.ExtractMel(pcm);
        sw.Stop();

        LogTiming($"ExtractMel {durationSec}s @ {sampleRate}Hz -> {mel.Length} mel values", sw);
        Assert.NotEmpty(mel);
    }
}
