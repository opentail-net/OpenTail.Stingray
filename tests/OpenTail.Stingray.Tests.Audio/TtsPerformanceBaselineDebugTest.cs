
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: measures wall-clock generation time for QwenTTS and CosyVoice3
/// on the same prompt, to establish a performance baseline before any optimization work. Includes
/// a warmup run (excluded from timing, since first-call JIT/model-load effects would otherwise
/// dominate) followed by several timed runs so a single slow/fast outlier doesn't skew the result.</summary>
public sealed class TtsPerformanceBaselineDebugTest : HeavyTestBase
{
    private const string Prompt = "Hello, I will make some lunch, darling!";
    private const int Runs = 3;

    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Baseline_QwenTts()
    {
        string? talkerPath = FindRepoFile("models/qwen-talker-0.6b-base-Q8_0.gguf");
        Assert.SkipUnless(talkerPath != null, "QwenTTS talker model not found");

        using var pipeline = QwenTtsPipeline.Load(talkerPath!);

        // Warmup (not timed): first call pays for JIT/tensor-cache warmup.
        var warm = pipeline.Generate(Prompt, seed: 42);
        Assert.NotEmpty(warm);

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        float[]? lastWav = null;
        for (int i = 0; i < Runs; i++)
        {
            var sw = Stopwatch.StartNew();
            var wav = pipeline.Generate(Prompt, seed: 42);
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = wav.Length;
            lastWav = wav;
        }

        if (lastWav != null)
        {
            string? outDir = FindRepoFile("docs/audio-samples");
            if (outDir != null)
            {
                string wavPath = Path.Combine(outDir, "qwen-tts-perf-turn2.wav");
                new AudioGenerationResult(lastWav, 24000).SaveWav(wavPath);
            }
        }

        double audioSec = sampleCount / 24000.0;
        double meanSec = Average(elapsedSec);
        double rtf = meanSec / audioSec;
        string msg = $"[QwenTTS] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[QwenTTS] runs(s)=[{string.Join(", ", Array.ConvertAll(elapsedSec, x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public void Baseline_CosyVoice3()
    {
        string? modelPath = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(modelPath != null, "CosyVoice3 GGUF model not found");

        using var pipeline = CosyVoice3Pipeline.Load(modelPath!);

        var warm = pipeline.Generate(Prompt, seed: 42);
        Assert.NotEmpty(warm);

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        float[]? lastWav = null;
        for (int i = 0; i < Runs; i++)
        {
            var sw = Stopwatch.StartNew();
            var wav = pipeline.Generate(Prompt, seed: 42);
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = wav.Length;
            lastWav = wav;
        }

        if (lastWav != null)
        {
            string? outDir = FindRepoFile("docs/audio-samples");
            if (outDir != null)
            {
                string wavPath = Path.Combine(outDir, "cosyvoice3-perf-turn1.wav");
                new AudioGenerationResult(lastWav, 24000).SaveWav(wavPath);
            }
        }

        double audioSec = sampleCount / 24000.0;
        double meanSec = Average(elapsedSec);
        double rtf = meanSec / audioSec;
        string msg = $"[CosyVoice3] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[CosyVoice3] runs(s)=[{string.Join(", ", Array.ConvertAll(elapsedSec, x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    private static double Average(double[] values)
    {
        double sum = 0;
        foreach (var v in values) sum += v;
        return sum / values.Length;
    }
}
