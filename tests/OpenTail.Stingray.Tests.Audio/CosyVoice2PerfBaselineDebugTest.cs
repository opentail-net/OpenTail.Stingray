
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMPORARY, throwaway perf timing bench for CosyVoice2 for PerformanceLeague.md backfill.
/// No C++ reference exists for CosyVoice2 (examples/cosyvoice.cpp doesn't implement the base
/// cosyvoice_model::llm_job path CosyVoice2 needs -- only cosyvoice_model_3's, per
/// CosyVoice2GenerateWavDebugTest), so this is OT-only new coverage, no ratio. Not part of the
/// permanent suite, delete after use.
/// </summary>
public sealed class CosyVoice2PerfBaselineDebugTest : HeavyTestBase
{
    private const string Prompt = "Hello, I will make some lunch, darling!";
    private const int Runs = 3;

    private static string? FindModelPath(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Baseline_CosyVoice2()
    {
        string? llmPath = FindModelPath("models/_models/cosyvoice2_llm.safetensors");
        string? tokDir = FindModelPath("models/cosyvoice2_tokenizer");
        string? flowPath = FindModelPath("models/_models/cosyvoice2_flow.safetensors");
        string? hiftPath = FindModelPath("models/_models/cosyvoice2_hift.safetensors");
        Assert.SkipUnless(llmPath != null && tokDir != null && flowPath != null && hiftPath != null,
            "CosyVoice2 model files not found");

        using var pipeline = OpenTail.Stingray.Audio.CosyVoice.CosyVoice2Pipeline.Load(llmPath!, tokDir!, flowPath!, hiftPath!);

        var warm = pipeline.Generate(Prompt, seed: 42);
        Assert.True(warm.Length > 0, "CosyVoice2 produced empty audio on warmup");

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        for (int i = 0; i < Runs; i++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var wav = pipeline.Generate(Prompt, seed: 42);
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = wav.Length;
        }

        double audioSec = sampleCount / (double)pipeline.SampleRate;
        double meanSec = elapsedSec.Average();
        double rtf = meanSec / audioSec;
        string msg = $"[CosyVoice2] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[CosyVoice2] runs(s)=[{string.Join(", ", elapsedSec.Select(x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindModelPath("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }
}
