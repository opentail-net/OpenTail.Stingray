
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMPORARY, throwaway perf timing bench for XTTS-v2 for PerformanceLeague.md backfill. The
/// existing Baseline_Xtts test (TtsPerformanceBaselineDebugTest.cs) requires a specific,
/// gitignored reference wav (docs/audio-samples/fishspeech-lunch-REFERENCE.wav) that isn't present
/// on this machine; this variant uses examples/audio.cpp/assets/resources/b.wav instead (any real
/// voice reference works functionally for a timing measurement). Not part of the permanent suite,
/// delete after use.
/// </summary>
public sealed class XttsPerfBaselineDebugTest : HeavyTestBase
{
    private const string Prompt = "Hello, I will make some lunch, darling!";
    private const int Runs = 3;

    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Baseline_Xtts_BWavRef()
    {
        string? checkpointDir = FindRepoFile("models/xtts-v2/model.safetensors") is { } p ? Path.GetDirectoryName(p) : null;
        string? refWav = FindRepoFile("examples/audio.cpp/assets/resources/b.wav");
        Assert.SkipUnless(checkpointDir != null && refWav != null, "XTTS checkpoint or b.wav not found");

        var pipeline = OpenTail.Stingray.Audio.Xtts.XttsPipeline.Load(checkpointDir!);

        var warm = pipeline.Generate(Prompt, refWav!, "en", seed: 42);
        Assert.NotEmpty(warm);

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        for (int i = 0; i < Runs; i++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var wav = pipeline.Generate(Prompt, refWav!, "en", seed: 42);
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = wav.Length;
        }

        double audioSec = sampleCount / (double)pipeline.DefaultSampleRate;
        double meanSec = elapsedSec.Average();
        double rtf = meanSec / audioSec;
        string msg = $"[XTTS-v2] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[XTTS-v2] runs(s)=[{string.Join(", ", elapsedSec.Select(x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }
}
