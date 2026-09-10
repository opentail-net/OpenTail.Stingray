
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMPORARY, throwaway perf timing bench for Whisper Tiny (39M, HF safetensors format via
/// WhisperPipeline.LoadFromSafetensors) for PerformanceLeague.md backfill -- the only Whisper size
/// this doc's ASR table was missing (Base/Small/Medium/Large-v3 already covered). Not part of the
/// permanent suite, delete after use.
/// </summary>
public sealed class WhisperTinyPerfBaselineDebugTest : HeavyTestBase
{
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
    public void Bench_Transcribe_BWav()
    {
        string? checkpointDir = FindRepoFile("models/whisper-tiny-hf");
        Assert.SkipUnless(checkpointDir != null, "models/whisper-tiny-hf not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/b.wav");
        Assert.SkipUnless(audioPath != null, "reference b.wav not found");

        using var pipeline = OpenTail.Stingray.Audio.Whisper.WhisperPipeline.LoadFromSafetensors(checkpointDir!);

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);
        double audioSec = samples.Length / 16000.0;

        var request = new SpeechToTextRequest
        {
            AudioSamples = samples,
            SampleRate = 16000,
            Language = "en",
            Task = SpeechTask.Transcribe
        };

        var warm = pipeline.Transcribe(request);

        double[] elapsedSec = new double[Runs];
        string lastText = "";
        for (int i = 0; i < Runs; i++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = pipeline.Transcribe(request);
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            lastText = result.Text;
        }

        double meanSec = elapsedSec.Average();
        double rtf = meanSec / audioSec;
        string msg = $"[Whisper-Tiny] audio={audioSec:F2}s text=\"{lastText}\"\n" +
                     $"[Whisper-Tiny] runs(s)=[{string.Join(", ", elapsedSec.Select(x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }
}
