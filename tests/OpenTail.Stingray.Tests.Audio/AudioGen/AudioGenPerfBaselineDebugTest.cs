using OpenTail.Stingray.Audio.AudioGen;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.Audio.AudioGen;

/// <summary>
/// TEMPORARY, throwaway perf timing bench for AudioGen-medium for PerformanceLeague.md backfill.
/// No numeric golden reference exists for this port yet (see AudioGenGenerationSmokeTests), so
/// this is a timing-only, non-degeneracy-checked measurement -- OT-only new coverage, no ratio.
/// Not part of the permanent suite, delete after use.
/// </summary>
public sealed class AudioGenPerfBaselineDebugTest : HeavyTestBase
{
    private const int Runs = 3;

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

    [Fact]
    public void Bench_Generate_3sAudio()
    {
        string? lmPath = FindRepoFile("models/audiogen-medium/audiogen-medium-lm.safetensors");
        string? codecPath = FindRepoFile("models/audiogen-medium/audiogen-medium-encodec16k.safetensors");
        string? t5Path = FindRepoFile("models/audiogen-medium/t5-large.safetensors");
        string? tokenizerPath = FindRepoFile("models/audiogen-medium/t5-large-tokenizer.json");
        Assert.SkipUnless(lmPath != null && codecPath != null && t5Path != null && tokenizerPath != null,
            "models/audiogen-medium weights not found");

        using var lmLoader = SafetensorsLoader.Open(lmPath!);
        using var codecLoader = SafetensorsLoader.Open(codecPath!);
        using var t5Loader = SafetensorsLoader.Open(t5Path!);

        var textEncoderWeights = AudioGenTextEncoderWeights.Load(t5Loader);
        var tokenizer = T5Tokenizer.FromFile(tokenizerPath!);
        var transformerWeights = new AudioGenTransformerWeights(lmLoader);
        var codecWeights = AudioGenEncodecDecoderWeights.Load(codecLoader);
        var generator = new AudioGenGenerator(textEncoderWeights, tokenizer, transformerWeights, codecWeights);

        const string prompt = "dog barking";
        const float durationSeconds = 3.0f;

        var warm = generator.Generate(prompt, durationSeconds: durationSeconds, seed: 0, guidanceScale: 3.0f, topK: 1);
        Assert.NotEmpty(warm);

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        for (int i = 0; i < Runs; i++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var pcm = generator.Generate(prompt, durationSeconds: durationSeconds, seed: 0, guidanceScale: 3.0f, topK: 1);
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = pcm.Length;
        }

        const int sampleRate = 16000; // EnCodec 16kHz variant used by AudioGen
        double audioSec = sampleCount / (double)sampleRate;
        double meanSec = elapsedSec.Average();
        double rtf = meanSec / audioSec;
        string msg = $"[AudioGen-medium] prompt=\"{prompt}\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[AudioGen-medium] runs(s)=[{string.Join(", ", elapsedSec.Select(x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime; non-degeneracy checked only, no numeric golden reference exists for this port yet)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }
}
