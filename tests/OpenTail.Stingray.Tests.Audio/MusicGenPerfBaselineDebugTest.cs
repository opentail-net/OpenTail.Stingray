using OpenTail.Stingray.Audio.MusicGen;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMPORARY, throwaway perf timing bench for MusicGen-small for PerformanceLeague.md backfill.
/// No numeric golden reference exists for this port yet (see MusicGenGenerationSmokeTests), so
/// this is a timing-only, non-degeneracy-checked measurement -- OT-only new coverage, no ratio.
/// Not part of the permanent suite, delete after use.
/// </summary>
public sealed class MusicGenPerfBaselineDebugTest : HeavyTestBase
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
        string? musicGenPath = FindRepoFile("models/musicgen-small/musicgen-small.safetensors");
        string? tokenizerPath = FindRepoFile("models/musicgen-small/t5-base-tokenizer.json");
        Assert.SkipUnless(musicGenPath != null && tokenizerPath != null,
            "models/musicgen-small/{musicgen-small.safetensors,t5-base-tokenizer.json} not found");

        using var musicGenLoader = SafetensorsLoader.Open(musicGenPath!);
        var textEncoderWeights = MusicGenTextEncoderWeights.Load(musicGenLoader);
        var tokenizer = T5Tokenizer.FromFile(tokenizerPath!);
        var transformerWeights = new MusicGenTransformerWeights(musicGenLoader);
        var codecWeights = MusicGenEncodecDecoderWeights.Load(musicGenLoader);
        var generator = new MusicGenGenerator(textEncoderWeights, tokenizer, transformerWeights, codecWeights);

        const string prompt = "acoustic guitar melody";
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

        const int sampleRate = 32000; // EnCodec 32kHz, MusicGen's native rate
        double audioSec = sampleCount / (double)sampleRate;
        double meanSec = elapsedSec.Average();
        double rtf = meanSec / audioSec;
        string msg = $"[MusicGen-small] prompt=\"{prompt}\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[MusicGen-small] runs(s)=[{string.Join(", ", elapsedSec.Select(x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime; non-degeneracy checked only, no numeric golden reference exists for this port yet)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }
}
