using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.StableAudio;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// TEMPORARY, throwaway perf timing bench for Stable Audio 3 Small SFX for
/// PerformanceLeague.md backfill. Reuses the same StableAudioPipeline/DiT/VAE code as Small
/// Music (byte-identical config, see StableAudio3SmallSfxTests' doc comment), pointed at the
/// real SFX checkpoint. No C++ reference exists for this pipeline. Not part of the permanent
/// suite, delete after use.
/// </summary>
public sealed class StableAudio3SmallSfxPerfBaselineDebugTest
{
    private const string DitDirRelative = "models/stable-audio-3-small-sfx-base";
    private const string T5GemmaDirRelative = "models/stable-audio-3-t5gemma";
    private const int Runs = 3;

    private static string? FindRepoDir(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath.Replace('/', Path.DirectorySeparatorChar));
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Bench_Generate_3sAudio_8Steps()
    {
        string? ditDir = FindRepoDir(DitDirRelative);
        string? t5gemmaDir = FindRepoDir(T5GemmaDirRelative);
        Assert.SkipUnless(ditDir != null, "models/stable-audio-3-small-sfx-base not found");
        Assert.SkipUnless(t5gemmaDir != null, "models/stable-audio-3-t5gemma not found");

        using var ditWeights = SafetensorsLoader.OpenDirectory(ditDir!);
        using var textEncoderWeights = SafetensorsLoader.OpenDirectory(t5gemmaDir!);
        using var pipeline = new StableAudioPipeline(ditWeights, textEncoderWeights, t5gemmaDir!);

        const string prompt = "a glass bottle shattering on a hard floor";
        const float durationSeconds = 3f;
        const int steps = 8;

        var warm = pipeline.Generate(new StableAudioRequest
        {
            Prompt = prompt,
            DurationSeconds = durationSeconds,
            Steps = steps,
            CfgScale = 6.0f,
            Seed = 1234,
            OutputPath = "",
        });
        Assert.True(warm.Length > 0, "generated zero samples on warmup");

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        for (int i = 0; i < Runs; i++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var pcm = pipeline.Generate(new StableAudioRequest
            {
                Prompt = prompt,
                DurationSeconds = durationSeconds,
                Steps = steps,
                CfgScale = 6.0f,
                Seed = 1234,
                OutputPath = "",
            });
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = pcm.Length;
        }

        const int sampleRate = 44100;
        double audioSec = sampleCount / (double)sampleRate;
        double meanSec = elapsedSec.Average();
        double rtf = meanSec / audioSec;
        string msg = $"[StableAudio3-SmallSfx] prompt=\"{prompt}\" steps={steps} audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[StableAudio3-SmallSfx] runs(s)=[{string.Join(", ", elapsedSec.Select(x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime; {steps}-step CFG, not the full recommended step count)";
        Console.Error.WriteLine(msg);
        string? docsDir = FindRepoDir("docs");
        if (docsDir != null)
            File.AppendAllText(Path.Combine(docsDir, "tts-benchmark-log.txt"), msg + "\n\n");
    }
}
