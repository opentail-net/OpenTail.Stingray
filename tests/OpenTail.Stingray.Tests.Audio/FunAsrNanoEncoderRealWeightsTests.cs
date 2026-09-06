
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real structural validation for the full Fun-ASR-Nano-2512 SAN-M encoder stack (stem +
/// 49 main layers + norm + 20 timestamp-prediction layers + norm) against the real downloaded
/// checkpoint (`models/paraformer-q8.gguf`, despite the filename this IS the current
/// Fun-ASR-Nano-2512 architecture -- see docs/audio-review-progress.md) and real audio, reusing
/// the already-correct <see cref="FunAsrRealMelExtractor"/> frontend.</summary>
public sealed class FunAsrNanoEncoderRealWeightsTests : HeavyTestBase
{
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
    public void Forward_RealCheckpoint_RealAudio_ProducesFiniteNonDegenerateOutput()
    {
        string? modelPath = FindRepoFile("models/paraformer-q8.gguf");
        Assert.SkipUnless(modelPath != null, "models/paraformer-q8.gguf not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(audioPath != null, "reference a.wav not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(modelPath!);
        var source = new OpenTail.Stingray.Audio.Rvc.RvcPackedTensorSource(model);
        var w = new OpenTail.Stingray.Audio.FunASR.FunAsrNanoEncoderWeights(source);

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);

        var melExtractor = new OpenTail.Stingray.Audio.FunASR.FunAsrRealMelExtractor();
        var logMel = melExtractor.ExtractLogMel(samples, waveformScale: 1f);
        var lfr = OpenTail.Stingray.Audio.FunASR.FunAsrNanoEncoder.ApplyRealLfr(logMel, 7, 6);
        // Trim to a short real clip's worth of LFR frames to keep this a reasonable-time structural check.
        int frames = Math.Min(lfr.Length, 150);
        var trimmed = new float[frames][];
        Array.Copy(lfr, trimmed, frames);

        var encoded = OpenTail.Stingray.Audio.FunASR.FunAsrNanoEncoder.Forward(w, trimmed);

        Assert.Equal(frames, encoded.Length);
        foreach (var row in encoded) Assert.Equal(OpenTail.Stingray.Audio.FunASR.FunAsrNanoEncoderWeights.DModel, row.Length);

        double sum = 0, sumSq = 0;
        int n = 0;
        foreach (var row in encoded)
        {
            foreach (var v in row)
            {
                Assert.True(float.IsFinite(v), "encoder output contains a non-finite value");
                sum += v;
                sumSq += (double)v * v;
                n++;
            }
        }
        double mean = sum / n;
        double std = Math.Sqrt(Math.Max(0, sumSq / n - mean * mean));
        Console.Error.WriteLine($"[FunAsrNanoEncoder] frames={frames} dModel={OpenTail.Stingray.Audio.FunASR.FunAsrNanoEncoderWeights.DModel} mean={mean:F5} std={std:F5}");
        Assert.True(std > 1e-6, "encoder output looks collapsed/degenerate");
    }
}
