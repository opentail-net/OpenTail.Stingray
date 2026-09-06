
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real structural validation for FunAsrNanoAdaptor against the real downloaded
/// checkpoint and real audio -- runs the full mel -> SAN-M encoder -> adaptor chain and confirms
/// finite, non-degenerate 1024-dim (Qwen3 hidden size) output ready for LLM splicing.</summary>
public sealed class FunAsrNanoAdaptorRealWeightsTests : HeavyTestBase
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
        var encoderWeights = new OpenTail.Stingray.Audio.FunASR.FunAsrNanoEncoderWeights(source);
        var adaptorWeights = new OpenTail.Stingray.Audio.FunASR.FunAsrNanoAdaptorWeights(source);

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);

        var melExtractor = new OpenTail.Stingray.Audio.FunASR.FunAsrRealMelExtractor();
        var logMel = melExtractor.ExtractLogMel(samples, waveformScale: 1f);
        var lfr = OpenTail.Stingray.Audio.FunASR.FunAsrNanoEncoder.ApplyRealLfr(logMel, 7, 6);
        int frames = Math.Min(lfr.Length, 150);
        var trimmed = new float[frames][];
        Array.Copy(lfr, trimmed, frames);

        var encoded = OpenTail.Stingray.Audio.FunASR.FunAsrNanoEncoder.Forward(encoderWeights, trimmed);
        var adapted = OpenTail.Stingray.Audio.FunASR.FunAsrNanoAdaptor.Forward(adaptorWeights, encoded);

        Assert.Equal(frames, adapted.Length);
        foreach (var row in adapted) Assert.Equal(OpenTail.Stingray.Audio.FunASR.FunAsrNanoAdaptorWeights.DModel, row.Length);

        double sum = 0, sumSq = 0;
        int n = 0;
        foreach (var row in adapted)
        {
            foreach (var v in row)
            {
                Assert.True(float.IsFinite(v), "adaptor output contains a non-finite value");
                sum += v;
                sumSq += (double)v * v;
                n++;
            }
        }
        double mean = sum / n;
        double std = Math.Sqrt(Math.Max(0, sumSq / n - mean * mean));
        Console.Error.WriteLine($"[FunAsrNanoAdaptor] frames={frames} dModel={OpenTail.Stingray.Audio.FunASR.FunAsrNanoAdaptorWeights.DModel} mean={mean:F5} std={std:F5}");
        Assert.True(std > 1e-6, "adaptor output looks collapsed/degenerate");
    }
}
