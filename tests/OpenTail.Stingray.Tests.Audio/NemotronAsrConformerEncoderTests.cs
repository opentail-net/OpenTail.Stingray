
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real structural validation for NemotronAsrConformerEncoder against the real downloaded
/// checkpoint and real audio -- confirms finite, non-degenerate output at the real joint_dim
/// (640), running the full mel -> subsampling -> 24-layer Conformer -> prompt-conditioning chain.
/// </summary>
public sealed class NemotronAsrConformerEncoderTests : HeavyTestBase
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
        string? modelPath = FindRepoFile("models/_models/nemotron-asr/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(audioPath != null, "reference a.wav not found");

        using var w = new OpenTail.Stingray.Audio.NemotronAsr.NemotronAsrWeights(modelPath!);
        var extractor = OpenTail.Stingray.Audio.NemotronAsr.NemotronAsrMelExtractor.FromWeights(w);

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);

        var (mel, frames, _) = extractor.ExtractMel(samples);
        var (subsampled, tEnc) = OpenTail.Stingray.Audio.NemotronAsr.NemotronAsrSubsampling.Forward(w, mel, frames);
        var encoded = OpenTail.Stingray.Audio.NemotronAsr.NemotronAsrConformerEncoder.Forward(w, subsampled);

        Assert.Equal(tEnc, encoded.Length);
        foreach (var row in encoded) Assert.Equal(w.JointDim, row.Length);

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
        Console.Error.WriteLine($"[NemotronAsrEncoder] tEnc={tEnc} jointDim={w.JointDim} mean={mean:F5} std={std:F5}");
        Assert.True(std > 1e-6, "encoder output looks collapsed/degenerate");
    }
}
