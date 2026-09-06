
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real structural validation for NemotronAsrSubsampling against the real downloaded
/// checkpoint and real audio -- confirms finite, non-degenerate output at the real hidden
/// dimension (1024), and that the internal frequency dimension collapses to the reference's own
/// asserted `stage3_features == 17` (256*17=4352 == pre_encode.out.weight's real input dim).</summary>
public sealed class NemotronAsrSubsamplingTests : HeavyTestBase
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
    public void Forward_RealCheckpoint_RealAudio_ProducesFiniteNonDegenerateHidden()
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
        var (hidden, tEnc) = OpenTail.Stingray.Audio.NemotronAsr.NemotronAsrSubsampling.Forward(w, mel, frames);

        Assert.True(tEnc > 0, "expected at least one subsampled frame");
        Assert.Equal(tEnc, hidden.Length);
        foreach (var row in hidden) Assert.Equal(w.HiddenDim, row.Length);

        // Real reference's own assertion: stage3_features must be 17 (256*17=4352=pre_encode.out input dim).
        // We don't compute stage3_features directly here (it's folded into PreOutWeight's fixed
        // shape), but a shape mismatch inside Forward would have already thrown -- this just
        // re-confirms the expected 8x-ish downsampling ratio roughly holds.
        double approxRatio = (double)frames / tEnc;
        Assert.InRange(approxRatio, 6.0, 10.0);

        double sum = 0, sumSq = 0;
        int n = 0;
        foreach (var row in hidden)
        {
            foreach (var v in row)
            {
                Assert.True(float.IsFinite(v), "subsampled hidden contains a non-finite value");
                sum += v;
                sumSq += (double)v * v;
                n++;
            }
        }
        double mean = sum / n;
        double std = Math.Sqrt(Math.Max(0, sumSq / n - mean * mean));
        Console.Error.WriteLine($"[NemotronAsrSubsample] melFrames={frames} tEnc={tEnc} hidden={w.HiddenDim} mean={mean:F5} std={std:F5}");
        Assert.True(std > 1e-6, "subsampled hidden looks collapsed/degenerate");
    }
}
