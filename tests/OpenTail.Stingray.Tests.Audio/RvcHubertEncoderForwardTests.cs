
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real forward-pass sanity test for RvcHubertEncoder against real weights and real
/// 16kHz audio -- checks finite, non-degenerate output, not yet a numeric match against the real
/// C++ reference's own dumped hidden states (a follow-up step).</summary>
public sealed class RvcHubertEncoderForwardTests : HeavyTestBase
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
    public void Forward_RealWeights_RealAudio_ProducesFiniteNonDegenerateOutput()
    {
        string? checkpointPath = FindRepoFile("examples/audio.cpp/models/RVC-GGUF/rvc-f16.gguf");
        Assert.SkipUnless(checkpointPath != null, "rvc-f16.gguf not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(audioPath != null, "reference a.wav not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(checkpointPath!);
        var source = new OpenTail.Stingray.Audio.Rvc.RvcPackedTensorSource(model);
        var w = new OpenTail.Stingray.Audio.Rvc.RvcHubertWeights(source);

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);

        // Trim to a length producing an even token count (conv stride product 320) so the
        // not-yet-implemented odd-token attention-mask path isn't exercised by this test.
        int usableSamples = (samples.Length / 640) * 640; // 640 = 320 * 2
        var clip = samples.AsSpan(0, Math.Min(usableSamples, samples.Length)).ToArray();
        Assert.True(clip.Length > 640, "clip too short after trimming");

        var hidden = OpenTail.Stingray.Audio.Rvc.RvcHubertEncoder.Forward(w, clip);

        Assert.NotEmpty(hidden);
        Assert.Equal(OpenTail.Stingray.Audio.Rvc.RvcHubertWeights.HiddenDim, hidden[0].Length);

        double sum = 0, sumSq = 0;
        int n = 0;
        foreach (var row in hidden)
        {
            foreach (var v in row)
            {
                Assert.True(float.IsFinite(v), "non-finite value in hidden state");
                sum += v;
                sumSq += (double)v * v;
                n++;
            }
        }
        double mean = sum / n;
        double std = Math.Sqrt(Math.Max(0, sumSq / n - mean * mean));
        Console.Error.WriteLine($"[RvcHubertFwd] frames={hidden.Length} mean={mean:F4} std={std:F4}");
        Assert.InRange(std, 1e-3, 100.0);
    }
}
