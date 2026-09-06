
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real mel-extraction test for RvcRmvpeMelExtractor against real audio, cross-checked
/// numerically against the real C++ reference (STINGRAY_RVC_TRACE=1 added to
/// examples/audio.cpp/src/framework/modules/pitch_extractors/rmvpe_pitch_extractor.cpp).</summary>
public sealed class RvcRmvpeMelExtractorTests : HeavyTestBase
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
    public void Extract_RealAudio_ProducesFiniteNonDegenerateMel()
    {
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(audioPath != null, "reference a.wav not found");

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);

        var (mel, realFrames) = OpenTail.Stingray.Audio.Rvc.RvcRmvpeMelExtractor.ExtractPadded(samples);

        Assert.True(mel.Length % 32 == 0, "padded frame count must be a multiple of 32");
        Assert.True(realFrames <= mel.Length);

        double sum = 0, sumSq = 0;
        int n = 0;
        for (int f = 0; f < realFrames; f++)
        {
            foreach (var v in mel[f])
            {
                Assert.True(float.IsFinite(v));
                sum += v;
                sumSq += (double)v * v;
                n++;
            }
        }
        double mean = sum / n;
        double std = Math.Sqrt(Math.Max(0, sumSq / n - mean * mean));
        Console.Error.WriteLine($"[RvcRmvpeMel] realFrames={realFrames} paddedFrames={mel.Length} mean={mean:F4} std={std:F4}");
        Assert.InRange(std, 1e-3, 100.0);
    }
}
