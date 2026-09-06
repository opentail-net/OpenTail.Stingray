
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real load + forward validation for OmniVoiceAcousticEncoderWeights/
/// OmniVoiceAcousticEncoder against the real downloaded checkpoint and real audio (resampled to
/// the codec's real 24kHz sample rate).</summary>
public sealed class OmniVoiceAcousticEncoderTests : HeavyTestBase
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
    public void Load_RealCheckpoint_And_Encode_RealAudio_ProducesFiniteNonDegenerateLatent()
    {
        string? path = FindRepoFile("models/_models/omnivoice/audio_tokenizer/model.safetensors");
        Assert.SkipUnless(path != null, "omnivoice audio_tokenizer model.safetensors not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(audioPath != null, "reference a.wav not found");

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(path!);
        var w = new OpenTail.Stingray.Audio.OmniVoice.OmniVoiceAcousticEncoderWeights(loader);

        AssertFinite(w.Conv1Weight, "Conv1Weight");
        AssertFinite(w.Conv2Weight, "Conv2Weight");
        foreach (var b in w.Blocks)
        {
            AssertFinite(b.ConvWeight, "Block.ConvWeight");
            AssertFinite(b.Res1.Conv1Weight, "Block.Res1.Conv1Weight");
        }

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 24000) samples = AudioResampler.Resample(samples, sr, 24000);
        // Trim to a short real clip to keep this a structural (not perf) check.
        int clipLen = Math.Min(samples.Length, 24000 * 2);
        var clip = samples.AsSpan(0, clipLen).ToArray();

        var (latent, frames) = OpenTail.Stingray.Audio.OmniVoice.OmniVoiceAcousticEncoder.Encode(w, clip);

        Assert.Equal(256 * frames, latent.Length);
        double sum = 0, sumSq = 0;
        foreach (var v in latent)
        {
            Assert.True(float.IsFinite(v));
            sum += v;
            sumSq += (double)v * v;
        }
        double mean = sum / latent.Length;
        double std = Math.Sqrt(Math.Max(0, sumSq / latent.Length - mean * mean));
        Console.Error.WriteLine($"[OmniVoiceAcousticEncode] frames={frames} mean={mean:F5} std={std:F5}");
        Assert.True(std > 1e-6, "acoustic encoder output has collapsed to near-silence");
    }

    private static void AssertFinite(float[] values, string label)
    {
        Assert.True(values.Length > 0, $"{label} is empty");
        foreach (var v in values)
            Assert.True(float.IsFinite(v), $"{label} contains a non-finite value");
    }
}
