using System.IO;
using OpenTail.Stingray.Audio.CosyVoice;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weights sanity coverage for CosyVoice2's HiFT vocoder forward DSP pass (see
/// docs/audio-review-progress.md's CosyVoice section). NOT yet golden-verified against a real
/// oracle -- confirms real weights (including the weight-norm fold) drive a real forward pass
/// end-to-end without NaN/Inf, the same bar every other pipeline's first real-weights test
/// passed before golden verification followed.
/// </summary>
public sealed class CosyVoiceHiftVocoderTests : HeavyTestBase
{
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
    public void Generate_RealWeights_ProducesFiniteBoundedWaveform()
    {
        string? path = FindRepoFile("models/cosyvoice2_hift.safetensors");
        Assert.SkipUnless(path != null, "models/cosyvoice2_hift.safetensors not found");

        using var w = new CosyVoiceHiftWeights(path!);

        int t = 16; // small mel frame count for a fast structural pass
        var mel = new float[80 * t];
        for (int i = 0; i < mel.Length; i++) mel[i] = 0.1f * MathF.Sin(i * 0.07f);

        var wav = CosyVoiceHiftVocoder.Generate(w, mel, t, new Random(42));

        int totalUp = w.IstftHopLen;
        foreach (var r in w.UpsampleRates) totalUp *= r;
        Assert.True(wav.Length > 0);
        Assert.True(wav.Length >= t * totalUp - w.IstftNFft); // roughly the expected upsampled sample count

        foreach (var v in wav)
        {
            Assert.False(float.IsNaN(v) || float.IsInfinity(v));
            Assert.InRange(v, -1.0f, 1.0f);
        }
    }
}
