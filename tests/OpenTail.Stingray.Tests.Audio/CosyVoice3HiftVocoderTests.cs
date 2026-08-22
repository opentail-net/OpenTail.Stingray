using System.IO;
using OpenTail.Stingray.Audio.CosyVoice;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weights sanity coverage for CosyVoice3's HiFT vocoder (<see cref="CosyVoice3HiftWeights"/>,
/// real GGUF weights, no weight-norm fold needed -- see the class doc comment). Mirrors
/// <see cref="CosyVoiceHiftVocoderTests"/>'s CosyVoice2 structural pass exactly (same
/// <see cref="CosyVoiceHiftVocoder"/> forward code, generic over <c>IHiFTVocoderWeights</c>).
/// </summary>
public sealed class CosyVoice3HiftVocoderTests : HeavyTestBase
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
        string? path = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(path != null, "models/cosyvoice3/CosyVoice3-2512_F16.gguf not found");

        using var w = new CosyVoice3HiftWeights(path!);

        int t = 16; // small mel frame count for a fast structural pass
        var mel = new float[80 * t];
        for (int i = 0; i < mel.Length; i++) mel[i] = 0.1f * MathF.Sin(i * 0.07f);

        var wav = CosyVoiceHiftVocoder.Generate(w, mel, t, new Random(42));

        int totalUp = w.IstftHopLen;
        foreach (var r in w.UpsampleRates) totalUp *= r;
        Assert.True(wav.Length > 0);
        Assert.True(wav.Length >= t * totalUp - w.IstftNFft);

        foreach (var v in wav)
        {
            Assert.False(float.IsNaN(v) || float.IsInfinity(v));
            Assert.InRange(v, -1.0f, 1.0f);
        }
    }
}
