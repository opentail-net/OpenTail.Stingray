using OpenTail.Stingray.Audio.PersonaPlex;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real-weight smoke test for <see cref="MimiCodecDecoder"/>.</summary>
public sealed class MimiCodecDecoderRealWeightsTests : HeavyTestBase
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
    public void Decode_OnRealCheckpoint_ProducesFiniteWaveform()
    {
        string? path = FindRepoFile("models/_models/personaplex/PersonaPlex-GGUF/personaplex-7b-v1-q8_0.gguf");
        Assert.SkipUnless(path != null, "personaplex-7b-v1-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        var w = MimiCodecDecoderWeights.Load(source.GetTensor);

        var rng = new Random(51);
        const int frames = 4;
        var codes = new int[frames][];
        for (int t = 0; t < frames; t++)
        {
            codes[t] = new int[MimiCodecDecoderWeights.ActiveCodebooks];
            for (int cb = 0; cb < MimiCodecDecoderWeights.ActiveCodebooks; cb++)
                codes[t][cb] = rng.Next(MimiCodecDecoderWeights.CodebookSize);
        }

        var waveform = MimiCodecDecoder.Decode(w, codes);

        Assert.True(waveform.Length > 0);
        Assert.All(waveform, v => Assert.True(float.IsFinite(v)));
        Assert.All(waveform, v => Assert.InRange(v, -1f, 1f));
        Assert.Contains(waveform, v => v != 0f);
    }
}
