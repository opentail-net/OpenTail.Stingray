using OpenTail.Stingray.Audio.HiggsAudio;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real-weight smoke test for <see cref="HiggsCodecDecoder"/>.</summary>
public sealed class HiggsCodecDecoderRealWeightsTests : HeavyTestBase
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
        string? path = FindRepoFile("models/_models/higgs_audio_tts/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf");
        Assert.SkipUnless(path != null, "higgs-audio-v3-tts-4b-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        var w = HiggsCodecDecoderWeights.Load(source.GetTensor);

        var rng = new Random(17);
        const int frames = 5;
        var codes = new int[frames][];
        for (int t = 0; t < frames; t++)
        {
            codes[t] = new int[HiggsCodecDecoderWeights.NumCodebooks];
            for (int cb = 0; cb < HiggsCodecDecoderWeights.NumCodebooks; cb++)
                codes[t][cb] = rng.Next(HiggsCodecDecoderWeights.CodebookSize);
        }

        var waveform = HiggsCodecDecoder.Decode(w, codes);

        int expectedUpsample = 1;
        foreach (int r in HiggsCodecDecoderWeights.UpsampleRatios) expectedUpsample *= r;
        Assert.Equal(960, expectedUpsample); // real kCodecHopLength

        Assert.True(waveform.Length > frames * expectedUpsample / 2, "waveform should be roughly frames*hop samples long");
        Assert.All(waveform, v => Assert.True(float.IsFinite(v)));
        Assert.Contains(waveform, v => v != 0f);
    }
}
