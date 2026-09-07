using OpenTail.Stingray.Audio.HiggsAudio;
using OpenTail.Stingray.Audio.OmniVoice;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, live end-to-end smoke test for OmniVoice's OWN full real reference-audio encode chain
/// (`codec_encode`-equivalent): real acoustic encoder + real hidden-state-averaged semantic
/// encoder (real stride=2, confirmed via this checkpoint's own real config, not assumed) + real
/// semantic post-network (reused from <see cref="HiggsSemanticPostEncoder"/>, confirmed to
/// degenerate to the same real structure for this checkpoint's real config) + real `fc` mixing
/// linear + real RVQ quantize, chained together for the first time in
/// <see cref="OmniVoiceCodecEncoder"/>.
/// </summary>
public sealed class OmniVoiceCodecEncoderFullChainRealWeightsTests : HeavyTestBase
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
    public void Encode_OnRealCheckpoint_ProducesInRangeCodesForEveryCodebook()
    {
        string? path = FindRepoFile("models/_models/omnivoice/audio_tokenizer/model.safetensors");
        Assert.SkipUnless(path != null, "omnivoice audio_tokenizer model.safetensors not found");

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(path!);

        var acousticWeights = new OmniVoiceAcousticEncoderWeights(loader);
        var semanticWeights = new OmniVoiceSemanticWeights(loader);
        var semanticPostWeights = HiggsSemanticPostEncoder.Weights.Load(loader.ReadF32);
        var codecWeights = new OmniVoiceAcousticDecoderWeights(loader);

        int totalAcousticStride = OmniVoiceAcousticEncoderWeights.DownsamplingRatios.Aggregate(1, (a, b) => a * b);
        int acousticFrameCount = 6;
        int samples24k = totalAcousticStride * acousticFrameCount;
        int samples16k = samples24k; // real headroom, not a claim of exact 24k:16k alignment -- see HiggsCodecEncoderFullChainRealWeightsTests's identical note

        var rng = new Random(37);
        var waveform24k = new float[samples24k];
        for (int i = 0; i < samples24k; i++) waveform24k[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        var waveform16k = new float[samples16k];
        for (int i = 0; i < samples16k; i++) waveform16k[i] = (float)(rng.NextDouble() * 0.2 - 0.1);

        var codes = OmniVoiceCodecEncoder.Encode(acousticWeights, semanticWeights, semanticPostWeights, codecWeights, waveform24k, waveform16k);

        Assert.Equal(OmniVoiceAcousticDecoderWeights.NumCodebooks, codes.Length);
        foreach (var codebook in codes)
        {
            Assert.Equal(acousticFrameCount, codebook.Length);
            Assert.All(codebook, c => Assert.InRange(c, 0, OmniVoiceAcousticDecoderWeights.CodebookSize - 1));
        }
    }
}
