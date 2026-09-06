
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real load + decode validation for OmniVoiceAcousticDecoderWeights/
/// OmniVoiceAcousticDecoder against the real downloaded checkpoint, with synthetic (random) codes
/// -- not yet golden-verified against a real reference decode (no CLI/warm-bench task wired up
/// for OmniVoice yet), same first-step-only status the other new OmniVoice components have.</summary>
public sealed class OmniVoiceAcousticDecoderTests : HeavyTestBase
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
    public void Load_RealCheckpoint_And_Decode_SyntheticCodes_ProducesFiniteNonDegenerateAudio()
    {
        string? path = FindRepoFile("models/_models/omnivoice/audio_tokenizer/model.safetensors");
        Assert.SkipUnless(path != null, "omnivoice audio_tokenizer model.safetensors not found");

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(path!);
        var w = new OpenTail.Stingray.Audio.OmniVoice.OmniVoiceAcousticDecoderWeights(loader);

        AssertFinite(w.Conv1Weight, "Conv1Weight");
        AssertFinite(w.Conv2Weight, "Conv2Weight");
        AssertFinite(w.FinalSnakeAlpha, "FinalSnakeAlpha");
        foreach (var b in w.Blocks)
        {
            AssertFinite(b.ConvTWeight, "Block.ConvTWeight");
            AssertFinite(b.Res1.Conv1Weight, "Block.Res1.Conv1Weight");
        }
        foreach (var q in w.Quantizers)
        {
            AssertFinite(q.CodebookEmbed, "Quantizer.CodebookEmbed");
            AssertFinite(q.ProjectOutWeight, "Quantizer.ProjectOutWeight");
        }

        const int frames = 10;
        var rng = new Random(3);
        var codes = new int[frames][];
        for (int t = 0; t < frames; t++)
        {
            codes[t] = new int[OpenTail.Stingray.Audio.OmniVoice.OmniVoiceAcousticDecoderWeights.NumCodebooks];
            for (int q = 0; q < codes[t].Length; q++)
                codes[t][q] = rng.Next(OpenTail.Stingray.Audio.OmniVoice.OmniVoiceAcousticDecoderWeights.CodebookSize);
        }

        var audio = OpenTail.Stingray.Audio.OmniVoice.OmniVoiceAcousticDecoder.Decode(w, codes);

        // 5 upsample stages with rates [8,5,4,2,3] -> total 960x upsample.
        int expectedSamples = frames;
        foreach (var r in OpenTail.Stingray.Audio.OmniVoice.OmniVoiceAcousticDecoderWeights.UpsamplingRatios) expectedSamples *= r;
        Assert.Equal(expectedSamples, audio.Length);

        double sum = 0, sumSq = 0;
        foreach (var v in audio)
        {
            Assert.True(float.IsFinite(v));
            sum += v;
            sumSq += (double)v * v;
        }
        double mean = sum / audio.Length;
        double std = Math.Sqrt(Math.Max(0, sumSq / audio.Length - mean * mean));
        Console.Error.WriteLine($"[OmniVoiceAcousticDecode] samples={audio.Length} mean={mean:F5} std={std:F5}");
        Assert.True(std > 1e-6, "decoder output has collapsed to near-silence");
    }

    private static void AssertFinite(float[] values, string label)
    {
        Assert.True(values.Length > 0, $"{label} is empty");
        foreach (var v in values)
            Assert.True(float.IsFinite(v), $"{label} contains a non-finite value");
    }
}
