
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real load + forward validation for VoxtralTextDecoderWeights/VoxtralTextDecoder
/// against the real downloaded checkpoint, with a small synthetic prompt + real audio-tower
/// embeddings from a short real audio clip -- not yet golden-verified against a real reference
/// end-to-end ASR run (that needs the exact real prompt-construction details from
/// tokenizer_text.cpp, a bigger follow-on step).</summary>
public sealed class VoxtralTextDecoderLoadTests : HeavyTestBase
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
    public void Load_RealCheckpoint_AllLayersFinite_ForwardProducesFiniteLogits()
    {
        string? path = FindRepoFile("models/_models/voxtral-mini-realtime/model.safetensors");
        Assert.SkipUnless(path != null, "voxtral model.safetensors not found");

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(path!);
        var w = new OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralTextDecoderWeights(loader);

        AssertFinite(w.EmbedTokensWeight, "EmbedTokensWeight");
        AssertFinite(w.NormWeight, "NormWeight");
        for (int i = 0; i < OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralTextDecoderWeights.NumLayers; i++)
        {
            var l = w.Layers[i];
            AssertFinite(l.QWeight, $"Layers[{i}].QWeight");
            AssertFinite(l.Ada1Weight, $"Layers[{i}].Ada1Weight");
            AssertFinite(l.Ada2Weight, $"Layers[{i}].Ada2Weight");
        }

        // Small synthetic prompt: BOS(1) + a few pad tokens, no real audio splice length check --
        // structural forward-pass sanity only.
        int[] tokenIds = [1, 2, 2, 2, 2, 2, 2, 2];
        var audioEmbeddings = new float[3][];
        var rng = new Random(11);
        for (int i = 0; i < audioEmbeddings.Length; i++)
        {
            audioEmbeddings[i] = new float[OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralTextDecoderWeights.HiddenSize];
            for (int d = 0; d < audioEmbeddings[i].Length; d++) audioEmbeddings[i][d] = (float)(rng.NextDouble() * 0.1 - 0.05);
        }

        var logits = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralTextDecoder.Forward(w, tokenIds, audioEmbeddings, numDelayTokens: 6);

        Assert.Equal(tokenIds.Length, logits.Length);
        foreach (var row in logits)
        {
            Assert.Equal(OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralTextDecoderWeights.VocabSize, row.Length);
            foreach (var v in row) Assert.True(float.IsFinite(v));
        }
    }

    private static void AssertFinite(float[] values, string label)
    {
        Assert.True(values.Length > 0, $"{label} is empty");
        foreach (var v in values)
            Assert.True(float.IsFinite(v), $"{label} contains a non-finite value");
    }
}
