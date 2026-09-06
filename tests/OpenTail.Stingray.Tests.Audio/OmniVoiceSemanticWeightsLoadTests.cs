
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real load validation for OmniVoiceSemanticWeights against the real downloaded
/// k2-fsa/OmniVoice audio_tokenizer checkpoint.</summary>
public sealed class OmniVoiceSemanticWeightsLoadTests : HeavyTestBase
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
    public void Load_RealCheckpoint_AllTensorsFiniteWithExpectedShapes()
    {
        string? path = FindRepoFile("models/_models/omnivoice/audio_tokenizer/model.safetensors");
        Assert.SkipUnless(path != null, "omnivoice audio_tokenizer model.safetensors not found");

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(path!);
        var w = new OpenTail.Stingray.Audio.OmniVoice.OmniVoiceSemanticWeights(loader);

        for (int i = 0; i < 7; i++) AssertFinite(w.ConvWeights[i], $"ConvWeights[{i}]");
        AssertFinite(w.ConvLayer0GroupNormWeight, "ConvLayer0GroupNormWeight");
        AssertFinite(w.FeatureProjWeight, "FeatureProjWeight");
        AssertFinite(w.PosConvWeight, "PosConvWeight");
        Assert.Equal(768 * 48 * 128, w.PosConvWeight.Length);
        AssertFinite(w.EncoderLayerNormWeight, "EncoderLayerNormWeight");

        Assert.Equal(OpenTail.Stingray.Audio.OmniVoice.OmniVoiceSemanticWeights.NumLayers, w.Layers.Length);
        for (int i = 0; i < w.Layers.Length; i++)
        {
            var l = w.Layers[i];
            AssertFinite(l.AttnQWeight, $"Layers[{i}].AttnQWeight");
            AssertFinite(l.Fc1Weight, $"Layers[{i}].Fc1Weight");
            AssertFinite(l.FinalLayerNormWeight, $"Layers[{i}].FinalLayerNormWeight");
        }
    }

    private static void AssertFinite(float[] values, string label)
    {
        Assert.True(values.Length > 0, $"{label} is empty");
        foreach (var v in values)
            Assert.True(float.IsFinite(v), $"{label} contains a non-finite value");
    }
}
