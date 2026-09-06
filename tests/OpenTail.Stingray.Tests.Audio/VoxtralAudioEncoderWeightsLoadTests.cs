
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real load validation for VoxtralAudioEncoderWeights against the real downloaded
/// checkpoint (models/_models/voxtral-mini-realtime/model.safetensors, ~8.9GB bf16-cast-to-f32
/// safetensors).</summary>
public sealed class VoxtralAudioEncoderWeightsLoadTests : HeavyTestBase
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
        string? path = FindRepoFile("models/_models/voxtral-mini-realtime/model.safetensors");
        Assert.SkipUnless(path != null, "voxtral-mini-realtime model.safetensors not found");

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(path!);
        var w = new OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights(loader);

        AssertFinite(w.Conv1Weight, "Conv1Weight");
        AssertFinite(w.Conv1Bias, "Conv1Bias");
        AssertFinite(w.Conv2Weight, "Conv2Weight");
        AssertFinite(w.Conv2Bias, "Conv2Bias");
        AssertFinite(w.NormWeight, "NormWeight");
        AssertFinite(w.Projector1Weight, "Projector1Weight");
        AssertFinite(w.Projector2Weight, "Projector2Weight");

        Assert.Equal(OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights.NumLayers, w.Layers.Length);
        for (int i = 0; i < w.Layers.Length; i++)
        {
            var l = w.Layers[i];
            AssertFinite(l.AttnNorm, $"Layers[{i}].AttnNorm");
            AssertFinite(l.QWeight, $"Layers[{i}].QWeight");
            AssertFinite(l.QBias, $"Layers[{i}].QBias");
            AssertFinite(l.KWeight, $"Layers[{i}].KWeight");
            AssertFinite(l.VWeight, $"Layers[{i}].VWeight");
            AssertFinite(l.VBias, $"Layers[{i}].VBias");
            AssertFinite(l.OWeight, $"Layers[{i}].OWeight");
            AssertFinite(l.OBias, $"Layers[{i}].OBias");
            AssertFinite(l.FinalNorm, $"Layers[{i}].FinalNorm");
            AssertFinite(l.GateWeight, $"Layers[{i}].GateWeight");
            AssertFinite(l.UpWeight, $"Layers[{i}].UpWeight");
            AssertFinite(l.DownWeight, $"Layers[{i}].DownWeight");
            AssertFinite(l.DownBias, $"Layers[{i}].DownBias");
        }

        // Real expected element counts, cross-checked against the checkpoint's own config.json.
        const int hidden = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights.HiddenSize;
        const int melBins = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights.NumMelBins;
        const int textHidden = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights.TextHiddenSize;
        const int factor = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights.DownsampleFactor;
        Assert.Equal(hidden * melBins * 3, w.Conv1Weight.Length);
        Assert.Equal(hidden * hidden * 3, w.Conv2Weight.Length);
        Assert.Equal(textHidden * hidden * factor, w.Projector1Weight.Length);
        Assert.Equal(textHidden * textHidden, w.Projector2Weight.Length);
    }

    private static void AssertFinite(float[] values, string label)
    {
        Assert.True(values.Length > 0, $"{label} is empty");
        foreach (var v in values)
            Assert.True(float.IsFinite(v), $"{label} contains a non-finite value");
    }
}
