
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
        AssertNonEmpty(w.Projector1Weight, "Projector1Weight");
        AssertNonEmpty(w.Projector2Weight, "Projector2Weight");

        Assert.Equal(OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights.NumLayers, w.Layers.Length);
        for (int i = 0; i < w.Layers.Length; i++)
        {
            var l = w.Layers[i];
            AssertFinite(l.AttnNorm, $"Layers[{i}].AttnNorm");
            AssertNonEmpty(l.QWeight, $"Layers[{i}].QWeight");
            AssertFinite(l.QBias, $"Layers[{i}].QBias");
            AssertNonEmpty(l.KWeight, $"Layers[{i}].KWeight");
            AssertNonEmpty(l.VWeight, $"Layers[{i}].VWeight");
            AssertFinite(l.VBias, $"Layers[{i}].VBias");
            AssertNonEmpty(l.OWeight, $"Layers[{i}].OWeight");
            AssertFinite(l.OBias, $"Layers[{i}].OBias");
            AssertFinite(l.FinalNorm, $"Layers[{i}].FinalNorm");
            AssertNonEmpty(l.GateWeight, $"Layers[{i}].GateWeight");
            AssertNonEmpty(l.UpWeight, $"Layers[{i}].UpWeight");
            AssertNonEmpty(l.DownWeight, $"Layers[{i}].DownWeight");
            AssertFinite(l.DownBias, $"Layers[{i}].DownBias");
        }

        // Real expected element counts, cross-checked against the checkpoint's own config.json.
        // Q8_0-quantized tensors (perf-sweep Phase 1.2e) are checked in bytes-per-Q8_0-block form.
        const int hidden = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights.HiddenSize;
        const int melBins = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights.NumMelBins;
        const int textHidden = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights.TextHiddenSize;
        const int factor = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights.DownsampleFactor;
        Assert.Equal(hidden * melBins * 3, w.Conv1Weight.Length);
        Assert.Equal(hidden * hidden * 3, w.Conv2Weight.Length);
        Assert.Equal(ExpectedQ8_0Bytes(textHidden * hidden * factor, hidden * factor), w.Projector1Weight.Length);
        Assert.Equal(ExpectedQ8_0Bytes(textHidden * textHidden, textHidden), w.Projector2Weight.Length);
    }

    private static int ExpectedQ8_0Bytes(int floatCount, int cols) => (floatCount / cols) * (cols / 32) * 34;

    private static void AssertFinite(float[] values, string label)
    {
        Assert.True(values.Length > 0, $"{label} is empty");
        foreach (var v in values)
            Assert.True(float.IsFinite(v), $"{label} contains a non-finite value");
    }

    /// <summary>Q8_0-quantized weight buffers (perf-sweep Phase 1.2e) are opaque bytes, not
    /// individually finite-checkable floats -- non-emptiness is the load-succeeded check here;
    /// correctness is covered by `ConvertF32ToQ8_0VerificationTests` and the real-audio transcript
    /// re-check in `VoxtralGenerationLoopTests`/`VoxtralEndToEndPrefillTests`.</summary>
    private static void AssertNonEmpty(byte[] values, string label) =>
        Assert.True(values.Length > 0, $"{label} is empty");
}
