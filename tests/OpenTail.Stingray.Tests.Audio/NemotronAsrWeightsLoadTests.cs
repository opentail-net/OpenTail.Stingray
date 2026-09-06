
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real load validation for NemotronAsrWeights against the real downloaded GGUF checkpoint
/// (`nvidia/nemotron-3.5-asr-streaming-0.6b`, Q8_0 quant) -- confirms every tensor name/shape this
/// port assumes actually resolves and dequantizes to finite values, before any forward-pass code is
/// written.</summary>
public sealed class NemotronAsrWeightsLoadTests : HeavyTestBase
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
    public void Load_RealCheckpoint_AllTensorsResolveAndAreFinite()
    {
        string? path = FindRepoFile("models/_models/nemotron-asr/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf");
        Assert.SkipUnless(path != null, "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf not found");

        using var w = new OpenTail.Stingray.Audio.NemotronAsr.NemotronAsrWeights(path!);

        Assert.Equal(24, w.NumLayers);
        Assert.Equal(1024, w.HiddenDim);
        Assert.Equal(8, w.NumHeads);
        Assert.Equal(128, w.HeadDim);
        Assert.Equal(4096, w.FfDim);
        Assert.Equal(9, w.ConvKernel);
        Assert.Equal(128, w.FeatIn);
        Assert.Equal(8, w.SubsampleFactor);
        Assert.Equal(256, w.SubsampleChannels);
        Assert.Equal(13088, w.VocabSize);
        Assert.Equal(13087, w.BlankTokenId);
        Assert.Equal(640, w.PredHidden);
        Assert.Equal(640, w.JointDim);
        Assert.Equal(2, w.PredNumLayers);

        AssertFinite(w.PreConv0Weight, "PreConv0Weight");
        AssertFinite(w.PreOutWeight, "PreOutWeight");
        AssertFinite(w.PosEncTable, "PosEncTable");
        AssertFinite(w.MelFilterbank, "MelFilterbank");
        AssertFinite(w.PredEmbedWeight, "PredEmbedWeight");
        AssertFinite(w.JointNet2Weight, "JointNet2Weight");

        foreach (var layer in w.Layers)
        {
            AssertFinite(layer.AttnQWeight, "AttnQWeight");
            AssertFinite(layer.AttnPosWeight, "AttnPosWeight");
            AssertFinite(layer.AttnPosBiasU, "AttnPosBiasU");
            AssertFinite(layer.ConvDwWeight, "ConvDwWeight");
            AssertFinite(layer.ConvNormWeight, "ConvNormWeight");
            AssertFinite(layer.Ff1Linear1Weight, "Ff1Linear1Weight");
        }

        AssertFinite(w.PredLstm0.WeightIh, "PredLstm0.WeightIh");
        AssertFinite(w.PredLstm0.WeightHh, "PredLstm0.WeightHh");
        AssertFinite(w.PredLstm1.WeightIh, "PredLstm1.WeightIh");
        Assert.Equal(4 * w.PredHidden * w.PredEmbedDim, w.PredLstm0.WeightIh.Length);
        Assert.Equal(4 * w.PredHidden * w.PredHidden, w.PredLstm0.WeightHh.Length);
        Assert.Equal(4 * w.PredHidden * w.PredHidden, w.PredLstm1.WeightIh.Length); // layer 1's input is layer 0's hidden

        Console.Error.WriteLine($"[NemotronAsrLoad] layers={w.NumLayers} hidden={w.HiddenDim} vocab={w.VocabSize} blank={w.BlankTokenId} predHidden={w.PredHidden}");
    }

    private static void AssertFinite(float[] values, string label)
    {
        Assert.True(values.Length > 0, $"{label} is empty");
        foreach (var v in values)
            Assert.True(float.IsFinite(v), $"{label} contains a non-finite value");
    }
}
