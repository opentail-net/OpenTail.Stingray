
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real load validation for RvcSynthesizerWeights against the bundled default voice in
/// rvc-f16.gguf (unprefixed enc_p/flow/dec/emb_g tensors, confirmed present alongside the
/// support_hubert_base/support_rmvpe components).</summary>
public sealed class RvcSynthesizerWeightsLoadTests : HeavyTestBase
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
        string? path = FindRepoFile("examples/audio.cpp/models/RVC-GGUF/rvc-f16.gguf");
        Assert.SkipUnless(path != null, "rvc-f16.gguf not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        var source = new OpenTail.Stingray.Audio.Rvc.RvcPackedTensorSource(model);

        // Real checkpoint bundles multiple named voices; "voice_v2_default_checkpoint" is v2
        // (768-wide phone embedding) with F0, 40kHz sample rate -- confirmed via a real tensor-name
        // dump (RvcSynthTensorNameProbeDebugTest) and the reference's voice_model.cpp conventions.
        var w = new OpenTail.Stingray.Audio.Rvc.RvcSynthesizerWeights(source, "voice_v2_default_checkpoint", sampleRate: 40000, v1: false, hasF0: true);

        Assert.Equal(4, w.UpsampleRates.Length);
        Assert.Equal([10, 10, 2, 2], w.UpsampleRates);
        Assert.Equal(400, w.HopSamples);

        AssertFinite(w.EmbPhoneWeight, "EmbPhoneWeight");
        AssertFinite(w.EmbPhoneBias, "EmbPhoneBias");
        AssertFinite(w.EmbPitchWeight!, "EmbPitchWeight");
        AssertFinite(w.ProjWeight, "ProjWeight");
        AssertFinite(w.EmbGWeight, "EmbGWeight");
        AssertFinite(w.DecConvPostWeight, "DecConvPostWeight");

        for (int i = 0; i < OpenTail.Stingray.Audio.Rvc.RvcSynthesizerWeights.TextLayers; i++)
        {
            var layer = w.AttnLayers[i];
            AssertFinite(layer.ConvQ.Weight, $"AttnLayers[{i}].ConvQ.Weight");
            AssertFinite(layer.EmbRelK, $"AttnLayers[{i}].EmbRelK");
            AssertFinite(layer.EmbRelV, $"AttnLayers[{i}].EmbRelV");
        }

        foreach (var flow in w.Flows)
        {
            AssertFinite(flow.Pre.Weight, "Flow.Pre.Weight");
            AssertFinite(flow.CondLayer.Weight, "Flow.CondLayer.Weight");
            foreach (var l in flow.InLayers) AssertFinite(l.Weight, "Flow.InLayers[].Weight");
            foreach (var l in flow.ResSkipLayers) AssertFinite(l.Weight, "Flow.ResSkipLayers[].Weight");
            AssertFinite(flow.Post.Weight, "Flow.Post.Weight");
        }

        for (int up = 0; up < 4; up++)
        {
            AssertFinite(w.DecUps[up].Weight, $"DecUps[{up}].Weight");
            AssertFinite(w.DecNoiseConvs[up]!.Value.Weight, $"DecNoiseConvs[{up}].Weight");
        }
        foreach (var rb in w.DecResBlocks)
        {
            foreach (var c in rb.Convs1) AssertFinite(c.Weight, "ResBlock.Convs1[].Weight");
            foreach (var c in rb.Convs2) AssertFinite(c.Weight, "ResBlock.Convs2[].Weight");
        }
    }

    private static void AssertFinite(float[] values, string label)
    {
        Assert.True(values.Length > 0, $"{label} is empty");
        foreach (var v in values)
            Assert.True(float.IsFinite(v), $"{label} contains a non-finite value");
    }
}
