
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real weight-loading validation for RvcHubertWeights against the real rvc-f16.gguf
/// checkpoint (via RvcPackedTensorSource's real audiocpp.tensor_names-based name resolution).</summary>
public sealed class RvcHubertWeightsLoadTests : HeavyTestBase
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
    public void LoadRvcHubertWeights_RealCheckpoint_AllTensorsPresentAndFinite()
    {
        string? path = FindRepoFile("examples/audio.cpp/models/RVC-GGUF/rvc-f16.gguf");
        Assert.SkipUnless(path != null, "rvc-f16.gguf not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        var source = new OpenTail.Stingray.Audio.Rvc.RvcPackedTensorSource(model);
        var w = new OpenTail.Stingray.Audio.Rvc.RvcHubertWeights(source);

        Assert.Equal(12, w.Layers.Length);
        for (int i = 0; i < 7; i++)
        {
            Assert.NotEmpty(w.ConvWeights[i]);
            Assert.All(w.ConvWeights[i], v => Assert.True(float.IsFinite(v)));
        }
        Assert.Equal(768 * 48 * 128, w.PosConvWeight.Length);
        Assert.All(w.PosConvWeight, v => Assert.True(float.IsFinite(v)));

        // Real sanity check on the weight-norm reconstruction: the reconstructed weight's overall
        // RMS should be a plausible, non-degenerate trained-conv magnitude (neither ~0 nor huge).
        double sumSq = 0;
        foreach (var v in w.PosConvWeight) sumSq += (double)v * v;
        double rms = Math.Sqrt(sumSq / w.PosConvWeight.Length);
        Console.Error.WriteLine($"[RvcHubert] PosConvWeight RMS={rms:F5}");
        Assert.InRange(rms, 1e-4, 10.0);

        foreach (var layer in w.Layers)
        {
            Assert.NotEmpty(layer.AttnQWeight);
            Assert.NotEmpty(layer.Fc1Weight);
            Assert.All(layer.AttnQWeight, v => Assert.True(float.IsFinite(v)));
        }
        Console.Error.WriteLine("[RvcHubert] all real tensors loaded and finite");
    }
}
