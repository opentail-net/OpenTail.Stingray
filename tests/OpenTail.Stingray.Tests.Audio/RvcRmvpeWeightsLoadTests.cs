
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real weight-loading validation for RvcRmvpeWeights against the real rvc-f16.gguf checkpoint.</summary>
public sealed class RvcRmvpeWeightsLoadTests : HeavyTestBase
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
    public void LoadRvcRmvpeWeights_RealCheckpoint_AllTensorsPresentAndFinite()
    {
        string? path = FindRepoFile("examples/audio.cpp/models/RVC-GGUF/rvc-f16.gguf");
        Assert.SkipUnless(path != null, "rvc-f16.gguf not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        var source = new OpenTail.Stingray.Audio.Rvc.RvcPackedTensorSource(model);
        var w = new OpenTail.Stingray.Audio.Rvc.RvcRmvpeWeights(source);

        Assert.Equal(5, w.EncoderLevels.Length);
        Assert.Equal(4, w.IntermediateLevels.Length);
        Assert.Equal(5, w.DecoderLevels.Length);

        void CheckFinite(float[] arr) => Assert.All(arr, v => Assert.True(float.IsFinite(v)));

        CheckFinite(w.EncoderInputBn.Weight);
        foreach (var level in w.EncoderLevels)
            foreach (var block in level.Blocks)
            {
                CheckFinite(block.Conv0Weight);
                CheckFinite(block.Conv3Weight);
                CheckFinite(block.Bn1.RunningVar);
                if (block.ShortcutWeight is not null) CheckFinite(block.ShortcutWeight);
            }
        foreach (var level in w.IntermediateLevels)
            foreach (var block in level.Blocks)
                CheckFinite(block.Conv0Weight);
        foreach (var level in w.DecoderLevels)
        {
            CheckFinite(level.UpsampleWeight);
            foreach (var block in level.Conv2Blocks)
                CheckFinite(block.Conv0Weight);
        }
        CheckFinite(w.CnnWeight);
        CheckFinite(w.GruForward.WeightIh);
        CheckFinite(w.GruReverse.WeightHh);
        CheckFinite(w.FcOutWeight);

        Assert.Equal(360 * 512, w.FcOutWeight.Length);
        Assert.Equal(3 * 16 * 3 * 3, w.CnnWeight.Length);

        Console.Error.WriteLine("[RvcRmvpe] all real tensors loaded and finite");
    }
}
