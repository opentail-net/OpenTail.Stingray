using OpenTail.Stingray.Diffusion.LTXVideo;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class LtxVideoRealWeightsTests
{
    private const string ModelFileName = "ltx-video-2b-v0.9.1.safetensors";

    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\{fileName}",
            $@"C:\p\opentail-llm\models\{fileName}",
            $@"E:\models\{fileName}",
        };
        foreach (var p in absoluteCandidates)
        {
            if (File.Exists(p)) return p;
        }

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void LtxVideo_RealModelFile_LoadsAndExposesPipeline()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var loader = SafetensorsLoader.Open(modelPath);
        Assert.NotNull(loader);
        Assert.True(loader.TensorCount > 0, "LTX-Video safetensors must contain tensors");

        using var pipeline = LtxVideoPipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal("LTX-Video", pipeline.Architecture);
    }

    /// <summary>Checks `LtxVideoModel.DetectConfig` against the real v0.9.1 checkpoint's own tensor
    /// shapes (verified directly against the safetensors JSON header --
    /// docs/055-ltx-video-implementation-plan.md's tensor inventory).</summary>
    [Fact]
    public void LtxVideo_RealModelFile_DetectConfigMatchesKnownArchitecture()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var loader = SafetensorsLoader.Open(modelPath);
        var model = new LtxVideoModel(loader);

        Assert.Equal(128, model.InChannels);
        Assert.Equal(128, model.OutChannels);
        Assert.Equal(2048, model.HiddenSize);
        Assert.Equal(32, model.NumHeads);
        Assert.Equal(64, model.HeadDim);
        Assert.Equal(28, model.NumLayers);
        Assert.Equal(2048, model.CrossAttentionDim);
        Assert.Equal(4096, model.CaptionChannels);
        Assert.False(model.CrossAttentionAdaln);
        Assert.False(model.SelfAttentionGated);
        Assert.False(model.CrossAttentionGated);
    }
}
