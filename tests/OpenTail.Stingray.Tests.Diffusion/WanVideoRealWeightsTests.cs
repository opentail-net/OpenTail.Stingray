
namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class WanVideoRealWeightsTests : HeavyTestBase
{
    private const string ModelFileName = "Wan2.1-T2V-1.3B-Q4_0.gguf";

    private static string? FindModelPath(string fileName) => OpenTail.Stingray.Tests.Audio.AudioTestPaths.FindModelFile(fileName);

    [Fact]
    public void WanVideo_RealModelFile_LoadsAndExposesTensors()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Wan Video 2.1 GGUF must contain tensors");

        using var pipeline = WanPipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal("WanVideo", pipeline.Architecture);
    }
}
