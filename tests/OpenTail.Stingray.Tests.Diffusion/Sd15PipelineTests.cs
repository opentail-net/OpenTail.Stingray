
namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class Sd15PipelineTests : HeavyTestBase
{
    private const string ModelFileName = "v1-5-pruned-emaonly.safetensors";

    private static string? FindModelPath(string fileName) => OpenTail.Stingray.Tests.Audio.AudioTestPaths.FindModelFile(fileName);

    [Fact]
    public void Sd15_RealModelFile_SafetensorsValid()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var st = SafetensorsLoader.Open(modelPath);
        Assert.NotNull(st);
        Assert.True(st.TensorCount > 0, "SD 1.5 safetensors must contain tensors");
    }

    [Fact]
    public void Sd15Pipeline_LoadRealModel_InitializesPipeline()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var pipeline = StableDiffusionPipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal("StableDiffusion1.5", pipeline.Architecture);
    }
}
