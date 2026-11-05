using OpenTail.Stingray.Diffusion.SDXL;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class SdxlRealWeightsTests : HeavyTestBase
{
    private const string ModelFileName = "sd_xl_turbo_1.0_fp16.safetensors";

    private static string? FindModelPath(string fileName) => OpenTail.Stingray.Tests.Audio.AudioTestPaths.FindModelFile(fileName);

    [Fact]
    public void Sdxl_RealModelFile_SafetensorsValid()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var st = SafetensorsLoader.Open(modelPath);
        Assert.NotNull(st);
        Assert.True(st.TensorCount > 0, "SDXL Turbo safetensors must contain tensors");
    }

    [Fact]
    public void SdxlPipeline_LoadRealSafetensors_InitializesPipeline()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var pipeline = SdxlPipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal("SDXL", pipeline.Architecture);
    }
}
