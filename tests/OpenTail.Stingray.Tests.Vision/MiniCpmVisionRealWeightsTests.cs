
namespace OpenTail.Stingray.Tests.Vision;

public sealed class MiniCpmVisionRealWeightsTests : HeavyTestBase
{
    private const string ModelFileName = "mmproj-minicpm-v-2_6-f16.gguf";

    private static string? FindModelPath(string fileName) => VisionTestPaths.FindAnywhere(fileName);

    [Fact]
    public void MiniCpmV_RealModelFile_LoadsAndExposesVisionTensors()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "MiniCPM-V mmproj must contain tensors");
    }
}
