
namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class ZImageRealWeightsTests : HeavyTestBase
{
    private const string ModelFileName = "z_image_turbo-Q4_0.gguf";

    private static string? FindModelPath(string fileName) => OpenTail.Stingray.Tests.Audio.AudioTestPaths.FindModelFile(fileName);

    [Fact]
    public void ZImage_RealModelFile_GgufLoadsAndInspectsTensors()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Z-Image GGUF must contain tensors");
    }
}
