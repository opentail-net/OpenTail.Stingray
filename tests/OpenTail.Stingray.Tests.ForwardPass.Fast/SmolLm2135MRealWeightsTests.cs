
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class SmolLm2135MRealWeightsTests
{
    private const string ModelFileName = "SmolLM2-135M-Instruct-Q4_K_M.gguf";

    private static string? FindModelPath(string fileName) => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(fileName);

    [Fact]
    public void SmolLM2_135M_RealModelFile_LoadsAndInspectsMetadata()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "SmolLM2 135M GGUF must contain tensors");
        Assert.True(model.Metadata.Count > 0, "SmolLM2 135M GGUF must contain metadata");
    }
}
