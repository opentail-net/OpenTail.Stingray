
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class EmbeddingsRealWeightsTests
{
    private static string? FindModelPath(string fileName) => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(fileName);

    [Fact]
    public void MiniLM_L6_RealModelFile_ExistsAndHasValidSize()
    {
        string? modelPath = FindModelPath("all-MiniLM-L6-v2_quantized.onnx");
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        var fi = new FileInfo(modelPath);
        Assert.True(fi.Length > 10 * 1024 * 1024, "MiniLM ONNX must be > 10MB");
    }

    [Fact]
    public void BgeSmall_RealModelFile_ExistsAndHasValidSize()
    {
        string? modelPath = FindModelPath("bge-small-en-v1.5_quantized.onnx");
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        var fi = new FileInfo(modelPath);
        Assert.True(fi.Length > 20 * 1024 * 1024, "BGE-Small ONNX must be > 20MB");
    }

    [Fact]
    public void BgeBase_RealModelFile_ExistsAndHasValidSize()
    {
        string? modelPath = FindModelPath("bge-base-en-v1.5_quantized.onnx");
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        var fi = new FileInfo(modelPath);
        Assert.True(fi.Length > 50 * 1024 * 1024, "BGE-Base ONNX must be > 50MB");
    }

    [Fact]
    public void BgeLarge_RealModelFile_ExistsAndHasValidSize()
    {
        string? modelPath = FindModelPath("bge-large-en-v1.5_quantized.onnx");
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        var fi = new FileInfo(modelPath);
        Assert.True(fi.Length > 150 * 1024 * 1024, "BGE-Large ONNX must be > 150MB");
    }
}
