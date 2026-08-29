
namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class WanVideoRealWeightsTests
{
    private const string ModelFileName = "Wan2.1-T2V-1.3B-Q4_0.gguf";

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
    public void WanVideo_RealModelFile_LoadsAndExposesTensors()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Wan Video 2.1 GGUF must contain tensors");

        using var pipeline = WanPipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal("WanVideo", pipeline.Architecture);
    }
}
