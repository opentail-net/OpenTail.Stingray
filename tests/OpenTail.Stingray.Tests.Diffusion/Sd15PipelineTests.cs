
namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class Sd15PipelineTests
{
    private const string ModelFileName = "v1-5-pruned-emaonly.safetensors";

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
    public void Sd15_RealModelFile_SafetensorsValid()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var st = SafetensorsLoader.Open(modelPath);
        Assert.NotNull(st);
        Assert.True(st.TensorCount > 0, "SD 1.5 safetensors must contain tensors");
    }

    [Fact]
    public void Sd15Pipeline_LoadRealModel_InitializesPipeline()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var pipeline = StableDiffusionPipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal("StableDiffusion1.5", pipeline.Architecture);
    }
}
