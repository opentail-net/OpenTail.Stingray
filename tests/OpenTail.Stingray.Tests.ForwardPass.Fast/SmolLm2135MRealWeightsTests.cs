
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class SmolLm2135MRealWeightsTests
{
    private const string ModelFileName = "SmolLM2-135M-Instruct-Q4_K_M.gguf";

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
    public void SmolLM2_135M_RealModelFile_LoadsAndInspectsMetadata()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "SmolLM2 135M GGUF must contain tensors");
        Assert.True(model.Metadata.Count > 0, "SmolLM2 135M GGUF must contain metadata");
    }
}
