
namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class ZImageRealWeightsTests
{
    private const string ModelFileName = "z_image_turbo-Q4_0.gguf";

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
            // Check models/ directly (curated/pinned checkpoints), then models/_models/ (the
            // rotating disk-cache set, symlinked to F:\_models -- checkpoints commonly live here
            // instead, and the previous version of this method never checked it, matching the
            // same systemic gap PerformanceLeague.md's "Known Measurement Gaps" section already
            // flagged for ParakeetRealWeightsTests).
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var pModels = Path.Combine(dir, "models", "_models", fileName);
            if (File.Exists(pModels)) return pModels;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void ZImage_RealModelFile_GgufLoadsAndInspectsTensors()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Z-Image GGUF must contain tensors");
    }
}
