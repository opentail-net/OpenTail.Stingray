
namespace OpenTail.Stingray.Tests.Vision;

public sealed class UnifiedVisionPipelineTests
{
    private static string? ResolveModelPath(string relativePath)
    {
        string[] searchRoots =
        [
            Directory.GetCurrentDirectory(),
            AppDomain.CurrentDomain.BaseDirectory,
            Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "../../../..")),
            @"C:\Git-Public\OpenTail.Stingray"
        ];

        foreach (var root in searchRoots)
        {
            var full = Path.Combine(root, relativePath);
            if (File.Exists(full)) return full;
        }

        return null;
    }

    [Fact]
    public void Open_ThrowsOnMissingFile()
    {
        Assert.Throws<FileNotFoundException>(() => UnifiedVisionPipeline.Open("nonexistent_model.gguf"));
    }

    [Fact]
    public void Open_LoadsAvailableMmprojModelsAndProducesEmbeddings()
    {
        string[] candidateNames =
        [
            "models/mmproj-gemma-4-12b-it-qat-q4_0.gguf",
            "models/gemma-4-E4B-it-mmproj.gguf",
            "models/mmproj-gemma-3-4b-it-f16.gguf"
        ];

        int testedCount = 0;
        foreach (var rel in candidateNames)
        {
            var resolved = ResolveModelPath(rel);
            if (resolved == null) continue;

            using var embedder = UnifiedVisionPipeline.Open(resolved);
            Assert.NotNull(embedder);
            Assert.NotEmpty(embedder.ProjectorType);
            Assert.True(embedder.EmbeddingDim > 0);
            Assert.True(embedder.ImageWidth > 0);
            Assert.True(embedder.ImageHeight > 0);

            // Generate synthetic solid RGB test image matching the native dimension
            int w = embedder.ImageWidth;
            int h = embedder.ImageHeight;
            byte[] rgb = new byte[w * h * 3];
            for (int i = 0; i < rgb.Length; i += 3)
            {
                rgb[i] = 120;
                rgb[i + 1] = 80;
                rgb[i + 2] = 200;
            }

            float[] tokens = embedder.EmbedImage(rgb, w, h, out int tokenCount);
            Assert.True(tokenCount > 0);
            Assert.Equal(tokenCount * embedder.EmbeddingDim, tokens.Length);

            // Check no NaNs
            for (int i = 0; i < Math.Min(tokens.Length, 100); i++)
            {
                Assert.False(float.IsNaN(tokens[i]), $"Found NaN in {embedder.ProjectorType} token output at index {i}");
            }

            testedCount++;
        }

        Assert.SkipUnless(testedCount > 0, "no mmproj test fixture available in models/ (expected in CI, which doesn't have the real model files)");
    }
}
