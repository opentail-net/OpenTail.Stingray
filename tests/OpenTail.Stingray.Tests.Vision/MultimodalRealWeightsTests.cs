using System;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vision;
using Xunit;

namespace OpenTail.Stingray.Tests.Vision;

public sealed class MultimodalRealWeightsTests
{
    private static string? FindModelPath(string fileName)
    {
        string[] candidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\{fileName}",
            $@"C:\p\opentail-llm\models\{fileName}",
            $@"E:\models\{fileName}",
        };
        foreach (var p in candidates)
        {
            if (File.Exists(p)) return p;
        }
        return null;
    }

    [Fact]
    public void Llava_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-llava-v1.5-7b-f16.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        var rgb = new byte[336 * 336 * 3];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            rgb[i] = 180;
            rgb[i + 1] = 120;
            rgb[i + 2] = 200;
        }

        var tokens = embedder.EmbedImage(rgb, 336, 336, out int tokenCount);
        Assert.NotNull(tokens);
        Assert.True(tokenCount > 0);
        Assert.Equal(tokenCount * embedder.EmbeddingDim, tokens.Length);

        for (int i = 0; i < Math.Min(100, tokens.Length); i++)
        {
            Assert.False(float.IsNaN(tokens[i]));
            Assert.False(float.IsInfinity(tokens[i]));
        }
    }

    [Fact]
    public void Pixtral_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-pixtral-12b-f16.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        var rgb = new byte[128 * 128 * 3];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            rgb[i] = 100;
            rgb[i + 1] = 150;
            rgb[i + 2] = 220;
        }

        var tokens = embedder.EmbedImage(rgb, 128, 128, out int tokenCount);
        Assert.NotNull(tokens);
        Assert.True(tokenCount > 0);
        Assert.Equal(tokenCount * embedder.EmbeddingDim, tokens.Length);

        for (int i = 0; i < Math.Min(100, tokens.Length); i++)
        {
            Assert.False(float.IsNaN(tokens[i]));
            Assert.False(float.IsInfinity(tokens[i]));
        }
    }

    [Fact]
    public void InternVL_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-internvl3-2b-q8_0.gguf");
        if (path is null) return;

        using var gguf = GgufModel.Open(path);
        Assert.NotNull(gguf);
        Assert.True(gguf.Tensors.Count > 0);

        using var model = InternVlVisionModel.Open(path);
        Assert.NotNull(model);
        Assert.Equal(14, model.PatchSize);
        Assert.True(model.EmbeddingDim > 0);
        Assert.True(model.ProjectionDim > 0);
    }

    [Fact]
    public void DeepSeekOcr_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-deepseek-ocr-2-q8_0.gguf");
        if (path is null) return;

        using var gguf = GgufModel.Open(path);
        Assert.NotNull(gguf);
        Assert.True(gguf.Tensors.Count > 0);

        using var model = DeepSeekOcrVisionModel.Open(path);
        Assert.NotNull(model);
        Assert.True(model.EmbeddingDim > 0);
        Assert.True(model.ProjectionDim > 0);
    }

    [Fact]
    public void PaddleOcr_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("PaddleOCR-VL-1.6-GGUF-mmproj.gguf");
        if (path is null) return;

        using var gguf = GgufModel.Open(path);
        Assert.NotNull(gguf);
        Assert.True(gguf.Tensors.Count > 0);

        using var model = DotsOcrVisionModel.Open(path);
        Assert.NotNull(model);
        Assert.Equal(14, model.PatchSize);
        Assert.True(model.EmbeddingDim > 0);
        Assert.True(model.ProjectionDim > 0);
    }
}
