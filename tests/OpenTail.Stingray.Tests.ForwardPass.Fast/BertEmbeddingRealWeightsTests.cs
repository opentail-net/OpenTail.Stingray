using System;
using System.IO;
using OpenTail.Stingray.Core.Embeddings;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class BertEmbeddingRealWeightsTests
{
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
    public void BgeSmall_RealGgufModel_ExecutesForwardPassAndProducesDenseVectors()
    {
        string? modelPath = FindModelPath("bge-small-en-v1.5-q8_0.gguf");
        if (modelPath is null) return;

        using var pipeline = new BertGgufEmbeddingPipeline(modelPath);
        Assert.Equal(384, pipeline.EmbeddingDimensions);

        var request = new EmbeddingRequest
        {
            Inputs = new[]
            {
                "Speech recognition and machine learning in C#",
                "High performance audio synthesis runtime",
                "Completely unrelated topic about cooking recipes"
            },
            Normalize = true
        };

        var result = pipeline.Embed(request);
        Assert.NotNull(result);
        Assert.Equal(3, result.Data.Count);

        // Verify vectors are normalized and dense
        foreach (var item in result.Data)
        {
            Assert.Equal(384, item.Vector.Length);
            float norm = 0f;
            foreach (var v in item.Vector) norm += v * v;
            Assert.True(MathF.Abs(MathF.Sqrt(norm) - 1.0f) < 1e-4f, "Vector must be unit normalized");
        }
    }

    [Fact]
    public void AllMiniLM_RealGgufModel_ExecutesForwardPassAndProducesDenseVectors()
    {
        string? modelPath = FindModelPath("all-MiniLM-L6-v2-Q8_0.gguf");
        if (modelPath is null) return;

        using var pipeline = new BertGgufEmbeddingPipeline(modelPath);
        Assert.Equal(384, pipeline.EmbeddingDimensions);

        var request = new EmbeddingRequest
        {
            Inputs = new[] { "Local edge inference without Python runtime" },
            Normalize = true
        };

        var result = pipeline.Embed(request);
        Assert.NotNull(result);
        Assert.Single(result.Data);
        Assert.Equal(384, result.Data[0].Vector.Length);
    }
}
