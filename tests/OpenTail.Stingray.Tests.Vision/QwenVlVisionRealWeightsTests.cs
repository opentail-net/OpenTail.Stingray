using System;
using System.IO;
using System.Linq;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vision;
using Xunit;

namespace OpenTail.Stingray.Tests.Vision;

public sealed class QwenVlVisionRealWeightsTests
{
    private const string ModelFileName = "mmproj-qwen2.5-vl-7b-f16.gguf";

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
    public void QwenVl_DumpTensorInventory()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var gguf = GgufModel.Open(modelPath);
        using var sw = new StreamWriter(@"C:\Git-Public\OpenTail.Stingray\qwen_tensors.txt");
        sw.WriteLine($"Total tensors: {gguf.Tensors.Count}");
        foreach (var t in gguf.Tensors)
        {
            sw.WriteLine($"{t.Name} | {t.DType} | [{string.Join(", ", t.Dimensions)}]");
        }
    }

    [Fact]
    public void QwenVl_RealModelFile_LoadsAndValidatesMetadata()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var gguf = GgufModel.Open(modelPath);
        Assert.NotNull(gguf);
        Assert.True(gguf.Tensors.Count > 0, "Qwen2.5-VL mmproj must contain tensors");
        Assert.True(gguf.Metadata.Count > 0, "Qwen2.5-VL mmproj must contain metadata");

        using var model = QwenVlVisionModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.Equal(14, model.PatchSize);
        Assert.True(model.EmbeddingDim > 0);
        Assert.True(model.ProjectionDim > 0);
        Assert.True(model.LayerCount > 0);
    }

    [Fact]
    public void QwenVl_RealModel_EmbedsImageEndToEnd()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var embedder = UnifiedVisionPipeline.Open(modelPath);
        Assert.NotNull(embedder);
        Assert.True(embedder.EmbeddingDim > 0);

        int w = 224;
        int h = 224;
        var rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            rgb[i] = 180;
            rgb[i + 1] = 90;
            rgb[i + 2] = 220;
        }

        float[] tokens = embedder.EmbedImage(rgb, w, h, out int tokenCount);
        Assert.True(tokenCount > 0, "Token count must be > 0");
        Assert.Equal(tokenCount * embedder.EmbeddingDim, tokens.Length);

        for (int i = 0; i < Math.Min(tokens.Length, 100); i++)
        {
            Assert.False(float.IsNaN(tokens[i]), $"Found NaN at {i}");
            Assert.False(float.IsInfinity(tokens[i]), $"Found Infinity at {i}");
        }
    }
}
