using System;
using System.IO;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class Qwen25RealWeightsTests
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
    public void Qwen25_05B_RealModelFile_LoadsAndInspectsMetadata()
    {
        string? modelPath = FindModelPath("qwen2.5-0.5b-instruct-q4_k_m.gguf");
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Qwen2.5 0.5B GGUF must contain tensors");
        Assert.True(model.Metadata.Count > 0, "Qwen2.5 0.5B GGUF must contain metadata");
    }

    [Fact]
    public void Qwen25_15B_RealModelFile_LoadsAndInspectsMetadata()
    {
        string? modelPath = FindModelPath("qwen2.5-1.5b-instruct-q4_k_m.gguf");
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Qwen2.5 1.5B GGUF must contain tensors");
        Assert.True(model.Metadata.Count > 0, "Qwen2.5 1.5B GGUF must contain metadata");
    }

    [Fact]
    public void Qwen25_3B_RealModelFile_LoadsAndInspectsMetadata()
    {
        string? modelPath = FindModelPath("qwen2.5-3b-instruct-q4_k_m.gguf");
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Qwen2.5 3B GGUF must contain tensors");
        Assert.True(model.Metadata.Count > 0, "Qwen2.5 3B GGUF must contain metadata");
    }
}
