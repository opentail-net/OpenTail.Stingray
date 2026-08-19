using System;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.LTXVideo;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class LtxVideoRealWeightsTests
{
    private const string ModelFileName = "ltx-video-2b-v0.9.1.safetensors";

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
    public void LtxVideo_RealModelFile_LoadsAndExposesPipeline()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var loader = SafetensorsLoader.Open(modelPath);
        Assert.NotNull(loader);
        Assert.True(loader.TensorCount > 0, "LTX-Video safetensors must contain tensors");

        using var pipeline = LtxVideoPipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal("LTX-Video", pipeline.Architecture);
    }
}
