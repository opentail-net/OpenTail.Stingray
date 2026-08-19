using System;
using System.IO;
using System.Linq;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.HunyuanVideo;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class HunyuanVideoRealWeightsTests
{
    private const string ModelFileName = "hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors";

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
    public void HunyuanVideo_RealModelFile_LoadsAndExposesTensors()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var loader = SafetensorsLoader.Open(modelPath);
        Assert.NotNull(loader);
        Assert.True(loader.TensorCount > 0, "HunyuanVideo safetensors must contain tensors");
    }

    [Fact]
    public void HunyuanVideoPipeline_LoadRealModel_ExecutesForwardPass()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var pipeline = HunyuanVideoPipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal("HunyuanVideo", pipeline.Architecture);

        // Generate a 1-frame 32x32 1-step test pass
        var frames = pipeline.Generate(
            prompt: "A cinematic hyperrealistic video of waves crashing against dramatic cliffs",
            width: 32,
            height: 32,
            numFrames: 1,
            steps: 1,
            guidance: 1.0f,
            seed: 42);

        Assert.NotNull(frames);
        Assert.NotEmpty(frames);
        Assert.Single(frames);
        Assert.Equal(32 * 32 * 3, frames[0].Length);
    }
}
