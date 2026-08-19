using System;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.SDXL;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class SdxlRealWeightsTests
{
    private const string ModelFileName = "sd_xl_turbo_1.0_fp16.safetensors";

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
    public void Sdxl_RealModelFile_SafetensorsValid()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var st = SafetensorsLoader.Open(modelPath);
        Assert.NotNull(st);
        Assert.True(st.TensorCount > 0, "SDXL Turbo safetensors must contain tensors");
    }

    [Fact]
    public void SdxlPipeline_LoadRealSafetensors_InitializesPipeline()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var pipeline = SdxlPipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal("SDXL", pipeline.Architecture);
    }
}
