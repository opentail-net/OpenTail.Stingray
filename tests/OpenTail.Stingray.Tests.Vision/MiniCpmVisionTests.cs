using System;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vision;
using Xunit;

namespace OpenTail.Stingray.Tests.Vision;

public sealed class MiniCpmVisionTests
{
    private const string ModelFileName = "mmproj-minicpm-v-2_6-f16.gguf";

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
    public void ImagePreprocessor_GeneratesThumbnailAndValidGrid()
    {
        int w = 896;
        int h = 448;
        var rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            rgb[i] = 120;
            rgb[i + 1] = 200;
            rgb[i + 2] = 80;
        }

        var slices = MiniCpmImagePreprocessor.Preprocess(rgb, w, h, imageSize: 448, maxSlices: 9);
        Assert.NotNull(slices);
        Assert.True(slices.Length >= 2);
        Assert.Equal(448, slices[0].Width);
        Assert.Equal(448, slices[0].Height);
        Assert.Equal(3 * 448 * 448, slices[0].Chw.Length);
    }

    [Fact]
    public void ImagePreprocessor_SmallImage_GeneratesSingleSlice()
    {
        int w = 224;
        int h = 224;
        var rgb = new byte[w * h * 3];
        var slices = MiniCpmImagePreprocessor.Preprocess(rgb, w, h, imageSize: 448);
        Assert.Single(slices);
        Assert.Equal(448, slices[0].Width);
        Assert.Equal(448, slices[0].Height);
    }

    [Fact]
    public void MiniCpm_RealModelFile_LoadsAndValidatesMetadata()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var gguf = GgufModel.Open(modelPath);
        Assert.NotNull(gguf);
        Assert.True(gguf.Tensors.Count > 0);

        using var model = MiniCpmVisionModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.Equal(14, model.PatchSize);
        Assert.True(model.EmbeddingDim > 0);
        Assert.True(model.ProjectionDim > 0);
        Assert.True(model.ResamplerQueryCount > 0);
    }
}
