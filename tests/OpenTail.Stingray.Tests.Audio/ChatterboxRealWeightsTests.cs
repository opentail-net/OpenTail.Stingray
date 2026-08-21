using System;
using System.IO;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Chatterbox;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio.Fast;

public sealed class ChatterboxRealWeightsTests : HeavyTestBase
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
    public void ChatterboxPipeline_LoadRealGgufModels_Synthesizes24kHzAudio()
    {
        string? t3Path = FindModelPath("chatterbox-turbo-t3-q4_k.gguf");
        string? s3GenPath = FindModelPath("chatterbox-turbo-s3gen-q4_k.gguf");

        if (t3Path is null) return;

        using var pipeline = ChatterboxPipeline.Load(t3Path, s3GenPath);
        Assert.NotNull(pipeline);
        Assert.Equal("Chatterbox-Turbo", pipeline.Architecture);
        Assert.Equal(24000, pipeline.DefaultSampleRate);

        var request = new AudioGenerationRequest
        {
            Text = "Chatterbox Turbo zero-shot expressive voice generation running natively in OpenTail Stingray.",
            Voice = "nova",
            Speed = 1.0f
        };

        var result = pipeline.Generate(request);
        Assert.NotNull(result);
        Assert.Equal(24000, result.SampleRate);
        Assert.NotEmpty(result.Samples);
        Assert.True(result.Duration.TotalSeconds > 0.5, "Generated audio must have positive duration");

        // Verify audio is non-silent and finite
        float energy = 0f;
        for (int i = 0; i < result.Samples.Length; i++)
        {
            float s = result.Samples[i];
            Assert.False(float.IsNaN(s), $"Sample {i} must not be NaN");
            Assert.False(float.IsInfinity(s), $"Sample {i} must not be Infinity");
            energy += s * s;
        }
        Assert.True(energy > 0.1f, "Audio energy must be non-zero");
    }

    [Fact]
    public void Chatterbox_T3_GgufRealModelFile_LoadsAndInspectsTensors()
    {
        string? modelPath = FindModelPath("chatterbox-turbo-t3-q4_k.gguf");
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Chatterbox T3 GGUF must have tensors");
        Assert.True(model.Metadata.Count > 0, "Chatterbox T3 GGUF must have metadata");
    }

    [Fact]
    public void Chatterbox_S3Gen_GgufRealModelFile_LoadsAndInspectsTensors()
    {
        string? modelPath = FindModelPath("chatterbox-turbo-s3gen-q4_k.gguf");
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Chatterbox S3Gen GGUF must have tensors");
    }
}
