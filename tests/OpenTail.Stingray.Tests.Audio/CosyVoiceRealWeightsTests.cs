using System;
using System.IO;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.CosyVoice;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class CosyVoiceRealWeightsTests
{
    private const string ModelFileName = "cosyvoice2_0.5b.safetensors";

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
    public void CosyVoice_RealModelFile_SafetensorsValid()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var st = SafetensorsLoader.Open(modelPath);
        Assert.NotNull(st);
        Assert.True(st.TensorCount > 0, "CosyVoice safetensors must contain tensors");
    }

    [Fact]
    public void CosyVoicePipeline_LoadRealSafetensors_SynthesizesAudio()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var pipeline = CosyVoicePipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal("CosyVoice3", pipeline.Architecture);
        Assert.Equal(24000, pipeline.DefaultSampleRate);

        var request = new AudioGenerationRequest
        {
            Text = "CosyVoice expressive multilingual speech synthesis with zero-shot voice cloning.",
            Voice = "default",
            Speed = 1.0f
        };

        var result = pipeline.Generate(request);
        Assert.NotNull(result);
        Assert.Equal(24000, result.SampleRate);
        Assert.True(result.Samples.Length > 0);
        Assert.True(result.Duration.TotalSeconds > 0.5);

        for (int i = 0; i < result.Samples.Length; i++)
        {
            Assert.False(float.IsNaN(result.Samples[i]), $"NaN in CosyVoice sample {i}");
            Assert.False(float.IsInfinity(result.Samples[i]), $"Infinity in CosyVoice sample {i}");
        }
    }
}
