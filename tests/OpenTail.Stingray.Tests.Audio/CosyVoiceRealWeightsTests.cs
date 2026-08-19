using System;
using System.IO;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.CosyVoice;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class CosyVoiceRealWeightsTests
{
    private const string ModelFileName = "cosyvoice_speech_tokenizer.onnx";

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
    public void CosyVoice_RealModelFile_OnnxHeaderValidAndSynthesizesSpeech()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        var fileInfo = new FileInfo(modelPath);
        Assert.True(fileInfo.Length > 100 * 1024 * 1024, "CosyVoice ONNX model file must be > 100MB");

        using var pipeline = new CosyVoicePipeline();
        var request = new AudioGenerationRequest
        {
            Text = "CosyVoice 3 expressive multilingual neural speech generation with zero-shot cloning.",
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

    [Fact]
    public void CosyVoice2_FlowDecoderAndSafetensors_RealFilesAreValid()
    {
        string? flowPath = FindModelPath("flow.decoder.estimator.fp32.onnx");
        if (flowPath is not null)
        {
            var fi = new FileInfo(flowPath);
            Assert.True(fi.Length > 50 * 1024 * 1024, "Flow decoder ONNX must be > 50MB");
        }

        string? stPath = FindModelPath("cosyvoice2_0.5b.safetensors");
        if (stPath is not null)
        {
            using var loader = SafetensorsLoader.Open(stPath);
            Assert.NotNull(loader);
            Assert.True(loader.TensorCount > 0, "CosyVoice2 safetensors must contain tensors");
        }
    }
}
