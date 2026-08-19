using System;
using System.IO;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.FunASR;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class FunAsrRealWeightsTests
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
    public void Paraformer_GgufRealModelFile_LoadsAndTranscribes()
    {
        string? modelPath = FindModelPath("paraformer-q8.gguf");
        if (modelPath is null) return;

        using var pipeline = FunAsrPipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal("Alibaba-FunASR-Nano", pipeline.Architecture);

        // Run transcription
        float[] audio = new float[16000 * 2];
        for (int i = 0; i < audio.Length; i++)
        {
            audio[i] = MathF.Sin(2.0f * MathF.PI * 440.0f * i / 16000.0f) * 0.5f;
        }

        var res = pipeline.Transcribe(new SpeechToTextRequest
        {
            AudioSamples = audio,
            SampleRate = 16000,
            Language = "zh"
        });

        Assert.NotNull(res);
        Assert.NotEmpty(res.Segments);
        Assert.False(string.IsNullOrWhiteSpace(res.Text));
    }

    [Fact]
    public void Paraformer_OnnxRealModelFile_LoadsAndTranscribes()
    {
        string? modelPath = FindModelPath("paraformer-zh-small.int8.onnx");
        if (modelPath is null) return;

        using var pipeline = FunAsrPipeline.Load(modelPath);
        Assert.NotNull(pipeline);

        float[] audio = new float[16000 * 2];
        for (int i = 0; i < audio.Length; i++)
        {
            audio[i] = MathF.Sin(2.0f * MathF.PI * 440.0f * i / 16000.0f) * 0.5f;
        }

        var res = pipeline.Transcribe(new SpeechToTextRequest
        {
            AudioSamples = audio,
            SampleRate = 16000,
            Language = "zh"
        });

        Assert.NotNull(res);
        Assert.NotEmpty(res.Segments);
    }

    [Fact]
    public void SenseVoice_OnnxRealModelFile_LoadsAndTranscribes()
    {
        string? modelPath = FindModelPath("sensevoice-small.int8.onnx");
        if (modelPath is null) return;

        using var pipeline = FunAsrPipeline.Load(modelPath);
        Assert.NotNull(pipeline);

        float[] audio = new float[16000 * 2];
        for (int i = 0; i < audio.Length; i++)
        {
            audio[i] = MathF.Sin(2.0f * MathF.PI * 220.0f * i / 16000.0f) * 0.4f;
        }

        var res = pipeline.Transcribe(new SpeechToTextRequest
        {
            AudioSamples = audio,
            SampleRate = 16000,
            Language = "zh"
        });

        Assert.NotNull(res);
        Assert.NotEmpty(res.Segments);
    }
}
