using System;
using System.IO;
using System.Linq;
using System.Text;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.QwenASR;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio.Fast;

public sealed class QwenAsrRealWeightsTests : HeavyTestBase
{
    private const string ModelFileName = "qwen3-asr-0.6b-q4_k.gguf";

    private static string? FindModelPath(string modelFile)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\{modelFile}",
            $@"C:\p\opentail-llm\models\{modelFile}",
            $@"E:\models\{modelFile}",
        };
        foreach (var p in absoluteCandidates)
        {
            if (File.Exists(p)) return p;
        }

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", modelFile);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void QwenAsr_RealWeights_GgufHeaderAndTensorMetadata_Valid()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);

        Assert.True(model.Header.Version >= 2, "Expected GGUF version >= 2");
        Assert.True(model.Tensors.Count > 0, "GGUF model must contain tensors");
        Assert.True(model.Metadata.Count > 0, "GGUF model must contain metadata");
    }

    [Fact]
    public void QwenAsrPipeline_LoadRealGguf_TranscribesAudioEndToEnd()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var pipeline = QwenAsrPipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal("Alibaba-Qwen3-ASR", pipeline.Architecture);
        Assert.Equal(16000, pipeline.SampleRate);

        int sampleRate = 16000;
        int durationSec = 2;
        var pcm = new float[sampleRate * durationSec];

        // Synthesize a speech-frequency carrier tone + harmonics
        for (int i = 0; i < pcm.Length; i++)
        {
            float t = (float)i / sampleRate;
            pcm[i] = 0.3f * MathF.Sin(2.0f * MathF.PI * 300.0f * t)
                   + 0.2f * MathF.Sin(2.0f * MathF.PI * 600.0f * t)
                   + 0.1f * MathF.Sin(2.0f * MathF.PI * 1200.0f * t);
        }

        var request = new SpeechToTextRequest
        {
            AudioSamples = pcm,
            SampleRate = sampleRate,
            Language = "en",
            Task = SpeechTask.Transcribe
        };

        var result = pipeline.Transcribe(request);

        Assert.NotNull(result);
        Assert.Equal("en", result.Language);
        Assert.True(result.Duration.TotalSeconds >= 1.9);
        Assert.NotNull(result.Segments);
        Assert.NotEmpty(result.Segments);
    }
}
