using System;
using System.IO;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Chatterbox;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class ChatterboxRealWeightsTests
{
    private const string ModelFileName = "chatterbox-speech_encoder.onnx";

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
    public void Chatterbox_RealModelFile_OnnxHeaderValidAndSynthesizesSpeech()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        var fileInfo = new FileInfo(modelPath);
        Assert.True(fileInfo.Length > 50 * 1024 * 1024, "Chatterbox ONNX model file must be > 50MB");

        var pipeline = new ChatterboxPipeline();
        var request = new AudioGenerationRequest
        {
            Text = "Chatterbox turbo zero shot expressive text to speech synthesis.",
            Voice = "nova",
            Speed = 1.0f
        };

        var result = pipeline.Generate(request);
        Assert.NotNull(result);
        Assert.Equal(24000, result.SampleRate);
        Assert.True(result.Samples.Length > 0);
        Assert.True(result.Duration.TotalSeconds > 0.5);

        for (int i = 0; i < result.Samples.Length; i++)
        {
            Assert.False(float.IsNaN(result.Samples[i]), $"NaN in Chatterbox sample {i}");
            Assert.False(float.IsInfinity(result.Samples[i]), $"Infinity in Chatterbox sample {i}");
        }
    }
}
