using System;
using System.IO;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Kokoro;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio.Fast;

public sealed class KokoroRealWeightsTests : HeavyTestBase
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
    public void Kokoro_RealGgufModelAndVoice_SynthesizesAudioWaveform()
    {
        string? modelPath = FindModelPath("kokoro-82m-q8_0.gguf");
        string? voicePath = FindModelPath("kokoro-voice-af_heart.gguf");

        if (modelPath is null) return;

        using var model = KokoroModel.Load(modelPath, voicePath);
        using var pipeline = new KokoroPipeline(model);

        var request = new AudioGenerationRequest
        {
            Text = "Hello! This is a test of Kokoro 82M voice synthesis in OpenTail Stingray.",
            Voice = "af_heart",
            Speed = 1.0f
        };

        var result = pipeline.Generate(request);
        Assert.NotNull(result);
        Assert.Equal(24000, result.SampleRate);
        Assert.NotEmpty(result.Samples);
        Assert.True(result.Duration.TotalSeconds > 0.5, "Generated audio must have non-trivial duration");

        // Verify audio is non-silent
        float energy = 0f;
        foreach (var s in result.Samples) energy += s * s;
        Assert.True(energy > 0.1f, "Audio energy must be non-zero");
    }
}
