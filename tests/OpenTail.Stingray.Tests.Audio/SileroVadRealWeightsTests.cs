using System;
using System.IO;
using OpenTail.Stingray.Audio.Vad;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class SileroVadRealWeightsTests
{
    private const string GgufFileName = "silero_vad.gguf";

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
    public void SileroVad_RealGgufModel_LoadsTensorsAndMetadata()
    {
        string? modelPath = FindModelPath(GgufFileName);
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Silero VAD GGUF must have tensors");
        Assert.True(model.Metadata.Count > 0, "Silero VAD GGUF must have metadata");
    }

    [Fact]
    public void SileroVad_RealGgufModel_DetectsSpeechAndSilence()
    {
        string? modelPath = FindModelPath(GgufFileName);
        if (modelPath is null) return;

        using var vad = SileroVad.Load(modelPath);
        Assert.NotNull(vad);

        // 1. Digital silence frame (512 zeros) -> should have low probability
        var silenceFrame = new float[512];
        float silenceProb = vad.ProcessFrame(silenceFrame);
        Assert.True(silenceProb < 0.2f, $"Silence frame probability ({silenceProb}) must be low");

        // 2. Harmonic speech tone frame (300Hz fundamental + 600Hz + 900Hz harmonics)
        var speechFrame = new float[512];
        for (int i = 0; i < speechFrame.Length; i++)
        {
            float t = (float)i / 16000.0f;
            speechFrame[i] = 0.5f * MathF.Sin(2.0f * MathF.PI * 300.0f * t)
                           + 0.3f * MathF.Sin(2.0f * MathF.PI * 600.0f * t)
                           + 0.2f * MathF.Sin(2.0f * MathF.PI * 900.0f * t);
        }

        // Process a few frames to ramp up state
        float speechProb = 0f;
        for (int step = 0; step < 4; step++)
        {
            speechProb = vad.ProcessFrame(speechFrame);
        }
        Assert.True(speechProb >= 0.0f && speechProb <= 1.0f, $"Speech probability ({speechProb}) must be within [0, 1]");
    }
}
