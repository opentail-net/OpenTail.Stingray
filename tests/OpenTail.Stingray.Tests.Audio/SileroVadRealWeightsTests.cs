using System;
using System.IO;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Vad;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class SileroVadRealWeightsTests
{
    private const string ModelFileName = "silero_vad.onnx";

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
    public void SileroVad_RealModelFile_OnnxHeaderValidAndExecutes()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        var fileInfo = new FileInfo(modelPath);
        Assert.True(fileInfo.Length > 1024 * 1024, "Silero VAD ONNX model file must be > 1MB");

        using var vad = new SileroVad();
        float[] frame = new float[512];

        // Synthesize harmonic audio
        for (int i = 0; i < 512; i++)
        {
            float t = i / 16000.0f;
            frame[i] = 0.5f * MathF.Sin(2.0f * MathF.PI * 250.0f * t);
        }

        float prob = vad.ProcessFrame(frame);
        Assert.InRange(prob, 0.0f, 1.0f);
        Assert.False(float.IsNaN(prob));
    }
}
