using System;
using System.IO;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.QwenASR;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class QwenForcedAlignerRealWeightsTests
{
    private const string ModelFileName = "qwen3-forcedaligner-0.6b.safetensors";

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
    public void QwenForcedAligner_RealModelFile_SafetensorsValidAndAligns()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var st = SafetensorsLoader.Open(modelPath);
        Assert.NotNull(st);
        Assert.True(st.TensorCount > 0, "ForcedAligner safetensors must contain tensors");

        using var aligner = new QwenAsrForcedAligner();
        int numFrames = 50;
        int dim = 1024;
        float[] dummyAudioTokens = new float[numFrames * dim];
        for (int i = 0; i < dummyAudioTokens.Length; i++)
        {
            dummyAudioTokens[i] = 0.1f * MathF.Sin(i * 0.05f);
        }

        var segments = aligner.Align(
            "Hello world from Qwen3 forced aligner test suite.",
            dummyAudioTokens,
            numFrames,
            dim,
            TimeSpan.Zero);

        Assert.NotNull(segments);
        Assert.True(segments.Count > 0);
        Assert.Equal("Hello", segments[0].Text);
    }
}
