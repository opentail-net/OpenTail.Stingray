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
    public void QwenForcedAligner_RealModelFile_SafetensorsValidAndAlignsWithDtw()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var aligner = QwenAsrForcedAligner.Load(modelPath);
        Assert.NotNull(aligner);

        int numFrames = 60;
        int dim = 1024;
        float[] dummyAudioTokens = new float[numFrames * dim];
        for (int i = 0; i < dummyAudioTokens.Length; i++)
        {
            dummyAudioTokens[i] = 0.1f * MathF.Sin(i * 0.05f);
        }

        string referenceText = "Hello world from Qwen3 forced aligner test suite.";
        var segments = aligner.Align(
            referenceText,
            dummyAudioTokens,
            numFrames,
            dim,
            TimeSpan.Zero);

        Assert.NotNull(segments);
        Assert.Equal(8, segments.Count); // 8 words

        // Verify timestamps are strictly monotonically increasing
        for (int i = 0; i < segments.Count; i++)
        {
            Assert.True(segments[i].End > segments[i].Start, $"Segment {i} must have positive duration");
            if (i > 0)
            {
                Assert.True(segments[i].Start >= segments[i - 1].Start, $"Segment {i} start must be >= previous start");
            }
        }
    }
}
