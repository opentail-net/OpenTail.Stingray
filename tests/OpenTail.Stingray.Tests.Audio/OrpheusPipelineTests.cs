using System;
using System.IO;
using OpenTail.Stingray.Audio.Orpheus;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio.Fast;

/// <summary>
/// End-to-end structural test for <see cref="OrpheusPipeline"/> -- real talker weights, real
/// SNAC weights, real prompt/detokenization. Not a golden-cosine test (no independent oracle
/// runs the FULL talker-&gt;codec pipeline for comparison; the talker LM and SNAC decoder are
/// each independently golden-verified elsewhere -- this test only confirms the full pipeline
/// runs end-to-end without crashing and produces a structurally plausible PCM signal).
/// </summary>
public sealed class OrpheusPipelineTests : HeavyTestBase
{
    private static string? FindModelPath(string fileName)
    {
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
    public void Synthesize_RealWeights_ProducesPlausiblePcm()
    {
        string? talkerPath = FindModelPath("orpheus-3b-0.1-ft.Q4_K_M.gguf");
        string? snacPath = FindModelPath("snac-24khz.gguf");
        Assert.SkipUnless(talkerPath != null && snacPath != null, "Orpheus/SNAC GGUF files not found");

        using var pipeline = new OrpheusPipeline(talkerPath!, snacPath!);
        var pcm = pipeline.Synthesize("Hello, this is a test.", voice: "tara", maxTokens: 140);

        Assert.NotEmpty(pcm);
        Assert.True(pcm.Length % 512 == 0, $"PCM length {pcm.Length} should be a multiple of the SNAC hop length (512)");

        float max = 0f, sumSq = 0f;
        foreach (var s in pcm)
        {
            Assert.True(s >= -1.0001f && s <= 1.0001f, $"sample {s} outside valid post-Tanh range");
            max = MathF.Max(max, MathF.Abs(s));
            sumSq += s * s;
        }
        float rms = MathF.Sqrt(sumSq / pcm.Length);
        Assert.True(max > 1e-4f, $"PCM is near-silent (max abs {max}) -- likely a wiring bug, not real generated audio");

        var outPath = Path.Combine(Path.GetTempPath(), "orpheus_test_output.wav");
        WriteWav(outPath, pcm, 24000);
    }

    private static void WriteWav(string path, float[] pcm, int sampleRate)
    {
        using var fs = new FileStream(path, FileMode.Create);
        using var bw = new BinaryWriter(fs);
        int dataSize = pcm.Length * 2;
        bw.Write("RIFF"u8);
        bw.Write(36 + dataSize);
        bw.Write("WAVE"u8);
        bw.Write("fmt "u8);
        bw.Write(16);
        bw.Write((short)1);
        bw.Write((short)1);
        bw.Write(sampleRate);
        bw.Write(sampleRate * 2);
        bw.Write((short)2);
        bw.Write((short)16);
        bw.Write("data"u8);
        bw.Write(dataSize);
        foreach (var s in pcm)
            bw.Write((short)Math.Clamp(s * 32767f, short.MinValue, short.MaxValue));
    }
}
