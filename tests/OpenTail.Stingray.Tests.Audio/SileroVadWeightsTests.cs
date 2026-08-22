using System;
using System.IO;
using OpenTail.Stingray.Audio.Vad;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="SileroVad"/>'s rewritten 16kHz forward pass
/// (see docs/audio-review-progress.md's Silero VAD section) -- runs the real
/// `models/silero_vad.onnx` via onnxruntime (`scratch-llamacpp-ref/silero_golden_input.txt` +
/// the golden probability captured alongside it) and checks this C# port's output against it,
/// not just shape/finite checks. The golden input is a seeded-random (numpy `default_rng(42)`)
/// synthetic 512-sample frame -- content doesn't matter for this check, only that the SAME
/// exact samples feed both the real ONNX graph and this C# port with a fresh (all-zero) LSTM
/// state on both sides.
/// </summary>
public sealed class SileroVadWeightsTests : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void ProcessFrame_RealWeights_MatchesOnnxGoldenProbability()
    {
        string? onnxPath = FindRepoFile("models/silero_vad.onnx");
        Assert.SkipUnless(onnxPath != null, "models/silero_vad.onnx not found");
        string? goldenPath = FindRepoFile("scratch-llamacpp-ref/silero_golden_input.txt");
        Assert.SkipUnless(goldenPath != null, "scratch-llamacpp-ref/silero_golden_input.txt not found (re-run the golden dump script)");

        var parts = File.ReadAllText(goldenPath!).Trim().Split(',');
        Assert.Equal(512, parts.Length);
        var frame = new float[512];
        for (int i = 0; i < 512; i++) frame[i] = float.Parse(parts[i]);

        using var vad = SileroVad.Load(onnxPath!);
        float prob = vad.ProcessFrame(frame);

        // Captured via a single onnxruntime session.run() call (ORT_DISABLE_ALL, matching this
        // doc's established golden-dump discipline) against the exact same 512-sample frame,
        // fresh zero LSTM state on both sides: 0.025505661964416504.
        const float goldenProb = 0.025505661964416504f;
        Assert.True(MathF.Abs(prob - goldenProb) < 0.0001f,
            $"prob={prob} vs golden={goldenProb}, diff={MathF.Abs(prob - goldenProb)}");
    }
}
