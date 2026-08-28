using System;
using System.IO;
using System.Linq;
using OpenTail.Stingray.Audio.FishSpeech;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMP bisection harness for the Fish Speech NaN-crash investigation (docs/audio-review-
/// progress.md's Fish Speech entries): confirmed the real prefill (36 layers, real ~30-token
/// prompt) produces 100% NaN logits and a 100% NaN last-layer hidden state. This isolates WHERE
/// by varying layer count (like the QwenTTS bisection) and prompt length, entirely in C# --
/// unlike QwenTTS's "wrong but finite" failure, a NaN/not-NaN signal needs no Python reference to
/// interpret. TODO revert/remove once the bug is found.
/// </summary>
public sealed class FishSpeechBisectTests : HeavyTestBase
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

    private static string? FindRepoDir(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Theory]
    [InlineData(16)]
    [InlineData(20)]
    [InlineData(24)]
    [InlineData(28)]
    [InlineData(32)]
    [InlineData(34)]
    [InlineData(35)]
    [InlineData(36)]
    public void Bisect_PrefillNaN_ByLayerCount(int numLayers)
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        string? tokDir = FindRepoDir("examples/s2.cpp");
        Assert.SkipUnless(modelPath != null && tokDir != null, "S2 Pro GGUF or examples/s2.cpp not found");

        using var pipeline = new FishSpeechPipeline(modelPath!, tokDir!, numLayers: numLayers);
        var prompt = pipeline.BuildPrompt("Hello, this is a test of speech synthesis.");
        var logits = pipeline.PrefillForBisection(prompt);

        int nan = logits.Count(float.IsNaN);
        Console.WriteLine($"numLayers={numLayers}: logits.Length={logits.Length} nan={nan} " +
            $"({100.0 * nan / logits.Length:F1}%)");
    }

    [Theory]
    [InlineData(30)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(34)]
    [InlineData(35)]
    public void Bisect_HiddenMagnitudeGrowth(int tapLayer)
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        string? tokDir = FindRepoDir("examples/s2.cpp");
        Assert.SkipUnless(modelPath != null && tokDir != null, "S2 Pro GGUF or examples/s2.cpp not found");

        using var pipeline = new FishSpeechPipeline(modelPath!, tokDir!, numLayers: 36);
        var prompt = pipeline.BuildPrompt("Hello, this is a test of speech synthesis.");
        var hidden = pipeline.PrefillHiddenTapForBisection(prompt, tapLayer);

        int nan = hidden.Count(float.IsNaN);
        int inf = hidden.Count(float.IsInfinity);
        float maxAbs = hidden.Where(float.IsFinite).Select(MathF.Abs).DefaultIfEmpty(0f).Max();
        Console.WriteLine($"tapLayer={tapLayer}: n={hidden.Length} nan={nan} inf={inf} maxAbsFinite={maxAbs:E3}");
    }

    [Fact]
    public void CheckLayer34WeightsForCorruption()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        Assert.SkipUnless(modelPath != null, "S2 Pro GGUF not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(modelPath!);
        foreach (var name in new[]
        {
            "layers.34.feed_forward.w2.weight",
            "layers.34.feed_forward.w1.weight",
            "layers.34.feed_forward.w3.weight",
            "layers.34.attention.wqkv.weight",
            "layers.34.attention.wo.weight",
            "layers.34.attention_norm.weight",
            "layers.34.ffn_norm.weight",
            "layers.34.attention.q_norm.weight",
            "layers.34.attention.k_norm.weight",
        })
        {
            var info = model.FindTensor(name);
            if (info is null) { Console.WriteLine($"{name}: MISSING"); continue; }
            var dst = new float[info.Value.ElementCount];
            var bytes = model.GetTensorData(info.Value);
            OpenTail.Stingray.Cpu.Dequantize.ToFloat32(bytes, dst, info.Value.DType, info.Value.ElementCount);
            int nan = dst.Count(float.IsNaN);
            int inf = dst.Count(float.IsInfinity);
            float min = dst.Length > 0 ? dst.Min() : 0;
            float max = dst.Length > 0 ? dst.Max() : 0;
            Console.WriteLine($"{name} [{info.Value.DType}]: n={dst.Length} nan={nan} inf={inf} min={min:E3} max={max:E3}");
        }
    }
}
