using System;
using System.IO;
using System.Reflection;
using OpenTail.Stingray.Audio.FishSpeech;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio.Fast;

/// <summary>SCRATCH/throwaway diagnostic. Delete after use.</summary>
public sealed class FishSpeechScratchTests2 : HeavyTestBase
{
    [Fact]
    public void Layer1_GivenRealLayer0Output_ComparedToOracle()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        Assert.SkipUnless(modelPath != null, "models/s2-pro-q4_k_m.gguf not found");

        using var w = new FishSpeechWeights(modelPath!);

        // Real oracle's layer-0 output at position 1 (from fastar_layer0_full_output.npy, dumped as text below)
        // pos0 and pos1 rows, DIM=2560 each -- pasted via a companion npy->txt dump.
        string? l0Path = FindRepoFile("scratch-llamacpp-ref/fastar_layer0_full_output.txt");
        Assert.SkipUnless(l0Path != null, "layer0 full output dump not found");

        var lines = File.ReadAllText(l0Path!).Trim().Split('\n');
        var row0 = Array.ConvertAll(lines[0].Split(','), float.Parse);
        var row1 = Array.ConvertAll(lines[1].Split(','), float.Parse);

        var x = new float[2][];
        x[0] = row0;
        x[1] = row1;

        var layerMethod = typeof(FishSpeechFastAr).GetMethod("Layer", BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(layerMethod);
        var result = (float[][])layerMethod!.Invoke(null, [x, w.FastLayers[1], w])!;

        File.WriteAllText(Path.Combine(Path.GetTempPath(), "fishspeech_layer1_given_real_input.txt"),
            $"pos1_first5=[{string.Join(",", result[1][..5])}]\n");
    }

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
}
