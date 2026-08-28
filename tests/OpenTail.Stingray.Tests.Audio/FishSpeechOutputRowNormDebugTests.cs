using System;
using System.IO;
using System.Linq;
using OpenTail.Stingray.Audio.FishSpeech;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMP debug harness (docs/audio-review-progress.md's Fish Speech "goat"/codebook-1 investigation,
/// 2026-08-28): checks whether token 929's row in the fast-AR output projection (the value that
/// dominates our own codebook-1 output 61% of the time, vs the reference's ~1-2% expected for a
/// healthy 1024-way near-uniform-ish distribution) is a global outlier in the weight matrix
/// itself (e.g. an anomalously large norm/bias that would win most near-ties regardless of the
/// actual hidden-state input) -- as opposed to a property of our specific hidden-state
/// trajectory. TODO remove once resolved.
/// </summary>
public sealed class FishSpeechOutputRowNormDebugTests : HeavyTestBase
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
    public void CheckToken929RowNorm()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        Assert.SkipUnless(modelPath != null, "S2 Pro GGUF not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(modelPath!);
        var info = model.FindTensor("fast_output.weight");
        Assert.NotNull(info);

        int codebookSize = 4096, embDim = 2560;
        var dst = new float[info!.Value.ElementCount];
        var bytes = model.GetTensorData(info.Value);
        OpenTail.Stingray.Cpu.Dequantize.ToFloat32(bytes, dst, info.Value.DType, info.Value.ElementCount);

        // dst is [codebookSize, embDim] row-major (row = output token, matching GGUF's real
        // [in=embDim, out=codebookSize] shape read as rows-of-outputs after Dequantize.ToFloat32,
        // same convention as every other GetTensor use in this codebase).
        var norms = new double[codebookSize];
        for (int r = 0; r < codebookSize; r++)
        {
            double sumSq = 0;
            for (int c = 0; c < embDim; c++) { float v = dst[r * embDim + c]; sumSq += (double)v * v; }
            norms[r] = Math.Sqrt(sumSq);
        }

        double mean = norms.Average();
        double std = Math.Sqrt(norms.Select(n => (n - mean) * (n - mean)).Average());
        var sorted = norms.OrderByDescending(n => n).ToArray();
        int rank929 = Array.IndexOf(norms.Select((n, i) => (n, i)).OrderByDescending(x => x.n).Select(x => x.i).ToArray(), 929);

        Console.WriteLine($"mean={mean:F4} std={std:F4} min={norms.Min():F4} max={norms.Max():F4}");
        Console.WriteLine($"row929 norm={norms[929]:F4} zscore={(norms[929] - mean) / std:F3} rank(1=largest)={rank929 + 1}/{codebookSize}");
        Console.WriteLine("top-5 largest-norm rows: " + string.Join(",", sorted.Take(5).Select(n => n.ToString("F2"))));
        // Also compare against a few arbitrary other tokens for context.
        foreach (int t in new[] { 0, 100, 500, 601, 752, 1000, 2000, 4000 })
            Console.WriteLine($"row{t} norm={norms[t]:F4}");
    }
}
