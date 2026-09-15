using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Isolation test (docs/081, 2026-09-14): checks whether the C# Wan port's own weight-loading
/// path returns the exact same raw bytes for blocks.0.ffn.0.weight/bias as reading the safetensors
/// file directly. Real checkpoint (read directly, bypassing this project's own loader): weight
/// norm=75.5376651465506, bias norm=0.819192863938423.
/// </summary>
public sealed class WanFfnWeightLoadCheckTest
{
    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\wan2.1\{fileName}",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;
        return null;
    }

    [Fact]
    public void FfnZeroWeightAndBias_MatchDirectFileRead()
    {
        string? path = FindModelPath("wan2.1-t2v-1.3b-dit.safetensors");
        if (path is null) return;

        using var loader = SafetensorsLoader.Open(path);
        var w = loader.ReadF32("blocks.0.ffn.0.weight");
        var b = loader.ReadF32("blocks.0.ffn.0.bias");

        double wSumSq = 0, bSumSq = 0;
        foreach (var v in w) wSumSq += (double)v * v;
        foreach (var v in b) bSumSq += (double)v * v;
        double wNorm = Math.Sqrt(wSumSq), bNorm = Math.Sqrt(bSumSq);
        Console.WriteLine($"[WanFfnWeightLoadCheck] w.Length={w.Length} wNorm={wNorm:F6} b.Length={b.Length} bNorm={bNorm:F6}");
        Console.WriteLine($"[WanFfnWeightLoadCheck] w[0..4]={w[0]:F8},{w[1]:F8},{w[2]:F8},{w[3]:F8},{w[4]:F8}");

        Assert.Equal(13762560, w.Length);
        Assert.Equal(8960, b.Length);
        Assert.True(Math.Abs(wNorm - 75.5376651465506) < 0.01, $"weight norm mismatch: {wNorm}");
        Assert.True(Math.Abs(bNorm - 0.819192863938423) < 0.001, $"bias norm mismatch: {bNorm}");
    }
}
