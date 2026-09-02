using System.Runtime.InteropServices;
using System.Text.Json;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Parity of the C# <see cref="Exaone4VisionEncoder"/> against the numpy reference
/// (scripts/exaone4_ref.py), which faithfully implements llama.cpp tools/mtmd/models/exaone4_5.cpp's
/// build() -- shares its windowed-attention mask construction and 4-section M-RoPE math with
/// Qwen2.5-VL (see scripts/qwen25vl_ref.py's doc comment). Same pattern as
/// <see cref="Qwen25VlVisionEmbedderParityTests"/>.
/// </summary>
public class Exaone4VisionEmbedderParityTests
{
    private static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
    }

    [Fact]
    public void Forward_MatchesNumpyReference()
    {
        var mmproj = VisionTestPaths.FindExaone4Mmproj();
        var fx = VisionTestPaths.FindFixtureDir("exaone4");
        if (mmproj is null || fx is null) return;
        var inPath = Path.Combine(fx, "input_chw.f32");
        var outPath = Path.Combine(fx, "output.f32");
        var metaPath = Path.Combine(fx, "meta.json");
        if (!File.Exists(inPath) || !File.Exists(outPath) || !File.Exists(metaPath)) return;

        using var doc = JsonDocument.Parse(File.ReadAllText(metaPath));
        int H = doc.RootElement.GetProperty("H").GetInt32();
        int W = doc.RootElement.GetProperty("W").GetInt32();
        int patch = doc.RootElement.GetProperty("patch").GetInt32();
        int nTokExpected = doc.RootElement.GetProperty("n_tokens").GetInt32();
        int patchesX = W / patch;
        int patchesY = H / patch;

        float[] input = ReadF32(inPath);    // [3,H,W]
        float[] golden = ReadF32(outPath);  // [nTok,5120]
        Assert.Equal(3 * H * W, input.Length);

        using var model = Exaone4VisionModel.Open(mmproj);
        var embedder = new Exaone4VisionEncoder(model);
        var got = embedder.Forward(input, W, H, patchesX, patchesY, out int nTok);

        Assert.Equal(nTokExpected, nTok);
        Assert.Equal(golden.Length, got.Length);

        int embd = embedder.ProjectionDim;
        double maxAbs = 0, sumAbs = 0;
        double minCos = 1.0;
        for (int t = 0; t < nTok; t++)
        {
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < embd; i++)
            {
                int k = t * embd + i;
                double a = got[k], b = golden[k];
                double d = Math.Abs(a - b);
                if (d > maxAbs) maxAbs = d;
                sumAbs += d;
                dot += a * b; na += a * a; nb += b * b;
            }
            double cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
            if (cos < minCos) minCos = cos;
        }
        double meanAbs = sumAbs / got.Length;

        Assert.True(minCos > 0.97,
            $"min per-token cosine {minCos:F6} too low (meanAbs={meanAbs:E3}, maxAbs={maxAbs:E3})");
        Assert.True(meanAbs < 5e-2,
            $"meanAbs {meanAbs:E3} too high (minCos={minCos:F6}, maxAbs={maxAbs:E3})");
    }
}
