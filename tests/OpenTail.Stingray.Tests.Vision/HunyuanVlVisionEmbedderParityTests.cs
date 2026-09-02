using System.Runtime.InteropServices;
using System.Text.Json;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Parity of the C# <see cref="HunyuanVlVisionEncoder"/> against the numpy reference
/// (scripts/hunyuanvl_ref.py), which faithfully implements llama.cpp tools/mtmd/models/
/// hunyuanvl.cpp's build() plus clip.cpp's PROJECTOR_TYPE_HUNYUANVL bilinear position-embedding
/// resize branch. Same pattern as <see cref="Exaone4VisionEmbedderParityTests"/>. Covers all four
/// gaps closed in this pass: tensor renames, the real strided-Conv2D merger, the bilinear
/// position-embedding resize, and image_newline/image_begin/image_end sequence wrapping.
/// </summary>
public class HunyuanVlVisionEmbedderParityTests
{
    private static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
    }

    [Fact]
    public void Forward_MatchesNumpyReference()
    {
        var mmproj = VisionTestPaths.FindHunyuanVlMmproj();
        var fx = VisionTestPaths.FindFixtureDir("hunyuanvl");
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
        float[] golden = ReadF32(outPath);  // [nTok,1024]
        Assert.Equal(3 * H * W, input.Length);

        using var model = HunyuanVlVisionModel.Open(mmproj);
        var embedder = new HunyuanVlVisionEncoder(model);
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
