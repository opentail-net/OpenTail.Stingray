using System.Runtime.InteropServices;
using System.Text.Json;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Parity of the C# <see cref="Glm4VisionEncoder"/> against the numpy reference
/// (scripts/glm4v_ref.py), which faithfully implements llama.cpp tools/mtmd/models/glm4v.cpp's
/// build() plus the real GGML_ROPE_TYPE_VISION math traced through ggml-cpu/ops.cpp. Golden
/// fixtures (input_chw.f32, output.f32, meta.json) are produced by that script, generated at this
/// checkpoint's native 336x336/24x24-patch image size so its learned position_embd (sized for
/// exactly 576 positions) applies with no resize. Same pattern as
/// <see cref="PixtralVisionEmbedderParityTests"/>.
/// </summary>
public class Glm4VisionEmbedderParityTests
{
    private static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
    }

    [Fact]
    public void Forward_MatchesNumpyReference()
    {
        var mmproj = VisionTestPaths.FindGlm4Mmproj();
        var fx = VisionTestPaths.FindFixtureDir("glm4v");
        if (mmproj is null || fx is null) return;   // gated on model + generated golden
        var inPath = Path.Combine(fx, "input_chw.f32");
        var outPath = Path.Combine(fx, "output.f32");
        var metaPath = Path.Combine(fx, "meta.json");
        if (!File.Exists(inPath) || !File.Exists(outPath) || !File.Exists(metaPath)) return;

        using var doc = JsonDocument.Parse(File.ReadAllText(metaPath));
        int H = doc.RootElement.GetProperty("H").GetInt32();
        int W = doc.RootElement.GetProperty("W").GetInt32();
        int gx = doc.RootElement.GetProperty("gx").GetInt32();
        int gy = doc.RootElement.GetProperty("gy").GetInt32();
        int nTokExpected = doc.RootElement.GetProperty("n_tokens").GetInt32();

        float[] input = ReadF32(inPath);    // [3,H,W]
        float[] golden = ReadF32(outPath);  // [nTok,4096]
        Assert.Equal(3 * H * W, input.Length);

        using var model = Glm4VisionModel.Open(mmproj);
        var embedder = new Glm4VisionEncoder(model);
        var got = embedder.Forward(input, W, H, gx, gy, out int nTok);

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
