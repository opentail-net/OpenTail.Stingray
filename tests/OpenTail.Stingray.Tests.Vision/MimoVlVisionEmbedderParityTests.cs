using System.Runtime.InteropServices;
using System.Text.Json;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Parity of the C# <see cref="MimoVlVisionEncoder"/> against the numpy reference
/// (scripts/mimovl_ref.py). Both real local checkpoints under this class's name are actually
/// `clip.projector_type=qwen2.5vl_merger` (confirmed via list-metadata), not the more elaborate
/// real row/col-banded-sink MIMOVL graph this class's own doc comment describes -- so this test
/// (like its reference script) validates the shared Qwen2.5-VL-family ViT/windowing/M-RoPE math
/// that the local checkpoints actually need, with plain MHA (no head_count_kv metadata present).
/// Same pattern as <see cref="Exaone4VisionEmbedderParityTests"/>.
/// </summary>
public class MimoVlVisionEmbedderParityTests
{
    private static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
    }

    [Fact]
    public void Forward_MatchesNumpyReference()
    {
        var mmproj = VisionTestPaths.FindMimoVlMmproj();
        var fx = VisionTestPaths.FindFixtureDir("mimovl");
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
        float[] golden = ReadF32(outPath);  // [nTok,4096]
        Assert.Equal(3 * H * W, input.Length);

        using var model = MimoVlVisionModel.Open(mmproj);
        var embedder = new MimoVlVisionEncoder(model);
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
