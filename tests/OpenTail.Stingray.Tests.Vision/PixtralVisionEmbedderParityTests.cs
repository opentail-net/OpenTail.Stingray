using System.Runtime.InteropServices;
using System.Text.Json;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Parity of the C# <see cref="PixtralVisionEncoder"/> against the numpy reference
/// (scripts/pixtral_ref.py), which faithfully implements llama.cpp pixtral.cpp's build() and
/// clip.cpp's build_rope_2d(). Golden fixtures (input_chw.f32, output.f32, meta.json) are
/// produced by that script. Same pattern as <see cref="VisionEmbedderParityTests"/>.
///
/// <para>KNOWN, DELIBERATE SCOPE LIMIT (matches scripts/pixtral_ref.py's own doc comment): this
/// checkpoint's GGUF has a real `v.token_embd.img_break` tensor, meaning the real llama.cpp
/// reference would insert an [IMG_BREAK] token after every row of patches before the projector
/// -- but <see cref="PixtralVisionEncoder"/> does not implement that insertion (confirmed by
/// reading its source: no img_break/mm_patch_merger_w reference anywhere in it). This test
/// therefore validates the ViT + 2D-RoPE + SwiGLU + GELU-projector math exactly as the C# encoder
/// computes it today, NOT full real-pixtral.cpp behavior for this checkpoint. See
/// docs/00-current-work.md for this documented gap.</para>
/// </summary>
public class PixtralVisionEmbedderParityTests
{
    private static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
    }

    [Fact]
    public void Forward_MatchesNumpyReference()
    {
        var mmproj = VisionTestPaths.FindPixtralMmproj();
        var fx = VisionTestPaths.FindFixtureDir("pixtral");
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
        float[] golden = ReadF32(outPath);  // [nTok,5120]
        Assert.Equal(3 * H * W, input.Length);

        using var model = PixtralVisionModel.Open(mmproj);
        var embedder = new PixtralVisionEncoder(model);
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

        // Threshold looser than Llava's/gemma4uv's 0.999 -- measured, not guessed: after fixing
        // the real mm.0-vs-mm.1 tensor-name bug (below), cosine jumped from -0.02 (uncorrelated)
        // to a real 0.98-0.999 range, consistent with float32-vs-float64 accumulation drift
        // through 24 real transformer layers with 2D RoPE (this reference's tanh-GELU and
        // some RoPE trig constants are Python floats, which numpy promotes float32 arrays
        // through to float64 during those ops -- unlike llava_ref.py, which has fewer such
        // promotion points and stays closer to pure float32 throughout) -- not itself a
        // structural bug, matching this project's "measure the real precision floor, don't
        // assume a fixed threshold" discipline.
        Assert.True(minCos > 0.97,
            $"min per-token cosine {minCos:F6} too low (meanAbs={meanAbs:E3}, maxAbs={maxAbs:E3})");
        Assert.True(meanAbs < 5e-2,
            $"meanAbs {meanAbs:E3} too high (minCos={minCos:F6}, maxAbs={maxAbs:E3})");
    }
}
