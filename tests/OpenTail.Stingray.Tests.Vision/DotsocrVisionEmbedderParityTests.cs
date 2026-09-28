namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// dots.ocr vision encoder + projector against the vendored llama.cpp reference. The golden values
/// are <c>tools/llama.cpp/llama-mtmd-debug.exe -m dots.ocr-Q8_0.gguf --mmproj mmproj-dots.ocr-Q8_0.gguf
/// -p encode -n 224 --image cb</c>: a 224x224 checkerboard fed straight to the encoder (all three
/// channels <c>(x+y)%2 ? 0 : 1</c>), 64 output tokens; the tool prints each stage's corner values
/// and sum. Pins the 2026-09-26 fixes (fused attn_qkv, SwiGLU gate, mm.post_norm, vision M-RoPE
/// section reset, erf GELU): before them the model ignored the image (1-token stop).
/// </summary>
public sealed class DotsocrVisionEmbedderParityTests : HeavyTestBase
{
    [Fact]
    public void Checkerboard224_MatchesLlamaMtmdDebug()
    {
        string? path = FindModel("mmproj-dots.ocr-Q8_0.gguf");
        Assert.SkipWhen(path is null, "mmproj-dots.ocr-Q8_0.gguf not present");
        using var model = DotsOcrVisionModel.Open(path!);
        var enc = new DotsOcrVisionEncoder(model);
        const int n = 224;
        var chw = new float[3 * n * n];
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                    chw[c * n * n + y * n + x] = ((x + y) % 2) != 0 ? 0f : 1f;

        float[] tokens = enc.Forward(chw, n, n, n / 14, n / 14, out int count);

        Assert.Equal(64, count);
        double sum = 0;
        foreach (float v in tokens) sum += v;
        // llama-mtmd-debug after_projector: sum = 15843.735352, row 0 = [-2.5777, -1.0951, 7.7399, ...].
        // Measured here: sum 15843.43, row 0 [-2.5900, -1.0668, 7.7198] (Q8_0 activation rounding).
        // 2026-09-27: with the batched GEMM linears (ggml-rounded activations) + GEMM attention the checkerboard gives
        // 15734.98, [-2.6632, ...]. Every patch is identical here, so only rounding separates engines; see the
        // rainbow test below for the non-degenerate check and the layer-by-layer drift that sets these tolerances.
        Assert.InRange(sum, 15843.735 * 0.99, 15843.735 * 1.01);
        Assert.InRange(tokens[0], -2.5777 - 0.1, -2.5777 + 0.1);
        Assert.InRange(tokens[1], -1.0951 - 0.1, -1.0951 + 0.1);
        Assert.InRange(tokens[2], 7.7399 - 0.1, 7.7399 + 0.1);
    }

    [Fact]
    public void Rainbow448_MatchesLlamaMtmdDebug()
    {
        string? path = FindModel("mmproj-dots.ocr-Q8_0.gguf");
        Assert.SkipWhen(path is null, "mmproj-dots.ocr-Q8_0.gguf not present");
        using var model = DotsOcrVisionModel.Open(path!);
        var enc = new DotsOcrVisionEncoder(model);
        const int n = 448;
        var chw = new float[3 * n * n];
        float c0 = n / 2.0f, maxDist = MathF.Sqrt(2 * c0 * c0);
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                float dx = x - c0, dy = y - c0;
                float hue = MathF.Atan2(dy, dx) / (2.0f * 3.14159265f);
                if (hue < 0) hue += 1.0f;
                float sat = MathF.Min(MathF.Sqrt(dx * dx + dy * dy) / maxDist, 1f);
                float h6 = hue * 6.0f; int i6 = (int)h6; float f = h6 - i6;
                float p = 1 - sat, q = 1 - sat * f, t = 1 - sat * (1 - f);
                (float r, float g, float b) = (i6 % 6) switch { 0 => (1f, t, p), 1 => (q, 1f, p), 2 => (p, 1f, t), 3 => (p, q, 1f), 4 => (t, p, 1f), _ => (1f, p, q) };
                chw[y * n + x] = r; chw[n * n + y * n + x] = g; chw[2 * n * n + y * n + x] = b;
            }
        float[] tokens = enc.Forward(chw, n, n, n / 14, n / 14, out int count);
        double sum = 0;
        foreach (float v in tokens) sum += v;
        Console.WriteLine($"[dots rainbow448] count={count} sum={sum:F4} row0=[{tokens[0]:F4}, {tokens[1]:F4}, {tokens[2]:F4}]");
        Assert.Equal(256, count);
        // llama-mtmd-debug -n 448 --image rainbow: after_projector 256 x 1536, sum 16959.291016,
        // row 0 [-0.9587, 0.0284, -5.1315]. Unlike the checkerboard (every 14x14 patch identical, so attention
        // is exactly uniform and only rounding noise separates engines), this input is not degenerate.
        // Layer outputs match llama.cpp closely through layer 22, then drift as activations grow (layer 40 sums
        // -134408 vs -138131; row-0 values reach ~130 at layer 41), in every engine variant (per-token Q8
        // activations: 16978.4 [-0.996, -0.148, -5.121]; GEMM: 16896.3 [-1.136, 0.107, -5.250]). No single layer
        // diverges, so the tolerance reflects that precision limit. End to end, dots.ocr reads the same text as
        // llama-server (docs/STATUS.md dots.ocr row).
        Assert.InRange(sum, 16959.291 * 0.99, 16959.291 * 1.01);
        Assert.InRange(tokens[0], -0.9587 - 0.25, -0.9587 + 0.25);
        Assert.InRange(tokens[1], 0.0284 - 0.25, 0.0284 + 0.25);
        Assert.InRange(tokens[2], -5.1315 - 0.25, -5.1315 + 0.25);
    }

    private static string? FindModel(string file)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            foreach (var p in new[] { Path.Combine(dir.FullName, "models", file), Path.Combine(dir.FullName, "models", "_models", file) })
                if (File.Exists(p)) return p;
            dir = dir.Parent;
        }
        return null;
    }
}
