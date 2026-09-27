namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Kimi-VL and Youtu-VL vision encoders + projectors against the vendored llama.cpp reference:
/// <c>tools/llama.cpp/llama-mtmd-debug.exe -m &lt;text gguf&gt; --mmproj &lt;mmproj&gt; -p encode -n 224 --image cb</c>
/// (224x224 checkerboard, all three channels <c>(x+y)%2 ? 0 : 1</c>, fed straight to the encoder).
/// Golden values are the last projector stage's sum and row-0 corner, recorded 2026-09-27.
/// </summary>
public sealed class KimiYoutuVisionEmbedderParityTests
{
    private static float[] Checkerboard(int n)
    {
        var chw = new float[3 * n * n];
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                    chw[c * n * n + y * n + x] = ((x + y) % 2) != 0 ? 0f : 1f;
        return chw;
    }

    /// <summary>llama-mtmd-debug's "rainbow" pattern (tools/mtmd/debug), as CHW.</summary>
    private static float[] Rainbow(int n)
    {
        var chw = new float[3 * n * n];
        float cx = n / 2.0f, cy = n / 2.0f;
        float maxDist = MathF.Sqrt(cx * cx + cy * cy);
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                float dx = x - cx, dy = y - cy;
                float hue = MathF.Atan2(dy, dx) / (2.0f * 3.14159265f);
                if (hue < 0) hue += 1.0f;
                float sat = MathF.Sqrt(dx * dx + dy * dy) / maxDist;
                if (sat > 1.0f) sat = 1.0f;
                float h6 = hue * 6.0f;
                int i6 = (int)h6;
                float f = h6 - i6, p = 1.0f - sat, q = 1.0f - sat * f, t = 1.0f - sat * (1.0f - f);
                (float r, float g, float b) = (i6 % 6) switch
                {
                    0 => (1f, t, p), 1 => (q, 1f, p), 2 => (p, 1f, t),
                    3 => (p, q, 1f), 4 => (t, p, 1f), _ => (1f, p, q),
                };
                chw[y * n + x] = r; chw[n * n + y * n + x] = g; chw[2 * n * n + y * n + x] = b;
            }
        return chw;
    }

    private static void Report(string name, float[] tokens, int count, int dim)
    {
        double sum = 0;
        foreach (float v in tokens) sum += v;
        Console.WriteLine($"[{name}] count={count} dim={tokens.Length / Math.Max(count, 1)} sum={sum:F4} row0=[{tokens[0]:F4}, {tokens[1]:F4}, {tokens[2]:F4}] expectDim={dim}");
    }

    private static void AssertMatches(float[] tokens, double refSum, double sumTol, float[] refRow0, float tol)
    {
        double sum = 0;
        foreach (float v in tokens) sum += v;
        Assert.InRange(sum, refSum - sumTol, refSum + sumTol);
        for (int i = 0; i < refRow0.Length; i++)
            Assert.InRange(tokens[i], refRow0[i] - tol, refRow0[i] + tol);
    }

    [Fact]
    public void Kimi_Checkerboard224_MatchesLlamaMtmdDebug()
    {
        string? path = FindModel("mmproj-kimivl-q8_0.gguf");
        Assert.SkipWhen(path is null, "mmproj-kimivl-q8_0.gguf not present");
        using var model = KimiVisionModel.Open(path!);
        float[] tokens = new KimiVisionEncoder(model).Forward(Checkerboard(224), 224, 224, 16, 16, out int count);
        Report("kimi cb224", tokens, count, 2048);

        // llama-mtmd-debug (-m kimi-vl-a3b-thinking-Q2_K.gguf --mmproj mmproj-kimivl-q8_0.gguf -n 224 --image cb)
        // proj_out: 64 x 2048, sum -103.105698, row 0 [0.0767, -0.0794, -0.0895]. Measured: -103.04, [0.0760, -0.0786, -0.0915].
        Assert.Equal(64, count);
        AssertMatches(tokens, -103.106, 2.0, [0.0767f, -0.0794f, -0.0895f], 0.01f);
    }

    [Fact]
    public void Kimi_Rainbow448_MatchesLlamaMtmdDebug()
    {
        string? path = FindModel("mmproj-kimivl-q8_0.gguf");
        Assert.SkipWhen(path is null, "mmproj-kimivl-q8_0.gguf not present");
        using var model = KimiVisionModel.Open(path!);
        float[] tokens = new KimiVisionEncoder(model).Forward(Rainbow(448), 448, 448, 32, 32, out int count);
        Report("kimi rainbow448", tokens, count, 2048);

        // Same reference with -n 448 --image rainbow: 256 x 2048, sum -707.471985, row 0 [0.2910, -0.1047, -0.1101].
        // Exercises the 2D RoPE and the 64x64 -> 32x32 antialiased position-table resize. Measured: -708.22, [0.2923, -0.0998, -0.1162].
        Assert.Equal(256, count);
        AssertMatches(tokens, -707.472, 7.0, [0.2910f, -0.1047f, -0.1101f], 0.015f);
    }

    [Fact]
    public void Youtu_Rainbow448_MatchesLlamaMtmdDebug()
    {
        string? path = FindModel("mmproj-youtu-vl-4b-BF16.gguf");
        Assert.SkipWhen(path is null, "mmproj-youtu-vl-4b-BF16.gguf not present");
        using var model = YoutuVlVisionModel.Open(path!);
        float[] tokens = new YoutuVlVisionEncoder(model).Forward(Rainbow(448), 448, 448, 28, 28, out int count);
        Report("youtu rainbow448", tokens, count, 2560);

        // llama-mtmd-debug (-m youtu-vl-4b-Q8_0.gguf --mmproj mmproj-youtu-vl-4b-BF16.gguf -n 448 --image rainbow):
        // 196 x 2560, sum 2104.656006, row 0 [0.4770, 1.0259, 0.0240]. The 448 image has 2x2 windows of
        // 16x16 patches, so this exercises window attention (layers 7/15/23/26 full, the rest windowed).
        // Measured: 2092.60, [0.4780, 1.0342, 0.0253]. A checkerboard is useless here: every 16x16 patch
        // is identical, and llama.cpp's own Q8_0 and BF16 mmproj disagree by 3% on it.
        Assert.Equal(196, count);
        AssertMatches(tokens, 2104.656, 25.0, [0.4770f, 1.0259f, 0.0240f], 0.02f);
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
