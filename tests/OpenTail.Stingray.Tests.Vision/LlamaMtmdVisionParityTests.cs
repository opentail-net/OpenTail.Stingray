namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Vision encoders + projectors (Kimi-VL, Youtu-VL, PaddleOCR-VL, Qwen2.5-VL) against the vendored llama.cpp reference:
/// <c>tools/llama.cpp/llama-mtmd-debug.exe -m &lt;text gguf&gt; --mmproj &lt;mmproj&gt; -p encode -n 224 --image cb</c>
/// (224x224 checkerboard, all three channels <c>(x+y)%2 ? 0 : 1</c>, fed straight to the encoder).
/// Golden values are the last projector stage's sum and row-0 corner, recorded 2026-09-27.
/// </summary>
public sealed class LlamaMtmdVisionParityTests
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

    [Fact]
    public void PaddleOcr_Rainbow448_MatchesLlamaMtmdDebug()
    {
        string? path = FindModel("mmproj-paddleocr-vl-1.6.gguf");
        Assert.SkipWhen(path is null, "mmproj-paddleocr-vl-1.6.gguf not present");
        using var model = PaddleOcrVisionModel.Open(path!);
        float[] tokens = new PaddleOcrVisionEncoder(model).Forward(Rainbow(448), 448, 448, 32, 32, out int count);
        Report("paddle rainbow448", tokens, count, 1024);

        // llama-mtmd-debug (-m paddleocr-vl-1.6.gguf --mmproj mmproj-paddleocr-vl-1.6.gguf -n 448 --image rainbow):
        // mlp_out 256 x 1024, sum -4653.626465, row 0 [2.1425, 0.9280, 1.8393].
        Assert.Equal(256, count);
        AssertMatches(tokens, -4653.626, 47.0, [2.1425f, 0.9280f, 1.8393f], 0.03f);
    }

    [Fact]
    public void Qwen25Vl_Rainbow448_MatchesLlamaMtmdDebug()
    {
        string? path = FindModel("mmproj-qwen2.5-vl-7b-f16.gguf");
        Assert.SkipWhen(path is null, "mmproj-qwen2.5-vl-7b-f16.gguf not present");
        using var model = QwenVlVisionModel.Open(path!);
        float[] tokens = new QwenVlVisionEncoder(model).Forward(Rainbow(448), 448, 448, out int count);
        Report("qwen25vl rainbow448", tokens, count, 3584);

        // llama-mtmd-debug (-m Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf --mmproj mmproj-qwen2.5-vl-7b-f16.gguf -n 448 --image rainbow):
        // 256 x 3584, sum 18550.607422, row 0 [0.0276, -0.3893, -1.7246]. Measured: 18591.93, [0.0619, -0.4211, -1.7408].
        // Layer 0 matches to 4 decimals (sum -6771.23 vs -6771.27); the residual is 32-layer drift (layer 31
        // sum within 0.005%). Before the 2026-09-27 section-reset RoPE fix: 16505.74, [-0.3269, 0.1654, -1.4568].
        Assert.Equal(256, count);
        AssertMatches(tokens, 18550.607, 190.0, [0.0276f, -0.3893f, -1.7246f], 0.05f);
    }

    [Fact]
    public void MimoVl_Rainbow448_MatchesLlamaMtmdDebug()
    {
        string? path = FindModel("mmproj-mimovl-7b-q8_0.gguf");
        Assert.SkipWhen(path is null, "mmproj-mimovl-7b-q8_0.gguf not present");
        using var model = MimoVlVisionModel.Open(path!);
        float[] tokens = new MimoVlVisionEncoder(model).Forward(Rainbow(448), 448, 448, 32, 32, out int count);
        Report("mimovl rainbow448", tokens, count, 4096);

        // llama-mtmd-debug (-m mimo-vl-7b-sft-Q2_K.gguf --mmproj mmproj-mimovl-7b-q8_0.gguf -n 448 --image rainbow):
        // 256 x 4096, sum -8317.800781, row 0 [0.9284, 0.8123, 1.3568].
        Assert.Equal(256, count);
        AssertMatches(tokens, -8317.801, 85.0, [0.9284f, 0.8123f, 1.3568f], 0.05f);
    }

    [Fact]
    public void Exaone45_Rainbow448_MatchesLlamaMtmdDebug()
    {
        string? path = FindModel("mmproj-exaone-4.5-q8_0.gguf");
        Assert.SkipWhen(path is null, "mmproj-exaone-4.5-q8_0.gguf not present");
        using var model = Exaone4VisionModel.Open(path!);
        float[] tokens = new Exaone4VisionEncoder(model).Forward(Rainbow(448), 448, 448, 32, 32, out int count);
        Report("exaone45 rainbow448", tokens, count, 5120);

        // llama-mtmd-debug (-m EXAONE-4.5-33B-Q4_K_M.gguf --mmproj mmproj-exaone-4.5-q8_0.gguf -n 448 --image rainbow):
        // 256 x 5120, sum 1411.853271, row 0 [0.1998, 0.0144, -0.2507].
        Assert.Equal(256, count);
        AssertMatches(tokens, 1411.853, 30.0, [0.1998f, 0.0144f, -0.2507f], 0.05f);
    }

    [Fact]
    public void DeepSeekOcr2_Rainbow1024_SamMatchesLlamaMtmdDebug()
    {
        string? path = FindModel("mmproj-deepseek-ocr-2-q8_0.gguf");
        Assert.SkipWhen(path is null, "mmproj-deepseek-ocr-2-q8_0.gguf not present");
        using var gguf = OpenTail.Stingray.Core.GgufModel.Open(path!);
        var enc = new DeepSeekOcr2VisionEncoder(gguf);
        float[] sam = enc.Sam(Rainbow(1024), 1024, out int side);
        Report("dsocr2 sam_output", sam, side * side, 896);

        // llama-mtmd-debug (-m deepseek-ocr-2-Q4_K_M.gguf --mmproj mmproj-deepseek-ocr-2-q8_0.gguf -n 1024 --image rainbow):
        // sam_output 16 x 16 x 896 token-major, sum -475.976685, row 0 [-0.1650, 0.0668, 0.0027]. The tool
        // segfaults later, inside the Qwen2 stage, so that stage is checked end to end instead.
        Assert.Equal(16, side);
        AssertMatches(sam, -475.977, 25.0, [-0.1650f, 0.0668f, 0.0027f], 0.03f);
    }

    [Fact]
    public void DeepSeekOcr2_Rainbow768Tile_SamMatchesLlamaMtmdDebug()
    {
        string? path = FindModel("mmproj-deepseek-ocr-2-q8_0.gguf");
        Assert.SkipWhen(path is null, "mmproj-deepseek-ocr-2-q8_0.gguf not present");
        using var gguf = OpenTail.Stingray.Core.GgufModel.Open(path!);
        float[] sam = new DeepSeekOcr2VisionEncoder(gguf).Sam(Rainbow(768), 768, out int side);
        Report("dsocr2 tile sam_output", sam, side * side, 896);

        // Same reference with -n 768 (a tile): exercises the bicubic 64->48 position-table resize and the
        // linear 127->95 relative-position resize of the global blocks. sam_output 12 x 12 x 896, sum
        // -277.124847, row 0 [-0.1602, 0.0755, -0.0094].
        Assert.Equal(12, side);
        AssertMatches(sam, -277.125, 15.0, [-0.1602f, 0.0755f, -0.0094f], 0.03f);
    }

    [Fact]
    public void Step3Vl_Rainbow448_MatchesLlamaMtmdDebug()
    {
        string? path = FindModel("mmproj-step3-vl-10b-F16.gguf");
        Assert.SkipWhen(path is null, "mmproj-step3-vl-10b-F16.gguf not present");
        using var model = Step3VlVisionModel.Open(path!);
        float[] tokens = new Step3VlVisionEncoder(model).Forward(Rainbow(448), 448, 448, 32, 32, out int count);
        Report("step3vl rainbow448", tokens, count, 4096);

        // llama-mtmd-debug (-m step3-vl-10b-Q2_K.gguf --mmproj mmproj-step3-vl-10b-F16.gguf -n 448 --image rainbow):
        // projector_out 64 x 4096, sum 3959.443115, row 0 [0.1267, 2.3648, 1.1197].
        Assert.Equal(64, count);
        AssertMatches(tokens, 3959.443, 40.0, [0.1267f, 2.3648f, 1.1197f], 0.05f);
    }

    [Fact]
    public void GraniteVision32_Rainbow384_MatchesLlamaMtmdDebug()
    {
        string? path = FindModel("mmproj-granite-vision-3.2-2b-f16.gguf");
        Assert.SkipWhen(path is null, "mmproj-granite-vision-3.2-2b-f16.gguf not present");
        using var model = LlavaVisionModel.Open(path!);
        float[] tokens = new LlavaVisionEncoder(model).Forward(Rainbow(384), 384, 384, 27, 27, out int count);
        Report("granite32 rainbow384", tokens, count, 2048);

        // llama-mtmd-debug (-m granite-vision-3.2-2b-Q3_K_S.gguf --mmproj mmproj-granite-vision-3.2-2b-f16.gguf -n 384
        // --image rainbow): 729 x 2048, sum -3284.434326, row 0 [0.0678, -0.1084, 0.0735]. Exercises the feature-layer
        // stack [4, 8, 16, 27] (4 x 1152 = 4608-wide projector input), which the port ignored until 2026-09-27.
        Assert.Equal(729, count);
        AssertMatches(tokens, -3284.434, 35.0, [0.0678f, -0.1084f, 0.0735f], 0.03f);
    }

    [Fact]
    public void GraniteVision40_Rainbow384_MatchesLlamaMtmdDebug()
    {
        string? path = FindModel("mmproj-granite-4.0-3b-vision-f16.gguf");
        Assert.SkipWhen(path is null, "mmproj-granite-4.0-3b-vision-f16.gguf not present");
        using var model = Granite4VisionModel.Open(path!);
        float[] tokens = new Granite4VisionEncoder(model).Forward(Rainbow(384), 384, 384, 24, 24, out int count);
        Report("granite40 rainbow384", tokens, count, 20480);

        // llama-mtmd-debug (-m granite-4.0-3b-vision-Q4_K_M.gguf --mmproj mmproj-granite-4.0-3b-vision-f16.gguf -n 384
        // --image rainbow): g4v_mmproj_out 144 x 20480 (8 QFormer blocks x 2560, concatenated per token: block 0 is the
        // input embedding, 1..7 the deepstack slices), sum 3587.707520, row 0 [-0.5940, -0.1377, 0.9753].
        Assert.Equal(144, count);
        AssertMatches(tokens, 3587.708, 40.0, [-0.5940f, -0.1377f, 0.9753f], 0.03f);
    }

    [Fact]
    public void Gemma4V_Rainbow224_MatchesLlamaMtmdDebug()
    {
        string? path = FindModel("gemma-4-E4B-it-mmproj.gguf");
        Assert.SkipWhen(path is null, "gemma-4-E4B-it-mmproj.gguf not present");
        using var model = Gemma4VVisionModel.Open(path!);
        var encoder = new Gemma4VVisionEncoder(model);
        float[] tokens = encoder.Forward(Rainbow(224));
        int count = encoder.TokenCount;
        Report("gemma4v rainbow224", tokens, count, 2560);
        Console.WriteLine($"[gemma4v] row0 tail [{tokens[2556]:F4}, {tokens[2557]:F4}, {tokens[2558]:F4}, {tokens[2559]:F4}]");

        // llama-mtmd-debug (-m gemma-4-E4B-it-Q4_K_M.gguf --mmproj gemma-4-E4B-it-mmproj.gguf -n 224 --image rainbow), 2026-09-27:
        // mm.input_projection output 16 x 2560, row 0 [0.3868, -0.1127, -0.3089, ..., -0.1903, 0.2254, ?]. The tool
        // asserts in clip.cpp while printing that last tensor, so its sum and final element are not available; the
        // pooled + normed + clamped input to the projection sums to -539.370239.
        // Measured 2026-09-27: row 0 [0.3864, -0.1130, -0.3097, ..., -0.1876, 0.2245, 0.5115], sum -106.2146.
        Assert.Equal(16, count);
        (int Index, float Ref)[] refs = [(0, 0.3868f), (1, -0.1127f), (2, -0.3089f), (2557, -0.1903f), (2558, 0.2254f)];
        foreach (var (i, r) in refs)
            Assert.InRange(tokens[i], r - 0.01f, r + 0.01f);
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
