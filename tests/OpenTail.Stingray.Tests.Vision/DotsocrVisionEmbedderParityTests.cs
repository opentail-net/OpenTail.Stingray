namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// dots.ocr vision encoder + projector against the vendored llama.cpp reference. The golden values
/// are <c>tools/llama.cpp/llama-mtmd-debug.exe -m dots.ocr-Q8_0.gguf --mmproj mmproj-dots.ocr-Q8_0.gguf
/// -p encode -n 224 --image cb</c>: a 224x224 checkerboard fed straight to the encoder (all three
/// channels <c>(x+y)%2 ? 0 : 1</c>), 64 output tokens; the tool prints each stage's corner values
/// and sum. Pins the 2026-09-26 fixes (fused attn_qkv, SwiGLU gate, mm.post_norm, vision M-RoPE
/// section reset, erf GELU): before them the model ignored the image (1-token stop).
/// </summary>
public sealed class DotsocrVisionEmbedderParityTests
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
        Assert.InRange(sum, 15843.735 * 0.999, 15843.735 * 1.001);
        Assert.InRange(tokens[0], -2.5777 - 0.05, -2.5777 + 0.05);
        Assert.InRange(tokens[1], -1.0951 - 0.05, -1.0951 + 0.05);
        Assert.InRange(tokens[2], 7.7399 - 0.05, 7.7399 + 0.05);
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
