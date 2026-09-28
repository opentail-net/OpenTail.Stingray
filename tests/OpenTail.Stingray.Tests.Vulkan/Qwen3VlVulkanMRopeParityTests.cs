using OpenTail.Stingray.Vision;

namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// Qwen3-VL image input on the Vulkan <see cref="GpuForwardPass"/> against the CPU <see cref="ForwardPass"/>
/// (docs/103 item 14, "CPU-only vision features"). Same soft tokens (projector + 3 deepstack slices, 8192 wide)
/// through both passes, with the image registered for M-RoPE (IMROPE) positions, then the logits after a text
/// suffix are compared. A negative control runs the GPU pass without registering the image: its 1D positions
/// must land measurably further from the CPU, which shows the per-pair positions are what closes the gap.
/// Needs <c>Qwen3VL-2B-Instruct-Q8_0.gguf</c> and <c>mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf</c> in
/// <c>models/_models</c> or in the directory named by <c>STINGRAY_QWEN3VL_DIR</c>.
/// </summary>
public sealed class Qwen3VlVulkanMRopeParityTests : HeavyTestBase
{
    private readonly ITestOutputHelper _out;
    public Qwen3VlVulkanMRopeParityTests(ITestOutputHelper output) => _out = output;

    private static string? Find(string file)
    {
        if (Environment.GetEnvironmentVariable("STINGRAY_QWEN3VL_DIR") is { Length: > 0 } envDir
            && File.Exists(Path.Combine(envDir, file)))
            return Path.Combine(envDir, file);
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8 && dir is not null; i++)
        {
            foreach (var p in new[] { Path.Combine(dir, "models", file), Path.Combine(dir, "models", "_models", file) })
                if (File.Exists(p)) return p;
            dir = Directory.GetParent(dir)?.FullName;
        }
        return null;
    }

    /// <summary>224x224 RGB with four solid quadrants (red, green / blue, yellow), so 2D positions matter.</summary>
    private static byte[] QuadrantImage(int size)
    {
        var rgb = new byte[size * size * 3];
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                bool right = x >= size / 2, bottom = y >= size / 2;
                (byte r, byte g, byte b) = (right, bottom) switch
                {
                    (false, false) => ((byte)230, (byte)20, (byte)20),
                    (true, false) => ((byte)20, (byte)200, (byte)40),
                    (false, true) => ((byte)30, (byte)40, (byte)220),
                    _ => ((byte)240, (byte)220, (byte)20),
                };
                int o = (y * size + x) * 3;
                rgb[o] = r; rgb[o + 1] = g; rgb[o + 2] = b;
            }
        return rgb;
    }

    private static float[] RunPrompt(IForwardPass fwd, int[] prefix, float[] soft, int nTok, int embd,
        (int Width, int Height) grid, int[] suffix, bool registerImage)
    {
        int pos = 0;
        ReadOnlySpan<float> logits = default;
        foreach (int t in prefix) logits = fwd.Forward(t, pos++);
        if (registerImage)
        {
            if (fwd is Engine.ForwardPass cpu) cpu.AddMRopeImage(pos, grid.Width, grid.Height);
            else if (fwd is GpuForwardPass gpu) gpu.AddMRopeImage(pos, grid.Width, grid.Height);
        }
        for (int i = 0; i < nTok; i++) logits = fwd.ForwardEmbedding(soft.AsSpan(i * embd, embd), pos++);
        foreach (int t in suffix) logits = fwd.Forward(t, pos++);
        return logits.ToArray();
    }

    private static double Cosine(float[] a, float[] b)
    {
        double d = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return d / Math.Sqrt(na * nb);
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    [Fact]
    public void Qwen3Vl_ImageLogits_VulkanMatchesCpu_WithMRopeAndDeepstack()
    {
        var textPath = Find("Qwen3VL-2B-Instruct-Q8_0.gguf");
        var mmprojPath = Find("mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf");
        Assert.SkipWhen(textPath is null || mmprojPath is null, "Qwen3-VL 2B Q8_0 + mmproj not present");
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no usable Vulkan device");
        using var _ = gpu;

        using var model = GgufModel.Open(textPath!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.True(hp.RopeSections is { Count: > 0 } && hp.NumDeepstack > 0, "expected qwen3vl M-RoPE + deepstack");
        var tok = GgufTokenizer.FromGgufModel(model);

        using var vision = UnifiedVisionPipeline.Open(mmprojPath!);
        const int size = 224;
        float[] soft = vision.EmbedImage(QuadrantImage(size), size, size, out int nTok);
        var grid = vision.LastTokenGrid;
        int embd = vision.EmbeddingDim;
        Assert.Equal(hp.EmbeddingDim * (1 + hp.NumDeepstack), embd);
        Assert.Equal(nTok, grid.Width * grid.Height);

        int[] prefix = [.. tok.Encode("<|im_start|>user\n<|vision_start|>")];
        int[] suffix = [.. tok.Encode("<|vision_end|>Which colour is in the top-left corner?<|im_end|>\n<|im_start|>assistant\n")];

        float[] cpuLogits;
        using (var backend = new CpuBackend())
        using (var cpu = new Engine.ForwardPass(model, backend, hp, maxContextLength: 1024))
            cpuLogits = RunPrompt(cpu, prefix, soft, nTok, embd, grid, suffix, registerImage: true);

        // Second control: M-RoPE registered but only the projector slice fed (deepstack dropped).
        var softNoDs = new float[nTok * hp.EmbeddingDim];
        for (int i = 0; i < nTok; i++) Array.Copy(soft, i * embd, softNoDs, i * hp.EmbeddingDim, hp.EmbeddingDim);

        float[] gpuLogits, gpuNoMRope, gpuNoDeepstack;
        using (var g = new GpuForwardPass(model, gpu!, hp, maxContextLength: 1024))
        {
            gpuLogits = RunPrompt(g, prefix, soft, nTok, embd, grid, suffix, registerImage: true);
            g.ResetCache();
            gpuNoMRope = RunPrompt(g, prefix, soft, nTok, embd, grid, suffix, registerImage: false);
            g.ResetCache();
            gpuNoDeepstack = RunPrompt(g, prefix, softNoDs, nTok, hp.EmbeddingDim, grid, suffix, registerImage: true);
        }

        double cos = Cosine(cpuLogits, gpuLogits), cosControl = Cosine(cpuLogits, gpuNoMRope);
        double cosNoDs = Cosine(cpuLogits, gpuNoDeepstack);
        int cpuTop = ArgMax(cpuLogits), gpuTop = ArgMax(gpuLogits), controlTop = ArgMax(gpuNoMRope);
        _out.WriteLine($"grid {grid.Width}x{grid.Height} ({nTok} tokens); cos(cpu, gpu) = {cos:F6}, top {cpuTop} '{tok.Decode([cpuTop])}' vs {gpuTop}; " +
            $"control without M-RoPE: cos {cosControl:F6}, top {controlTop} '{tok.Decode([controlTop])}'; " +
            $"control without deepstack: cos {cosNoDs:F6}");

        Assert.Equal(cpuTop, gpuTop);
        Assert.True(cos > 0.999, $"cos(cpu, gpu) = {cos:F6}");
        Assert.True(cos - cosControl > 1e-3, $"control (no M-RoPE) cos {cosControl:F6} is not clearly worse than {cos:F6}");
        Assert.True(cos - cosNoDs > 1e-3, $"control (no deepstack) cos {cosNoDs:F6} is not clearly worse than {cos:F6}");
    }
}
