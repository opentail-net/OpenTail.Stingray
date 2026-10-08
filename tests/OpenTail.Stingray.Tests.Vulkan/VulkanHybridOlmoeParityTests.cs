namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// <see cref="HybridForwardPass"/> (GPU layers + CPU layers, MoE experts through the SLRU slot cache) on OLMoE
/// against the all-CPU <see cref="Engine.ForwardPass"/>, at full and at a deliberately tiny expert-slot capacity.
///
/// <para>Found 2026-10-03: <c>stingray -g 8</c> on OLMoE answered "1 + 1 =" with "Top Level:" while
/// <c>-g 0</c> and <c>-g -1</c> said "= 2". OLMoE's QK-norm normalises across all heads (one 2048-wide row), and the
/// hybrid pass still normalised each head on its own (GPU <c>HeadNorm</c> and the CPU <c>PerChannelRmsNorm</c>),
/// although <c>GpuForwardPass</c> and the CPU <c>ForwardPass</c> had been fixed. No test covered the hybrid pass with
/// a MoE model. Small slot capacities force cache misses, which the hybrid serves on the CPU.</para>
///
/// <para>Same contract as <see cref="VulkanLayerSplitParityTests"/>: cosine above 0.99 everywhere, and an argmax
/// disagreement only on a near-tie of the CPU's own logits. Skips visibly without the checkpoint or a Vulkan device.</para>
/// </summary>
public sealed class VulkanHybridOlmoeParityTests : HeavyTestBase
{
    private const string Olmoe = "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf";

    [Theory]
    [InlineData(Olmoe, 4, -1)]  // all experts resident on the GPU layers
    [InlineData(Olmoe, 4, 8)]   // 8 slots for 4 layers x 64 experts: constant eviction and CPU fallback
    [InlineData(Olmoe, 8, 16)]
    [InlineData(Olmoe, 16, -1)] // every layer through the hybrid pass: isolates its GPU half
    [InlineData(Olmoe, 1, -1)]  // one GPU layer: isolates its CPU half
    // A 128-expert, 48-layer MoE (qwen3moe, 18 GB): 4 GPU layers x 128 experts through 16 slots.
    [InlineData("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", 4, 16)]
    // Weighted per-head QK-norm on the CPU layers: the same ordering fix (norm before RoPE) applies to Qwen3.
    [InlineData("Qwen3-0.6B-Q8_0.gguf", 8, -1)]
    public void PrefillAndDecodeLogits_AgreeWithCpu(string file, int gpuLayers, int expertSlots)
    {
        string? path = FindModelPath(file);
        Assert.SkipWhen(path is null, $"{file} not present");
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var _gpu = gpu;

        using var model = GgufModel.Open(path!);
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        int[] prompt = GgufTokenizer.FromGgufModel(model)
            .Encode("The scheduler assigns runnable threads to cores, balancing throughput against latency.").ToArray();
        const int steps = 8;

        var cpu = new List<float[]>();
        var forced = new int[steps];
        using (var backend = new CpuBackend())
        using (var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 512))
        {
            var l = fwd.Prefill(prompt).ToArray();
            cpu.Add(l);
            for (int s = 0; s < steps; s++)
            {
                forced[s] = Argmax(l);
                l = fwd.Forward(forced[s], prompt.Length + s).ToArray();
                cpu.Add(l);
            }
        }

        var placement = new LayerPlacement(
            GpuLayers: gpuLayers, CpuLayers: hp.NumLayers - gpuLayers,
            GpuWeightBytes: 0, GpuKvBytes: 0, RecommendedCtxSize: 512);
        var hyb = new List<float[]>();
        using (var fwd = new HybridForwardPass(model, gpu!, hp, placement, expertSlotCapacity: expertSlots))
        {
            hyb.Add(fwd.Prefill(prompt).ToArray());
            for (int s = 0; s < steps; s++)
                hyb.Add(fwd.Forward(forced[s], prompt.Length + s).ToArray());
        }

        double worst = 1;
        int flips = 0;
        var failures = new List<string>();
        for (int i = 0; i < cpu.Count; i++)
        {
            float[] c = cpu[i], g = hyb[i];
            double cos = Cosine(c, g);
            worst = Math.Min(worst, cos);
            int ca = Argmax(c), ga = Argmax(g);
            string stage = i == 0 ? "prefill" : $"decode {i}";
            if (ca != ga)
            {
                flips++;
                float range = c.Max() - c.Min(), gap = c[ca] - c[ga];
                if (gap / range >= 0.02)
                    failures.Add($"{stage}: CPU {ca} vs hybrid {ga}, CPU gap {gap:F3} of range {range:F3}");
            }
            if (cos <= 0.99) failures.Add($"{stage}: cosine {cos:F6}");
        }
        Console.WriteLine($"[hybrid] {file} -g {gpuLayers} slots {expertSlots}: worst cosine {worst:F6}, argmax flips {flips}/{cpu.Count}");
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    private static int Argmax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static double Cosine(float[] a, float[] b)
    {
        double d = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return d / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    private static string? FindModelPath(string file)
    {
        foreach (var p in new[] { Path.Combine(@"F:\_models", file), Path.Combine(@"E:\_models", file) })
            if (System.IO.File.Exists(p)) return p;
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            foreach (var p in new[] { Path.Combine(dir.FullName, "models", file), Path.Combine(dir.FullName, "models", "_models", file) })
                if (System.IO.File.Exists(p)) return p;
            dir = dir.Parent;
        }
        return null;
    }
}
