namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// A32 diagnostic: where the OLMoE hybrid's logits part from the CPU engine. The same prompt and the same forced token sequence
/// (the CPU engine's argmax) go through the CPU engine and three hybrid variants: all layers on the CPU, the GPU layers with every
/// routed expert on the CPU path (<see cref="HybridForwardPass.ForceCpuExpertPath"/>), and the GPU layers with the default expert
/// path. Per stage, the largest absolute logit difference, the smallest cosine and the argmax disagreements are printed. The
/// forced-CPU-expert variant is run twice and must be bitwise identical (the output does not depend on upload timing).
/// Heavy-gated; skips visibly without the checkpoint or a Vulkan device.
/// </summary>
public sealed class HybridLogitsDiagnosticTests : HeavyTestBase
{
    private const string Olmoe = "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf";
    private const int Steps = 8;

    [Fact]
    public void OlmoeHybridLogits_VersusCpuEngine_ByVariant()
    {
        string? path = OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(Olmoe);
        Assert.SkipWhen(path is null, $"{Olmoe} not present");
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var _gpu = gpu;

        using var model = GgufModel.Open(path!);
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        int[] prompt = GgufTokenizer.FromGgufModel(model)
            .Encode("The scheduler assigns runnable threads to cores, balancing throughput against latency.").ToArray();

        var cpu = new List<float[]>();
        var forced = new int[Steps];
        using (var backend = new CpuBackend())
        using (var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 512))
        {
            var l = fwd.Prefill(prompt).ToArray();
            cpu.Add(l);
            for (int s = 0; s < Steps; s++)
            {
                forced[s] = Argmax(l);
                l = fwd.Forward(forced[s], prompt.Length + s).ToArray();
                cpu.Add(l);
            }
        }
        Console.WriteLine($"[a32-logits] CPU engine: prompt {prompt.Length} tokens, {Steps} forced steps, vocabulary {cpu[0].Length}");

        var variants = new (string Name, int GpuLayers, bool ForceCpu)[]
        {
            ("hybrid, all layers CPU (-g 0)", 0, false),
            ("hybrid, 16 GPU layers, experts forced to CPU", 16, true),
            ("hybrid, 16 GPU layers, default experts", 16, false),
        };
        foreach (var v in variants)
        {
            List<float[]> hyb;
            try { hyb = RunHybrid(model, gpu!, hp, v.GpuLayers, v.ForceCpu, prompt, forced); }
            catch (Exception ex)
            {
                Console.WriteLine($"[a32-logits] {v.Name}: not constructible: {ex.GetType().Name}: {ex.Message}");
                continue;
            }
            var (maxAbs, minCos, flips, stageMax) = Compare(cpu, hyb);
            Console.WriteLine(FormattableString.Invariant(
                $"[a32-logits] {v.Name}: max_abs_diff {maxAbs:E3}, min_cosine {minCos:F7}, argmax_disagree {flips}/{cpu.Count}"));
            Console.WriteLine("[a32-logits]   per stage max_abs_diff: " + string.Join(" ", stageMax.Select(x => x.ToString("E1", System.Globalization.CultureInfo.InvariantCulture))));
            Assert.True(minCos > 0.99, $"{v.Name}: cosine {minCos:F6} against the CPU engine");
        }

        // The forced-CPU-expert output must not depend on timing: two runs are bitwise identical.
        var run1 = RunHybrid(model, gpu!, hp, 16, true, prompt, forced);
        var run2 = RunHybrid(model, gpu!, hp, 16, true, prompt, forced);
        for (int i = 0; i < run1.Count; i++)
            Assert.Equal(run1[i], run2[i]);
        Console.WriteLine("[a32-logits] forced-CPU-expert variant: two runs bitwise identical");
    }

    private static List<float[]> RunHybrid(GgufModel model, VulkanBackend gpu, ModelHyperparams hp, int gpuLayers, bool forceCpu,
        int[] prompt, int[] forced)
    {
        var placement = new LayerPlacement(
            GpuLayers: gpuLayers, CpuLayers: hp.NumLayers - gpuLayers,
            GpuWeightBytes: 0, GpuKvBytes: 0, RecommendedCtxSize: 512);
        var logits = new List<float[]>();
        using var fwd = new HybridForwardPass(model, gpu, hp, placement, expertSlotCapacity: -1) { ForceCpuExpertPath = forceCpu };
        logits.Add(fwd.Prefill(prompt).ToArray());
        for (int s = 0; s < Steps; s++)
            logits.Add(fwd.Forward(forced[s], prompt.Length + s).ToArray());
        return logits;
    }

    private static (double MaxAbs, double MinCos, int Flips, double[] StageMax) Compare(List<float[]> reference, List<float[]> other)
    {
        double maxAbs = 0, minCos = 1;
        int flips = 0;
        var stageMax = new double[reference.Count];
        for (int i = 0; i < reference.Count; i++)
        {
            float[] c = reference[i], g = other[i];
            double stage = 0;
            for (int k = 0; k < c.Length; k++) stage = Math.Max(stage, Math.Abs((double)c[k] - g[k]));
            stageMax[i] = stage;
            maxAbs = Math.Max(maxAbs, stage);
            minCos = Math.Min(minCos, Cosine(c, g));
            if (Argmax(c) != Argmax(g)) flips++;
        }
        return (maxAbs, minCos, flips, stageMax);
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
}
