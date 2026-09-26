namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// Gemma 4 E4B split across Vulkan and CPU (<see cref="Gemma4VulkanSplitForwardPass"/>, -g N) against
/// the all-CPU <see cref="Engine.ForwardPass"/>, at a small split and at the largest legal one: a real prompt, then
/// eight teacher-forced steps on the CPU's greedy tokens. Same contract as
/// <see cref="VulkanArchLogitParityTests"/>: cosine above 0.99 everywhere, and an argmax
/// disagreement only on a near-tie of the CPU's own logits. Skips visibly without the checkpoint.
/// </summary>
public sealed class Gemma4SplitParityTests : HeavyTestBase
{
    private const string ModelFile = "gemma-4-E4B-it-Q4_K_M.gguf";

    [Theory]
    [InlineData(4)]
    [InlineData(-1)] // MaxGpuLayers
    public void PrefillAndDecodeLogits_AgreeWithCpu(int split)
    {
        string? path = FindModelPath();
        Assert.SkipWhen(path is null, $"{ModelFile} not present");
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var _gpu = gpu;

        using var model = GgufModel.Open(path!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
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

        var gpuLogits = new List<float[]>();
        using (var fwd = new Gemma4VulkanSplitForwardPass(model, gpu!, hp, split > 0 ? split : Gemma4VulkanSplitForwardPass.MaxGpuLayers(hp), maxContextLength: 512))
        {
            gpuLogits.Add(fwd.Prefill(prompt).ToArray());
            for (int s = 0; s < steps; s++)
                gpuLogits.Add(fwd.Forward(forced[s], prompt.Length + s).ToArray());

            // SupportsPartialRewind: rewinding to the prompt and replaying step 1 is exact.
            Assert.True(fwd.SupportsPartialRewind);
            fwd.TruncateTo(prompt.Length);
            Assert.Equal(gpuLogits[1], fwd.Forward(forced[0], prompt.Length).ToArray());
        }

        double worst = 1;
        int flips = 0;
        var failures = new List<string>();
        for (int i = 0; i < cpu.Count; i++)
        {
            float[] c = cpu[i], g = gpuLogits[i];
            double cos = Cosine(c, g);
            worst = Math.Min(worst, cos);
            int ca = Argmax(c), ga = Argmax(g);
            string stage = i == 0 ? "prefill" : $"decode {i}";
            if (ca != ga)
            {
                flips++;
                float range = c.Max() - c.Min(), gap = c[ca] - c[ga];
                if (gap / range >= 0.02)
                    failures.Add($"{stage}: CPU {ca} vs Vulkan {ga}, CPU gap {gap:F3} of range {range:F3}");
            }
            Console.WriteLine($"[gemma4-split] {stage}: cos {cos:F6} argmax CPU {ca} GPU {ga}");
            if (cos <= 0.99) failures.Add($"{stage}: cosine {cos:F6}");
        }
        Console.WriteLine($"[gemma4-split] split {split}: worst cosine {worst:F6}, argmax flips {flips}/{cpu.Count}");
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

    private static string? FindModelPath()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            foreach (var p in new[] { Path.Combine(dir.FullName, "models", ModelFile), Path.Combine(dir.FullName, "models", "_models", ModelFile) })
                if (File.Exists(p)) return p;
            dir = dir.Parent;
        }
        return null;
    }
}
