namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// The Vulkan hybrid (<c>-g N</c>) applies the model scale hyperparameters (embedding, residual, logit, attention-scale override) like the
/// full Vulkan pass. Before it did, Granite 3.3-2B answered with garbage (sequential hybrid against the CPU pass: minimum cosine -0.36) and
/// gemma-3-4b disagreed (0.47). These tests teacher-force one prompt through the CPU pass and through the hybrid token by token and compare
/// the logits at every step. Real weights and a real Vulkan device; skip visibly without either.
/// </summary>
public sealed unsafe class HybridScaleParityTests : HeavyTestBase
{
    private const string Text =
        "In 1843 Ada Lovelace published notes describing how a machine could manipulate symbols according to rules, " +
        "an idea that reached far beyond arithmetic. A century later wartime codebreakers built electronic devices to " +
        "search enormous spaces of possibilities, and stored-program computers followed within a few years. Transistors " +
        "replaced vacuum tubes, integrated circuits replaced transistors, and the cost of a calculation fell by many " +
        "orders of magnitude.";

    private static string? Find(string file)
    {
        foreach (var p in new[] { Path.Combine(@"F:\_models", file), Path.Combine(@"E:\_models", file), Path.Combine(@"H:\_models", file) })
            if (File.Exists(p)) return p;
        return null;
    }

    private static double Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double d = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return d / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    private static double Norm(ReadOnlySpan<float> a)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += (double)a[i] * a[i];
        return Math.Sqrt(s);
    }

    private static int Argmax(ReadOnlySpan<float> v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    [Theory]
    [InlineData("ibm-granite_granite-3.3-2b-instruct-Q4_K_M.gguf", 8)]
    [InlineData("ibm-granite_granite-3.3-2b-instruct-Q4_K_M.gguf", 20)]
    [InlineData("gemma-3-4b-it-Q4_K_M.gguf", 8)]
    [InlineData("gemma-3-4b-it-Q4_K_M.gguf", 20)]
    // The CPU-resident embedding and output branch (taken for very large vocabularies): it computes the logits itself, so it needs the logit scale too.
    [InlineData("ibm-granite_granite-3.3-2b-instruct-Q4_K_M.gguf", 8, true)]
    [InlineData("gemma-3-4b-it-Q4_K_M.gguf", 8, true)]
    public void SequentialHybrid_AgreesWithTheCpuPass_ForModelsThatSetScaleFields(string file, int gpuLayers, bool forceCpuOutput = false)
    {
        string? path = Find(file);
        Assert.SkipWhen(path is null, $"{file} not present");
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var gpuOwner = gpu!;
        using var model = GgufModel.Open(path!);
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        Assert.True(hp.EmbeddingScale != 1f || hp.ResidualScale != 1f || hp.LogitScale != 1f || hp.AttentionScaleOverride != 0f, "the model sets no scale field");
        int[] prompt = GgufTokenizer.FromGgufModel(model).Encode(Text).ToArray();
        var sw = System.Diagnostics.Stopwatch.StartNew();

        var cpuLogits = new List<float[]>();
        using (var backend = new CpuBackend())
        using (var cpu = new Engine.ForwardPass(model, backend, hp, maxContextLength: 1024))
            for (int i = 0; i < prompt.Length; i++) cpuLogits.Add(cpu.Forward(prompt[i], i).ToArray());

        var cos = new List<double>();
        var ratio = new List<double>();   // logit magnitude: cosine ignores a uniform scale, so a missing logit scale needs this
        int agree = 0;
        using (var hybrid = new HybridForwardPass(model, gpuOwner, hp,
                   new LayerPlacement(gpuLayers, hp.NumLayers - gpuLayers, 0, 0, 1024), expertSlotCapacity: -1, forceCpuEmbeddingOutput: forceCpuOutput))
            for (int i = 0; i < prompt.Length; i++)
            {
                var l = hybrid.Forward(prompt[i], i);
                cos.Add(Cosine(l, cpuLogits[i]));
                ratio.Add(Norm(l) / Norm(cpuLogits[i]));
                if (Argmax(l) == Argmax(cpuLogits[i])) agree++;
            }
        cos.Sort();
        ratio.Sort();
        Console.WriteLine($"[scale-parity] {file} -g {gpuLayers}{(forceCpuOutput ? " (CPU-resident output)" : "")}: {prompt.Length} steps, min cosine {cos[0]:F5}, median {cos[cos.Count / 2]:F5}, argmax agree {agree}/{prompt.Length}, median norm ratio {ratio[ratio.Count / 2]:F4}, {sw.Elapsed.TotalSeconds:F1}s");
        Assert.True(cos[0] > 0.98, $"minimum per-step cosine {cos[0]:F4} (median {cos[cos.Count / 2]:F4})");
        Assert.True(cos[cos.Count / 2] > 0.995, $"median per-step cosine {cos[cos.Count / 2]:F4}");
        Assert.InRange(ratio[ratio.Count / 2], 0.98, 1.02);
    }

    /// <summary>Control: the full Vulkan pass (which already applies every scale field) against the CPU pass on the same prompt.</summary>
    [Theory]
    [InlineData("gemma-3-4b-it-Q4_K_M.gguf")]
    [InlineData("ibm-granite_granite-3.3-2b-instruct-Q4_K_M.gguf")]
    public void Control_FullGpuPass_VsTheCpuPass(string file)
    {
        string? path = Find(file);
        Assert.SkipWhen(path is null, $"{file} not present");
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var gpuOwner = gpu!;
        using var model = GgufModel.Open(path!);
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        int[] prompt = GgufTokenizer.FromGgufModel(model).Encode(Text).ToArray();

        var cpuLogits = new List<float[]>();
        using (var backend = new CpuBackend())
        using (var cpu = new Engine.ForwardPass(model, backend, hp, maxContextLength: 1024))
            for (int i = 0; i < prompt.Length; i++) cpuLogits.Add(cpu.Forward(prompt[i], i).ToArray());

        var cos = new List<double>();
        int agree = 0;
        using (var full = new GpuForwardPass(model, gpuOwner, hp, maxContextLength: 1024))
            for (int i = 0; i < prompt.Length; i++)
            {
                var l = full.Forward(prompt[i], i);
                cos.Add(Cosine(l, cpuLogits[i]));
                if (Argmax(l) == Argmax(cpuLogits[i])) agree++;
            }
        cos.Sort();
        Console.WriteLine($"[scale-parity] CONTROL full GPU {file}: {prompt.Length} steps, min cosine {cos[0]:F5}, median {cos[cos.Count / 2]:F5}, argmax agree {agree}/{prompt.Length}");
    }
}