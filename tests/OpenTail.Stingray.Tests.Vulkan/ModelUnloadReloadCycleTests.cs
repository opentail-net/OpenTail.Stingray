namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// Load A, run A, unload A, load B, run B, unload B, load A, run A (TensorSharp-plan phase 7.3), on the CPU and Vulkan
/// forward passes. A lifetime bug (a buffer freed twice, a stale handle, memory not returned so the third load fails)
/// shows up as a crash, a failed allocation, or A's second run differing from its first. Greedy decoding on one device is
/// deterministic, so the two A runs must produce the identical token sequence. The same cycle on CUDA needs an NVIDIA GPU.
/// </summary>
public sealed class ModelUnloadReloadCycleTests : HeavyTestBase
{
    private const string A = "SmolLM2-135M-Instruct-Q4_K_M.gguf";
    private const string B = "SmolLM2-360M-Instruct-Q4_K_M.gguf";

    private static string? Find(string file)
    {
        foreach (var p in new[] { Path.Combine(@"F:\_models", file), Path.Combine(@"E:\_models", file) })
            if (File.Exists(p)) return p;
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            foreach (var p in new[] { Path.Combine(dir.FullName, "models", file), Path.Combine(dir.FullName, "models", "_models", file) })
                if (File.Exists(p)) return p;
            dir = dir.Parent;
        }
        return null;
    }

    /// <summary>Loads the model, greedily decodes 10 tokens after a short prompt, and disposes everything.</summary>
    private static int[] Run(string path, VulkanBackend? gpu)
    {
        using var model = GgufModel.Open(path);
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        int[] prompt = GgufTokenizer.FromGgufModel(model).Encode("The capital of France is").ToArray();
        var tokens = new List<int>();

        if (gpu is null)
        {
            using var backend = new CpuBackend();
            using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 256);
            Decode(fwd.Prefill(prompt), (t, pos) => fwd.Forward(t, pos), prompt.Length, tokens);
        }
        else
        {
            using var fwd = new GpuForwardPass(model, gpu, hp, maxContextLength: 256);
            Decode(fwd.Prefill(prompt), (t, pos) => fwd.Forward(t, pos), prompt.Length, tokens);
        }
        return [.. tokens];
    }

    private delegate ReadOnlySpan<float> Step(int token, int position);

    private static void Decode(ReadOnlySpan<float> logits, Step step, int promptLength, List<int> tokens)
    {
        for (int i = 0; i < 10; i++)
        {
            int best = 0;
            for (int k = 1; k < logits.Length; k++) if (logits[k] > logits[best]) best = k;
            tokens.Add(best);
            logits = step(best, promptLength + i);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LoadRunUnloadCycle_ReproducesTheFirstRun(bool vulkan)
    {
        string? a = Find(A), b = Find(B);
        Assert.SkipWhen(a is null || b is null, $"{A} and {B} are both required");
        VulkanBackend? gpu = null;
        if (vulkan)
        {
            try { gpu = new VulkanBackend(); } catch { gpu = null; }
            Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        }
        using var _gpu = gpu;

        int[] firstA = Run(a!, gpu);
        int[] onlyB = Run(b!, gpu);
        int[] secondA = Run(a!, gpu);
        int[] secondB = Run(b!, gpu);

        Assert.Equal(firstA, secondA);
        Assert.Equal(onlyB, secondB);
        Assert.NotEqual(firstA, onlyB); // the two models really are different, so equality above is not vacuous
        Console.WriteLine($"[unload-cycle] {(vulkan ? "Vulkan" : "CPU")}: A {string.Join(",", firstA)} | B {string.Join(",", onlyB)}");
    }
}
