namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// Per-architecture Vulkan-vs-CPU logit parity on real checkpoints: prefill of a real sentence, then
/// eight teacher-forced decode steps (both backends are fed the CPU's greedy tokens, so a single
/// argmax flip cannot snowball into unrelated text). Same contract as
/// <see cref="VulkanCpuLogitParityTests"/>: cosine above a floor at every step, and an argmax
/// disagreement is only allowed when it is a near-tie on the CPU's own logits. Each model skips
/// visibly when its GGUF is not on this machine.
/// </summary>
public sealed class VulkanArchLogitParityTests : HeavyTestBase
{
    private readonly ITestOutputHelper _out;
    public VulkanArchLogitParityTests(ITestOutputHelper output) => _out = output;

    private const string PromptText =
        "The scheduler assigns runnable threads to cores, balancing throughput against latency, " +
        "and cache locality shapes where each thread is placed.";

    [Theory]
    [InlineData("Phi-3-mini-4k-instruct-Q4_K_M.gguf")]      // fused attn_qkv + fused gate/up
    [InlineData("OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf")]   // MoE + full-width (per-channel) QK-norm
    [InlineData("Mistral-7B-Instruct-v0.3-Q4_K_M.gguf")]
    [InlineData("stablelm-zephyr-3b.Q4_K_M.gguf")]           // LayerNorm + bias, partial NEOX RoPE
    [InlineData("c4ai-command-r7b-12-2024-Q4_K_M.gguf")]     // cohere2: parallel residual, LayerNorm, SWA + NoPE globals
    [InlineData("gpt2.Q8_0.gguf")]                           // learned position table, non-gated GELU + biases
    [InlineData("starcoder2-3b.Q4_K_M.gguf")]                // non-gated GELU + biases, LayerNorm
    [InlineData("pythia-160m.Q8_0.gguf")]                    // gpt-neox: parallel residual, partial NEOX RoPE, biases
    [InlineData("swiss-ai.Apertus-8B-Instruct-2509.Q4_K_M.gguf")] // non-gated xIELU FFN
    [InlineData("Maincoder-1B-Q4_K_M.gguf")]                 // weighted QK-norm AFTER RoPE
    [InlineData("tencent_Hunyuan-0.5B-Instruct-Q8_0.gguf")]  // hunyuan-dense: QK-norm after RoPE
    [InlineData("gemma-4-E4B-it-Q4_K_M.gguf")]
    [InlineData("gemma-3-4b-it-Q4_K_M.gguf")]
    [InlineData("qwen2.5-0.5b-instruct-q4_k_m.gguf")]       // attention bias
    [InlineData("qwen2.5-3b-instruct-q4_k_m.gguf")]         // attention bias on the batched prefill trunk (all Q4_K/Q6_K)
    [InlineData("Qwen3-0.6B-Q8_0.gguf")]                     // QK-norm
    public void PrefillAndDecodeLogits_AgreeWithCpu(string file)
    {
        string? path = FindModelPath(file);
        Assert.SkipWhen(path is null, $"{file} not present");
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var _gpu = gpu;

        using var model = GgufModel.Open(path!);
        // The model-aware overload, as the CLI and server use: it infers tensor-shape features
        // (e.g. IsPerChannelQkNorm) that the metadata-only one misses — without it this test
        // compared two equally-wrong OLMoE configurations and passed a broken Vulkan path.
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        int[] prompt = GgufTokenizer.FromGgufModel(model).Encode(PromptText).ToArray();
        const int steps = 8;

        var cpuLogits = new List<float[]>();
        var forced = new int[steps];
        using (var backend = new CpuBackend())
        using (var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 512))
        {
            var l = fwd.Prefill(prompt).ToArray();
            cpuLogits.Add(l);
            for (int s = 0; s < steps; s++)
            {
                forced[s] = Argmax(l);
                l = fwd.Forward(forced[s], prompt.Length + s).ToArray();
                cpuLogits.Add(l);
            }
        }

        var gpuLogits = new List<float[]>();
        using (var fwd = new GpuForwardPass(model, gpu!, hp, maxContextLength: 512))
        {
            gpuLogits.Add(fwd.Prefill(prompt).ToArray());
            for (int s = 0; s < steps; s++)
                gpuLogits.Add(fwd.Forward(forced[s], prompt.Length + s).ToArray());
        }

        double worstCos = 1;
        int flips = 0;
        var failures = new List<string>();
        for (int i = 0; i < cpuLogits.Count; i++)
        {
            float[] c = cpuLogits[i], g = gpuLogits[i];
            double cos = Cosine(c, g);
            worstCos = Math.Min(worstCos, cos);
            int ca = Argmax(c), ga = Argmax(g);
            string stage = i == 0 ? "prefill" : $"decode {i}";
            if (ca != ga)
            {
                flips++;
                float range = c.Max() - c.Min(), gap = c[ca] - c[ga];
                if (gap / range >= 0.02)
                    failures.Add($"{stage}: CPU {ca} vs Vulkan {ga}, CPU gap {gap:F3} of range {range:F3} (not a near-tie)");
            }
            if (cos <= 0.99) failures.Add($"{stage}: cosine {cos:F6}");
        }
        _out.WriteLine($"{file}: worst cosine {worstCos:F6}, argmax flips {flips}/{cpuLogits.Count}");
        Console.WriteLine($"[arch-parity] {file}: worst cosine {worstCos:F6}, argmax flips {flips}/{cpuLogits.Count}");
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
