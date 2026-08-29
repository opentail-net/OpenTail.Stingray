
namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// SnapKV used to force prefill off the batched trunk entirely, because eviction scoring needs each
/// token's post-RoPE query and only the per-token <c>Forward</c> path captured it. That made every
/// prompt above the SnapKV budget take the slow path — measured at 6.4 vs 51.7 t/s on a 3218-token
/// prompt, an 8x cliff on default CLI settings. The batched trunk now captures the trailing
/// window's queries from <c>_qK</c> directly.
///
/// <para>These pin the contract that enabling the batched path did not change what SnapKV
/// computes: with SnapKV active, batched and per-token prefill must agree on the logits AND leave
/// the cache in the same evicted state, so continued decoding matches too.</para>
/// </summary>
public sealed class VulkanSnapKvBatchedPrefillTests : HeavyTestBase
{
    private const string BudgetVar = "STINGRAY_SNAPKV_BUDGET";
    private const string WindowVar = "STINGRAY_SNAPKV_WINDOW";

    private static string? FindModelPath()
    {
        var dir = AppDomain.CurrentDomain.BaseDirectory;
        while (dir is not null)
        {
            var p = Path.Combine(dir, "models", "SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
            if (File.Exists(p)) return p;
            dir = Directory.GetParent(dir)?.FullName;
        }
        return null;
    }

    private static VulkanBackend? TryCreate()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    /// <summary>A prompt comfortably above the small budget set below, so eviction really runs.</summary>
    private static int[] BuildPrompt(int n)
    {
        var t = new int[n];
        for (int i = 0; i < n; i++) t[i] = 5 + (i * 7) % 2000;
        return t;
    }

    [Fact]
    public void BatchedPrefillWithSnapKv_MatchesPerTokenPrefill()
    {
        var path = FindModelPath();
        Assert.SkipUnless(path is not null, "model fixture not present in this environment");
        using var gpu = TryCreate();
        Assert.SkipUnless(gpu is not null, "no usable GPU backend in this environment");

        string? priorBudget = Environment.GetEnvironmentVariable(BudgetVar);
        string? priorWindow = Environment.GetEnvironmentVariable(WindowVar);
        try
        {
            // Small budget + window so a short test prompt still triggers real eviction.
            Environment.SetEnvironmentVariable(BudgetVar, "64");
            Environment.SetEnvironmentVariable(WindowVar, "16");

            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
            int[] prompt = BuildPrompt(96);

            float[] reference, batched;
            int[] refDecode, batchedDecode;

            static (float[] Logits, int[] Decoded) Run(
                GgufModel m, VulkanBackend g, ModelHyperparams h, int[] prompt, bool disableBatched)
            {
                using var fwd = new GpuForwardPass(m, g, h, maxContextLength: 512);
                fwd.DisableBatchedPrefill = disableBatched;
                var logits = fwd.Prefill(prompt).ToArray();
                // Continue decoding: proves the evicted cache is in the same state, not just that
                // the final logits happened to line up.
                int next = Sampler.Greedy(logits);
                var produced = new int[4];
                for (int i = 0; i < produced.Length; i++)
                {
                    produced[i] = next;
                    next = Sampler.Greedy(fwd.Forward(next, prompt.Length + i));
                }
                return (logits, produced);
            }

            (reference, refDecode) = Run(model, gpu, hp, prompt, disableBatched: true);
            (batched, batchedDecode) = Run(model, gpu, hp, prompt, disableBatched: false);

            Assert.Equal(reference.Length, batched.Length);

            double maxAbs = 0;
            for (int i = 0; i < reference.Length; i++)
                maxAbs = Math.Max(maxAbs, Math.Abs(reference[i] - batched[i]));

            // Same tolerance rationale as VulkanBatchedPrefillTests: the batched trunk reduces in a
            // different order (and now uses flash attention's online softmax), so this is FP-noise
            // parity, not bit equality. A masking or capture-offset bug would diverge by O(1).
            Assert.True(maxAbs < 5e-3,
                $"batched prefill with SnapKV diverged from the per-token path: max abs {maxAbs:R}");
            Assert.Equal(Sampler.Greedy(reference), Sampler.Greedy(batched));
            Assert.Equal(refDecode, batchedDecode);
        }
        finally
        {
            Environment.SetEnvironmentVariable(BudgetVar, priorBudget);
            Environment.SetEnvironmentVariable(WindowVar, priorWindow);
        }
    }
}
