namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// Streamed experts must not save memory in name only (SharpMind's lesson: its streaming looked right while cached
/// repacks kept every layer's weights alive). With a cache far smaller than the experts a run touches, every token
/// uploads and evicts; if an evicted slot's GPU buffers or its host staging arrays were retained, the live GPU buffer
/// count or the process's private memory would climb with the token count. This runs 40 warm-up tokens, records the
/// live buffer count and private bytes, runs 400 more, forces a GC, and requires both to be flat. Real OLMoE weights and
/// a real Vulkan device; skips visibly without either.
/// </summary>
public sealed class StreamedExpertMemoryBoundTests : HeavyTestBase
{
    private const string Olmoe = "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf";

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

    private static long PrivateBytesAfterGc()
    {
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
        using var p = System.Diagnostics.Process.GetCurrentProcess();
        p.Refresh();
        return p.PrivateMemorySize64;
    }

    [Fact]
    public void ManyTokensThroughATinyExpertCache_KeepGpuBuffersAndPrivateMemoryFlat()
    {
        string? path = Find(Olmoe);
        Assert.SkipWhen(path is null, $"{Olmoe} not present");
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var _gpu = gpu;

        using var model = GgufModel.Open(path!);
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        int[] prompt = GgufTokenizer.FromGgufModel(model).Encode("The history of computing began with").ToArray();
        var placement = new LayerPlacement(
            GpuLayers: 4, CpuLayers: hp.NumLayers - 4, GpuWeightBytes: 0, GpuKvBytes: 0, RecommendedCtxSize: 512);

        using var fwd = new HybridForwardPass(model, gpu!, hp, placement, expertSlotCapacity: 12); // 4 layers x 64 experts through 12 slots

        static int Argmax(ReadOnlySpan<float> v)
        {
            int best = 0;
            for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
            return best;
        }

        int pos = prompt.Length;
        int next = Argmax(fwd.Prefill(prompt));
        void Decode(int steps)
        {
            for (int s = 0; s < steps; s++)
                next = Argmax(fwd.Forward(next, pos++));
        }

        Decode(40);
        // Let in-flight prefetch uploads settle so the baseline is not caught mid-upload.
        Thread.Sleep(500);
        int buffers0 = gpu!.LiveBufferCount;
        long bytes0 = PrivateBytesAfterGc();

        Decode(400);
        Thread.Sleep(500);
        int buffers1 = gpu.LiveBufferCount;
        long bytes1 = PrivateBytesAfterGc();

        Console.WriteLine($"[mem-bound] tokens 40 -> 440: live GPU buffers {buffers0} -> {buffers1}; private bytes {bytes0 / 1048576.0:F1} -> {bytes1 / 1048576.0:F1} MiB");

        // 12 slots x 3 tensors = 36 expert buffers at most, plus whatever a prefetch has in flight at the moment of sampling.
        Assert.True(buffers1 <= buffers0 + 12,
            $"live GPU buffers grew from {buffers0} to {buffers1} over 400 tokens: evicted slots are not being freed");
        // 400 tokens x (4 layers x 8 experts) = 12,800 expert uploads of ~10 MB each would be 128 GB if retained.
        Assert.True(bytes1 - bytes0 < 256L * 1024 * 1024,
            $"private memory grew by {(bytes1 - bytes0) / 1048576.0:F0} MiB over 400 tokens: streamed expert data is being retained");
    }
}
