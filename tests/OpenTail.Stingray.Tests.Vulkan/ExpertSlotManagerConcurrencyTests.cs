namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// <see cref="ExpertSlotManager"/> under the access pattern the Vulkan hybrid pass produces: the forward pass looks
/// experts up for every token while a background prefetcher uploads others. <c>Preload</c> used to hold the manager's
/// lock for the whole upload (allocate + copy about 10 MB per OLMoE expert), so every lookup behind it stalled; measured
/// with STINGRAY_MOE_TIMING that was 14-43 ms of cache lookup per MoE layer on a small cache, which turned 20 tok/s
/// into 2 tok/s. Real OLMoE weights and a real Vulkan device; skips visibly without either.
/// </summary>
public sealed class ExpertSlotManagerConcurrencyTests : HeavyTestBase
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

    [Fact]
    public void LookupsDoNotWaitForUploads_AndConcurrentPreloadsAreSafe()
    {
        string? path = Find(Olmoe);
        Assert.SkipWhen(path is null, $"{Olmoe} not present");
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var _gpu = gpu;

        using var model = GgufModel.Open(path!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var dtypes = new Dictionary<nint, DType>();
        const int layers = 4, experts = 24, capacity = 40; // 96 distinct experts through 40 slots: constant eviction

        var errors = new System.Collections.Concurrent.ConcurrentQueue<Exception>();
        var latencies = new List<double>(4000);
        long hits = 0, lookups = 0;
        using var stop = new CancellationTokenSource();

        using (var manager = new ExpertSlotManager(gpu!, model, hp, capacity, dtypes))
        {
            // Three prefetch threads with overlapping keys (the same expert requested twice at once must upload once).
            var workers = Enumerable.Range(0, 3).Select(t => new Thread(() =>
            {
                var rng = new Random(100 + t);
                try
                {
                    while (!stop.IsCancellationRequested)
                        manager.Preload(rng.Next(layers), rng.Next(experts));
                }
                catch (Exception e) { errors.Enqueue(e); }
            }) { IsBackground = true }).ToArray();
            foreach (var w in workers) w.Start();

            // The "forward pass": 4000 lookups while uploads are running; each one is timed.
            var look = new Random(7);
            var sw = new System.Diagnostics.Stopwatch();
            for (int i = 0; i < 4000; i++)
            {
                sw.Restart();
                bool hit = manager.TryGetCached(look.Next(layers), look.Next(experts), out var slot);
                sw.Stop();
                latencies.Add(sw.Elapsed.TotalMilliseconds);
                lookups++;
                if (hit)
                {
                    hits++;
                    Assert.NotNull(slot.Gate);
                    Assert.NotNull(slot.Up);
                    Assert.NotNull(slot.Down);
                }
                if (i % 50 == 0) Thread.Sleep(1); // let the uploaders interleave rather than starve
            }

            stop.Cancel();
            foreach (var w in workers) Assert.True(w.Join(TimeSpan.FromSeconds(60)), "a prefetch thread did not finish");
        } // Dispose with the cache full

        Assert.True(errors.IsEmpty, string.Join("; ", errors.Select(e => e.Message)));
        Assert.True(hits > 0, "no lookup ever hit: the uploads never published a slot");

        latencies.Sort();
        double p50 = latencies[latencies.Count / 2], p99 = latencies[(int)(latencies.Count * 0.99)], max = latencies[^1];
        Console.WriteLine($"[slot-concurrency] {lookups} lookups, {hits} hits; latency p50 {p50:F4} ms, p99 {p99:F4} ms, max {max:F3} ms");
        // An upload is several milliseconds. A lookup that waits behind one would put p99 at that level or above.
        Assert.True(p99 < 2.0, $"lookup p99 {p99:F3} ms: lookups are waiting behind uploads");
    }
}
