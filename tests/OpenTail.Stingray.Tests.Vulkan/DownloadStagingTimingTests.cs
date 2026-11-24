using System.Diagnostics;

namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// A31 measurement: <c>VulkanBackend.Download</c> of a device tensor into host memory, with the ordinary download staging type
/// and with the host-cached one (<see cref="VulkanBackend.DownloadStagingHostCached"/>). The two settings are timed in turn, twice,
/// so drift affects both; each timing is the median of five runs after two warm-ups. The test asserts that the data arrives
/// unchanged and prints the figures. The default stays off; it is changed only if the figures show a measurable gain.
/// </summary>
public sealed class DownloadStagingTimingTests
{
    private const int Repeats = 5;

    [Fact]
    public void Download_OrdinaryVersusHostCachedStaging_IsTimed_AndTheDataIsUnchanged()
    {
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var _gpu = gpu;

        Console.WriteLine("[download-staging] tensor              ordinary_ms  host_cached_ms  speedup  MiB");
        var rng = new Random(20260610);
        foreach (var (name, count) in new[] { ("block_1224x2048", 1224 * 2048), ("block_2014x2048", 2014 * 2048), ("logits_50304", 50304) })
        {
            var data = RandomFloats(rng, count);
            var tensor = gpu!.Upload(data, TensorShape.D1(count));
            var back = new float[count];

            var ordinary = new List<double>();
            var cached = new List<double>();
            for (int round = 0; round < 2; round++)
            {
                gpu.DownloadStagingHostCached = false;
                ordinary.Add(MedianMs(() => gpu.Download(tensor, back)));
                Assert.Equal(data, back);

                gpu.DownloadStagingHostCached = true;
                cached.Add(MedianMs(() => gpu.Download(tensor, back)));
                Assert.Equal(data, back);
            }

            double a = ordinary.Average(), b = cached.Average();
            Console.WriteLine(FormattableString.Invariant(
                $"[download-staging] {name,-20}  {a,11:F3}  {b,14:F3}  {a / b,7:F2}  {count * 4.0 / (1 << 20),4:F2}"));
            gpu.Free(tensor);
        }
        gpu!.DownloadStagingHostCached = false;   // leave the default in place
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var a = new float[count];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    /// <summary>Median of the measured runs after two warm-ups.</summary>
    private static double MedianMs(Action action)
    {
        action();
        action();
        var times = new double[Repeats];
        for (int i = 0; i < Repeats; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            action();
            times[i] = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        }
        Array.Sort(times);
        return times[Repeats / 2];
    }
}
