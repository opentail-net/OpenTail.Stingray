using System.Linq;
using System.Runtime.InteropServices;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Perf-loop iteration 7 (docs/perf-loop-progress.md): a ChatGPT-assisted review pointed out our
/// measured decode throughput (20-25 tok/s x 1.06 GiB model = ~21-27 GB/s) is plausible on
/// DDR4-3200 dual-channel (51.2 GB/s theoretical peak), but a claimed 4x gap to llama.cpp would
/// imply ~85-106 GB/s -- physically impossible on this memory class. Combined with this box's
/// already-documented core-count fluctuation (12 vs 16 across sessions) and the project's own
/// historical baseline being measured at 24 threads (this box currently has 12), this strongly
/// suggests a resource-constrained/shared VM, not bare metal. This test measures ACTUAL achievable
/// sequential-read bandwidth on this box directly, multi-threaded, at real model-sized (~1GB)
/// scan volumes, to establish a hard ceiling independent of any kernel/dispatch code.
/// </summary>
public sealed unsafe class RawMemoryBandwidthPerfTests(ITestOutputHelper output)
{
    [Fact]
    public unsafe void PerfGauge_RawSequentialReadBandwidth()
    {
        // ~1.06 GiB, matching SmolLM2-1.7B-Instruct-Q4_K_M's resident weight size (confirmed via
        // this session's own "[ForwardPass] Pre-faulted 1.06 GiB" load-time log line).
        long totalBytes = (long)(1.06 * 1024 * 1024 * 1024);
        int threads = Environment.ProcessorCount;

        byte* buf = (byte*)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
        try
        {
            // Touch every page once so the timed pass isn't paying first-fault cost.
            for (long i = 0; i < totalBytes; i += 4096) buf[i] = 1;

            long checksum = 0;
            void ScanOnce()
            {
                long chunk = totalBytes / threads;
                long localSum = 0;
                var sums = new long[threads];
                Parallel.For(0, threads, t =>
                {
                    long from = t * chunk;
                    long to = (t == threads - 1) ? totalBytes : from + chunk;
                    long s = 0;
                    byte* p = buf + from;
                    long len = to - from;
                    // Sum 8 bytes at a time as longs -- sequential read, minimal compute, so this
                    // is bandwidth-bound not compute-bound (same "raw scan" concept the review
                    // suggested as rung 1 of a 3-rung raw/parse/kernel comparison).
                    long i = 0;
                    for (; i + 8 <= len; i += 8)
                        s += *(long*)(p + i);
                    for (; i < len; i++)
                        s += p[i];
                    sums[t] = s;
                });
                foreach (var s in sums) localSum += s;
                checksum = localSum;
            }

            const int warmup = 3;
            const int timedRuns = 6; // iteration 5's established n=6 minimum for this box
            for (int i = 0; i < warmup; i++) ScanOnce();

            double[] ms = new double[timedRuns];
            for (int run = 0; run < timedRuns; run++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                ScanOnce();
                sw.Stop();
                ms[run] = sw.Elapsed.TotalMilliseconds;
            }

            var gbps = ms.Select(m => (totalBytes / 1e9) / (m / 1000.0)).ToArray();
            double mean = gbps.Average();
            double stdev = Math.Sqrt(gbps.Select(g => (g - mean) * (g - mean)).Sum() / (gbps.Length - 1));

            output.WriteLine(
                $"Raw sequential-read bandwidth, {threads} threads, {totalBytes / (1024.0 * 1024 * 1024):F2} GiB scanned per run, {timedRuns} runs:\n" +
                string.Join("\n", ms.Select((m, i) => $"  run{i + 1}: {m:F2}ms -> {gbps[i]:F2} GB/s")) +
                $"\nmean = {mean:F2} GB/s, stdev = {stdev:F2} GB/s\n" +
                $"DDR4-3200 dual-channel theoretical peak: 51.2 GB/s\n" +
                $"checksum (prevents dead-code elimination): {checksum}");

            Assert.True(checksum != 0 || totalBytes == 0); // sanity: the scan actually ran
        }
        finally
        {
            NativeMemory.AlignedFree(buf);
        }
    }
}
