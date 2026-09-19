using System.Diagnostics;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// docs/093 Step 1: benchmark MultiHeadAttentionTiled in isolation at production shape.
/// Covers the joint attention dispatch that was NOT part of the GEMM ladder (Flux2GpuGemmLadderBenchmarkTests).
/// Production shape: 1280 total tokens (1024 img + 256 txt), 48 heads, headDim=128, dim=6144.
/// Measures wall-clock time for this single dispatch in isolation across several timed trials.
/// </summary>
public sealed class Flux2GpuAttentionBenchmarkTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    [Fact]
    public void JointAttention_ReportsThroughputAndTiming()
    {
        using var backend = TryCreateVulkan();
        if (backend is null) return;

        const int nImg = 1024;
        const int nTxt = 256;
        const int nSeq = nImg + nTxt; // 1280
        const int numHeads = 48;
        const int headDim = 128;
        const int dim = numHeads * headDim; // 6144
        const int totalElements = nSeq * dim;

        var rng = new Random(42);
        var q = new float[totalElements];
        var k = new float[totalElements];
        var v = new float[totalElements];
        for (int i = 0; i < totalElements; i++)
        {
            q[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            k[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        var qGpu = backend.Upload(q, TensorShape.D2(nSeq, dim), exact: true);
        var kGpu = backend.Upload(k, TensorShape.D2(nSeq, dim), exact: true);
        var vGpu = backend.Upload(v, TensorShape.D2(nSeq, dim), exact: true);
        var outGpu = backend.Allocate(TensorShape.D2(nSeq, dim));

        try
        {
            // Warm-up (pipeline compile & initial execution)
            backend.MultiHeadAttentionTiled(outGpu, qGpu, kGpu, vGpu, nSeq, nSeq, numHeads, headDim);
            backend.Synchronize();

            double bestMs = double.MaxValue;
            double sumMs = 0;
            const int trials = 5;
            for (int trial = 0; trial < trials; trial++)
            {
                var sw = Stopwatch.StartNew();
                backend.MultiHeadAttentionTiled(outGpu, qGpu, kGpu, vGpu, nSeq, nSeq, numHeads, headDim);
                backend.Synchronize();
                sw.Stop();
                double ms = sw.Elapsed.TotalMilliseconds;
                sumMs += ms;
                bestMs = Math.Min(bestMs, ms);
                Console.WriteLine($"[Flux2 Attention] Trial {trial + 1}/{trials}: {ms:F2}ms");
            }

            double avgMs = sumMs / trials;
            // FLOPs: QK dot = 2 * heads * qSeq * kvSeq * headDim; Softmax ~ 3 * heads * qSeq * kvSeq; AV dot = 2 * heads * qSeq * kvSeq * headDim.
            // Matrix ops alone = 4 * heads * qSeq * kvSeq * headDim.
            double gflop = (4.0 * numHeads * nSeq * nSeq * headDim) / 1e9;
            double gflops = gflop / (bestMs / 1000.0);
            double passTotalBest = bestMs * 8;
            double passTotalAvg = avgMs * 8;

            Console.WriteLine($"[Flux2 Attention] shape=[{nSeq}x{dim}] nh={numHeads} hd={headDim} best={bestMs:F2}ms avg={avgMs:F2}ms {gflop:F2}GFLOP -> {gflops:F1} GFLOP/s");
            Console.WriteLine($"[Flux2 Attention] 8-block total: best={passTotalBest:F1}ms avg={passTotalAvg:F1}ms");
        }
        finally
        {
            backend.Free(qGpu);
            backend.Free(kGpu);
            backend.Free(vGpu);
            backend.Free(outGpu);
        }
    }
}
