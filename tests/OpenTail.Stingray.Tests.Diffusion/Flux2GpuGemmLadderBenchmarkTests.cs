using System.Diagnostics;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// docs/093 Phase 0, Experiment 3: real production-shape GEMM throughput ladder. Now the critical
/// diagnostic after Experiment 1+2 (docs/093, `Flux2DoubleBlockGpuBenchmarkTests`) proved the
/// double-block loop's GPU bottleneck is genuine execution time, not CPU-side dispatch overhead,
/// and that batching doesn't help. This test answers: is that execution time dominated by the
/// GEMMs themselves running below their isolated ~611 GFLOP/s <c>SgemmF16</c> benchmark at THESE
/// specific production shapes, or by the ~16 non-GEMM utility dispatches per block?
///
/// Benchmarks the exact 8 GEMM shapes FLUX.2's double-block loop uses (both img/txt-stream row
/// counts, all 4 weight-matrix shapes), using the same <c>Sgemm</c> dispatch and FP16 weight
/// upload path production code actually uses -- not a generic square-matrix microbenchmark.
/// </summary>
public sealed class Flux2GpuGemmLadderBenchmarkTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static void BenchOne(VulkanBackend backend, string label, int m, int k, int n)
    {
        var rng = new Random(m * 31 + k * 17 + n);
        var a = new float[m * k];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);

        var wF32 = new float[n * k];
        for (int i = 0; i < wF32.Length; i++) wF32[i] = (float)(rng.NextDouble() * 2 - 1);
        var wHalf = new Half[wF32.Length];
        for (int i = 0; i < wF32.Length; i++) wHalf[i] = (Half)wF32[i];

        var aGpu = backend.Upload(a, TensorShape.D2(m, k), exact: true);
        var wGpu = backend.UploadHalf(wHalf, TensorShape.D2(n, k));
        var cGpu = backend.Allocate(TensorShape.D2(m, n));

        try
        {
            // Warm-up (pipeline compile).
            backend.Sgemm(cGpu, aGpu, wGpu, m, k, n);
            backend.Synchronize();

            double bestMs = double.MaxValue;
            for (int trial = 0; trial < 3; trial++)
            {
                var sw = Stopwatch.StartNew();
                backend.Sgemm(cGpu, aGpu, wGpu, m, k, n);
                backend.Synchronize();
                sw.Stop();
                bestMs = Math.Min(bestMs, sw.Elapsed.TotalMilliseconds);
            }

            double gflop = 2.0 * m * k * n / 1e9;
            double gflops = gflop / (bestMs / 1000.0);
            Console.WriteLine($"[Flux2 GEMM ladder] {label} [{m}x{k}x{n}] best={bestMs:F2}ms {gflop:F2}GFLOP -> {gflops:F1} GFLOP/s");
        }
        finally
        {
            backend.Free(aGpu);
            backend.Free(wGpu);
            backend.Free(cGpu);
        }
    }

    [Fact]
    public void ProductionShapeGemmLadder_ReportsThroughput()
    {
        using var backend = TryCreateVulkan();
        if (backend is null) return;

        const int d = 6144;
        const int mlpHidden = 18432; // d * 3 (Flux2Params.MlpRatio)
        const int nImg = 1024, nTxt = 256;

        // The 8 real production shapes from Flux2DiT.DoubleBlockGpu, both stream row counts.
        BenchOne(backend, "QKV proj", nImg, d, d * 3);
        BenchOne(backend, "QKV proj", nTxt, d, d * 3);
        BenchOne(backend, "O proj", nImg, d, d);
        BenchOne(backend, "O proj", nTxt, d, d);
        BenchOne(backend, "FFN up", nImg, d, 2 * mlpHidden);
        BenchOne(backend, "FFN up", nTxt, d, 2 * mlpHidden);
        BenchOne(backend, "FFN down", nImg, mlpHidden, d);
        BenchOne(backend, "FFN down", nTxt, mlpHidden, d);
    }
}
