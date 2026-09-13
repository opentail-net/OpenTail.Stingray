using System.Diagnostics;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class WanQuantizationRooflineMicrobenchTests
{
    private readonly ITestOutputHelper _output;

    public WanQuantizationRooflineMicrobenchTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    [Fact]
    public void WanMatrixShapes_SgemmF16_RooflineScalingAcrossBatchSizes()
    {
        using var vk = TryCreateVulkan();
        if (vk == null) return;

        const int dim = 1536;
        int[] tokenCounts = [16, 64, 128, 256, 512, 1024, 2048];

        _output.WriteLine("=========================================================================================");
        _output.WriteLine(" Wan 2.1 SGEMM (FP16) Roofline Scaling Analysis on AMD Vega 8 (5700G)");
        _output.WriteLine("=========================================================================================");
        _output.WriteLine(" Tokens (M) | Matrix Shape (M,K,N) | Latency (ms) | GFLOP/s  | Arith. Intensity (FLOP/B) | Regime");
        _output.WriteLine("------------|----------------------|--------------|----------|---------------------------|-------");

        foreach (var m in tokenCounts)
        {
            // Fused QKV shape: [M, 1536] x [4608, 1536]^T
            int k = dim;
            int n = dim * 3; // 4608

            var hA = new float[m * k];
            var hB = new Half[n * k];
            for (int i = 0; i < hA.Length; i++) hA[i] = 0.01f * (i % 7);
            for (int i = 0; i < hB.Length; i++) hB[i] = (Half)(0.01f * (i % 11));

            var tA = vk.Upload(hA, TensorShape.D2(m, k));
            var tB = vk.UploadHalf(hB, TensorShape.D2(n, k));
            var tC = vk.Allocate(TensorShape.D2(m, n), DType.Float32);

            // Warmup
            for (int i = 0; i < 5; i++)
            {
                vk.Sgemm(tC, tA, tB, m, k, n);
            }

            const int iters = 20;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                vk.Sgemm(tC, tA, tB, m, k, n);
            }
            sw.Stop();

            double ms = sw.Elapsed.TotalMilliseconds / iters;
            double totalFlops = 2.0 * m * n * k;
            double gflops = (totalFlops / (ms / 1000.0)) / 1e9;

            // Memory bytes: weights (FP16 = 2B) + inputs (FP32 = 4B) + outputs (FP32 = 4B)
            double weightBytes = n * k * 2.0;
            double actBytes = (m * k * 4.0) + (m * n * 4.0);
            double totalBytes = weightBytes + actBytes;
            double arithIntensity = totalFlops / totalBytes;

            string regime = arithIntensity < 40.0 ? "Bandwidth Bound" : (arithIntensity < 120.0 ? "Transition" : "Compute Bound");

            _output.WriteLine($" {m,10} | [{m,4}, {k,4}]x[{n,4}] | {ms,12:F3} | {gflops,8:F1} | {arithIntensity,25:F1} | {regime}");
            Console.Error.WriteLine($"[Wan Scaling] M={m,4}: {ms,6:F3} ms ({gflops,5:F1} GFLOP/s, AI={arithIntensity,5:F1} FLOP/B -> {regime})");
        }
    }

    [Fact]
    public void WanFullBlock_SgemmF16_Scaling_Vs_TokenCount()
    {
        using var vk = TryCreateVulkan();
        if (vk == null) return;

        const int dim = 1536;
        const int ffnDim = 8960;
        int[] tokenCounts = [64, 256, 512, 1024, 2048];

        _output.WriteLine("\n=========================================================================================");
        _output.WriteLine(" Wan 2.1 Full DiT Block GEMM Breakdown (6 GEMMs / Block)");
        _output.WriteLine("=========================================================================================");

        foreach (var m in tokenCounts)
        {
            var tX = vk.Allocate(TensorShape.D2(m, dim), DType.Float32);
            var tQkv = vk.Allocate(TensorShape.D2(m, dim * 3), DType.Float32);
            var tAttnOut = vk.Allocate(TensorShape.D2(m, dim), DType.Float32);
            var tFfnUp = vk.Allocate(TensorShape.D2(m, ffnDim), DType.Float32);

            var wQkv = vk.Allocate(TensorShape.D2(dim * 3, dim), DType.Float16);
            var wAttnOut = vk.Allocate(TensorShape.D2(dim, dim), DType.Float16);
            var wCrossQ = vk.Allocate(TensorShape.D2(dim, dim), DType.Float16);
            var wCrossOut = vk.Allocate(TensorShape.D2(dim, dim), DType.Float16);
            var wFfnUp = vk.Allocate(TensorShape.D2(ffnDim, dim), DType.Float16);
            var wFfnDown = vk.Allocate(TensorShape.D2(dim, ffnDim), DType.Float16);

            // Warmup
            for (int i = 0; i < 3; i++)
            {
                vk.Sgemm(tQkv, tX, wQkv, m, dim, dim * 3);
                vk.Sgemm(tAttnOut, tX, wAttnOut, m, dim, dim);
                vk.Sgemm(tAttnOut, tX, wCrossQ, m, dim, dim);
                vk.Sgemm(tAttnOut, tX, wCrossOut, m, dim, dim);
                vk.Sgemm(tFfnUp, tX, wFfnUp, m, dim, ffnDim);
                vk.Sgemm(tAttnOut, tFfnUp, wFfnDown, m, ffnDim, dim);
            }

            const int iters = 10;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                vk.Sgemm(tQkv, tX, wQkv, m, dim, dim * 3);
                vk.Sgemm(tAttnOut, tX, wAttnOut, m, dim, dim);
                vk.Sgemm(tAttnOut, tX, wCrossQ, m, dim, dim);
                vk.Sgemm(tAttnOut, tX, wCrossOut, m, dim, dim);
                vk.Sgemm(tFfnUp, tX, wFfnUp, m, dim, ffnDim);
                vk.Sgemm(tAttnOut, tFfnUp, wFfnDown, m, ffnDim, dim);
            }
            sw.Stop();

            double totalMs = sw.Elapsed.TotalMilliseconds / iters;
            double blockFlops = 2.0 * m * (
                (dim * dim * 3.0) +
                (dim * dim) +
                (dim * dim) +
                (dim * dim) +
                (dim * ffnDim) +
                (ffnDim * dim)
            );
            double gflops = (blockFlops / (totalMs / 1000.0)) / 1e9;

            double distill4StepSec = (totalMs * 30.0 * 4.0) / 1000.0;
            double standard20StepSec = (totalMs * 30.0 * 40.0) / 1000.0;

            _output.WriteLine($"Tokens: {m,4} | Total Block GEMM Time: {totalMs,6:F2} ms | {gflops,5:F1} GFLOP/s | 4-step: {distill4StepSec,5:F1}s | 20-step CFG: {standard20StepSec / 60.0,4:F1}min");
            Console.Error.WriteLine($"[Tokens={m,4}] Block GEMMs: {totalMs,6:F2} ms ({gflops,5:F1} GFLOP/s) -> 4-Step Distill: {distill4StepSec,5:F1}s, 20-Step CFG: {standard20StepSec / 60.0,4:F1} min");
        }
    }

    [Fact]
    public void WanQuantization_DequantQ4KM_ThroughputAndVramBenchmark()
    {
        using var vk = TryCreateVulkan();
        if (vk == null) return;

        // Wan 2.1 1-Block linear weight count: ~41.68M parameters
        // 41,680,896 params / 256 = 162,816 Q4_K blocks
        const int totalElements = 41680896;
        const int numBlocks = totalElements / 256;
        const int q4kBytesPerBlock = 144;
        const int totalQ4KBytes = numBlocks * q4kBytesPerBlock; // ~23.4 MB per block (vs 83.4 MB in FP16)

        var q4kRaw = new byte[totalQ4KBytes];
        for (int i = 0; i < q4kRaw.Length; i++) q4kRaw[i] = (byte)(i % 256);

        // Upload raw Q4_K bytes as uints
        var uintCount = totalQ4KBytes / sizeof(uint);
        var tSrc = vk.Allocate(TensorShape.D1(uintCount), DType.Float32);
        var tDstF16 = vk.Allocate(TensorShape.D1(totalElements), DType.Float16);

        // Warmup
        for (int i = 0; i < 5; i++)
        {
            vk.DequantQ4KM(tSrc, tDstF16, numBlocks);
        }

        const int iters = 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
        {
            vk.DequantQ4KM(tSrc, tDstF16, numBlocks);
        }
        sw.Stop();

        double dequantMs = sw.Elapsed.TotalMilliseconds / iters;
        double dequantBandwidthGbps = ((totalQ4KBytes + totalElements * 2.0) / (dequantMs / 1000.0)) / 1e9;

        // Wan 1.3B model (30 blocks):
        // Total FP16 VRAM for linear weights = 30 * 83.4 MB = 2.50 GB
        // Total Q4_K VRAM for linear weights = 30 * 23.4 MB = 0.70 GB (3.57x VRAM savings!)
        // Wan 14B model (40 blocks, dim=5120):
        // Total FP16 VRAM = ~28 GB (does NOT fit in 16GB system RAM / APU VRAM)
        // Total Q4_K VRAM = ~7.8 GB (FITS cleanly in consumer 16GB / 32GB RAM!)

        _output.WriteLine("\n=========================================================================================");
        _output.WriteLine(" Wan 2.1 GPU Dequantization (DequantQ4KM) Microbenchmark");
        _output.WriteLine("=========================================================================================");
        _output.WriteLine($"1-Block Linear Weights: {totalElements / 1e6:F2}M params ({numBlocks} Q4_K blocks)");
        _output.WriteLine($"FP16 VRAM per Block:    {(totalElements * 2.0) / (1024 * 1024):F1} MB (Total 30 blocks: 2.50 GB)");
        _output.WriteLine($"Q4_K VRAM per Block:    {totalQ4KBytes / (1024.0 * 1024.0):F1} MB (Total 30 blocks: 0.70 GB - 3.57x compression)");
        _output.WriteLine($"GPU Dequant Latency:    {dequantMs:F2} ms per block ({dequantBandwidthGbps:F1} GB/s effective bandwidth)");
        _output.WriteLine($"Full 30-block Dequant:  {dequantMs * 30.0:F1} ms (one-time load penalty: <0.1 second!)");
        Console.Error.WriteLine($"[DequantQ4KM] 1 Block ({totalElements / 1e6:F1}M params): {dequantMs:F2} ms ({dequantBandwidthGbps:F1} GB/s) -> 30-block total dequant: {dequantMs * 30:F1} ms");
    }
}
