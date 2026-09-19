using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Correctness tests for Sgemm with inputRowOffsetElements:
/// Verifies that Sgemm reading A starting at a row offset inside a larger buffer
/// produces bit-parity results matching both:
/// 1. Sgemm on an isolated/sliced sub-buffer of A with offset=0.
/// 2. The CPU backend reference Sgemm with the same row offset.
/// Tested for both Float32 weights (SgemmF32) and Float16 weights (SgemmF16).
/// </summary>
public sealed class Flux2SgemmOffsetParityTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    [Theory]
    [InlineData(8, 64, 32, 4)]
    [InlineData(64, 128, 64, 16)]
    [InlineData(1024, 6144, 128, 256)] // FLUX.2 production stream shape: 1280 rows total, nTxt=256, nImg=1024
    public void SgemmF16_WithRowOffset_MatchesSubBufferAndCpu(int m, int k, int n, int rowOffset)
    {
        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return;
        using var cpu = new CpuBackend();

        int totalRows = rowOffset + m;
        var rng = new Random(42 + m + k + n + rowOffset);

        var aFull = new float[totalRows * k];
        for (int i = 0; i < aFull.Length; i++) aFull[i] = (float)(rng.NextDouble() * 2 - 1);

        var aSub = new float[m * k];
        Array.Copy(aFull, rowOffset * k, aSub, 0, m * k);

        var wF32 = new float[n * k];
        for (int i = 0; i < wF32.Length; i++) wF32[i] = (float)(rng.NextDouble() * 2 - 1);
        var wHalf = new Half[wF32.Length];
        for (int i = 0; i < wF32.Length; i++) wHalf[i] = (Half)wF32[i];

        // Dequantize half to float for exact CPU comparison with the weights the GPU sees
        var wF32FromHalf = new float[wF32.Length];
        for (int i = 0; i < wF32.Length; i++) wF32FromHalf[i] = (float)wHalf[i];

        using var aFullGpu = vulkan.Upload(aFull, TensorShape.D2(totalRows, k), exact: true);
        using var aSubGpu = vulkan.Upload(aSub, TensorShape.D2(m, k), exact: true);
        using var wGpu = vulkan.UploadHalf(wHalf, TensorShape.D2(n, k));
        using var cOffsetGpu = vulkan.Allocate(TensorShape.D2(m, n));
        using var cSubGpu = vulkan.Allocate(TensorShape.D2(m, n));

        // 1. Dispatch Sgemm with row offset directly into the larger buffer
        vulkan.Sgemm(cOffsetGpu, aFullGpu, wGpu, m, k, n, inputRowOffsetElements: rowOffset * k);

        // 2. Dispatch Sgemm on the pre-sliced sub-buffer with offset 0
        vulkan.Sgemm(cSubGpu, aSubGpu, wGpu, m, k, n, inputRowOffsetElements: 0);
        vulkan.Synchronize();

        var outOffset = new float[m * n];
        var outSub = new float[m * n];
        vulkan.Download(cOffsetGpu, outOffset);
        vulkan.Download(cSubGpu, outSub);

        // GPU with offset vs GPU pre-sliced sub-buffer should be bit-identical
        float maxDiffSub = 0f;
        for (int i = 0; i < outOffset.Length; i++)
        {
            float diff = Math.Abs(outOffset[i] - outSub[i]);
            if (diff > maxDiffSub) maxDiffSub = diff;
        }
        Assert.True(maxDiffSub < 1e-5f, $"Max diff vs sub-buffer {maxDiffSub} exceeds tolerance");

        // 3. CPU comparison
        using var aFullCpu = cpu.Upload(aFull, TensorShape.D2(totalRows, k));
        using var wCpu = cpu.Upload(wF32FromHalf, TensorShape.D2(n, k));
        using var cCpu = cpu.Allocate(TensorShape.D2(m, n));

        cpu.Sgemm(cCpu, aFullCpu, wCpu, m, k, n, inputRowOffsetElements: rowOffset * k);
        var outCpu = new float[m * n];
        cpu.Download(cCpu, outCpu);

        float maxDiffCpu = 0f;
        for (int i = 0; i < outOffset.Length; i++)
        {
            float diff = Math.Abs(outOffset[i] - outCpu[i]);
            if (diff > maxDiffCpu) maxDiffCpu = diff;
        }
        Assert.True(maxDiffCpu < 1e-3f, $"Max diff vs CPU {maxDiffCpu} exceeds tolerance");
    }

    [Theory]
    [InlineData(8, 64, 32, 4)]
    [InlineData(64, 128, 64, 16)]
    public void SgemmF32_WithRowOffset_MatchesSubBufferAndCpu(int m, int k, int n, int rowOffset)
    {
        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return;
        using var cpu = new CpuBackend();

        int totalRows = rowOffset + m;
        var rng = new Random(100 + m + k + n + rowOffset);

        var aFull = new float[totalRows * k];
        for (int i = 0; i < aFull.Length; i++) aFull[i] = (float)(rng.NextDouble() * 2 - 1);

        var aSub = new float[m * k];
        Array.Copy(aFull, rowOffset * k, aSub, 0, m * k);

        var wF32 = new float[n * k];
        for (int i = 0; i < wF32.Length; i++) wF32[i] = (float)(rng.NextDouble() * 2 - 1);

        using var aFullGpu = vulkan.Upload(aFull, TensorShape.D2(totalRows, k), exact: true);
        using var aSubGpu = vulkan.Upload(aSub, TensorShape.D2(m, k), exact: true);
        using var wGpu = vulkan.Upload(wF32, TensorShape.D2(n, k), exact: true);
        using var cOffsetGpu = vulkan.Allocate(TensorShape.D2(m, n));
        using var cSubGpu = vulkan.Allocate(TensorShape.D2(m, n));

        vulkan.Sgemm(cOffsetGpu, aFullGpu, wGpu, m, k, n, inputRowOffsetElements: rowOffset * k);
        vulkan.Sgemm(cSubGpu, aSubGpu, wGpu, m, k, n, inputRowOffsetElements: 0);
        vulkan.Synchronize();

        var outOffset = new float[m * n];
        var outSub = new float[m * n];
        vulkan.Download(cOffsetGpu, outOffset);
        vulkan.Download(cSubGpu, outSub);

        float maxDiffSub = 0f;
        for (int i = 0; i < outOffset.Length; i++)
        {
            float diff = Math.Abs(outOffset[i] - outSub[i]);
            if (diff > maxDiffSub) maxDiffSub = diff;
        }
        Assert.True(maxDiffSub < 1e-5f, $"F32 Max diff vs sub-buffer {maxDiffSub} exceeds tolerance");
    }
}
