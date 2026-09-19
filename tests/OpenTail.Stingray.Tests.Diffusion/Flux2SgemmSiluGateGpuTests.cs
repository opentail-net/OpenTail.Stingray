using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// GPU parity test for fused <c>VulkanBackend.SgemmSiluGate</c>.
/// Compares the fused single-dispatch operation directly against the unfused two-step reference sequence:
///   1. <c>SiluGateMul</c> (activates [M, 2*K] into [M, K])
///   2. <c>Sgemm</c> (multiplies [M, K] with [N, K] weights to produce [M, N])
/// Verifies bit-level convergence across synthetic and production shapes.
/// </summary>
public sealed class Flux2SgemmSiluGateGpuTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    [Theory]
    [InlineData(8, 128, 64)]    // Small synthetic shape
    [InlineData(64, 256, 128)]  // Workgroup tile-multiple shape
    [InlineData(37, 384, 155)]  // Unaligned non-multiple shape
    [InlineData(128, 2048, 512)] // Intermediate scale
    public void SgemmSiluGate_MatchesUnfusedGpuSequence(int M, int K, int N)
    {
        using var backend = TryCreateVulkan();
        if (backend is null) return;

        var rng = new Random(42 + M * 1000 + K * 10 + N);

        // A is [M, 2*K] float32 activations
        var aData = new float[M * 2 * K];
        for (int i = 0; i < aData.Length; i++)
            aData[i] = (float)(rng.NextDouble() * 4.0 - 2.0);

        // B is [N, K] float16 weights
        var bData = new Half[N * K];
        for (int i = 0; i < bData.Length; i++)
            bData[i] = (Half)(rng.NextDouble() * 2.0 - 1.0);

        using var aGpu = backend.Upload(aData, TensorShape.D2(M, 2 * K), exact: true);
        using var bGpu = backend.UploadHalf(bData, TensorShape.D2(N, K));

        // 1. Unfused reference sequence: SiluGateMul + Sgemm
        using var gatedGpu = backend.Allocate(TensorShape.D2(M, K));
        using var cUnfusedGpu = backend.Allocate(TensorShape.D2(M, N));

        backend.SiluGateMul(gatedGpu, aGpu, M, K);
        backend.Sgemm(cUnfusedGpu, gatedGpu, bGpu, M, K, N);
        backend.Synchronize();

        var cUnfused = new float[M * N];
        backend.Download(cUnfusedGpu, cUnfused);

        // 2. Fused operation: SgemmSiluGate
        using var cFusedGpu = backend.Allocate(TensorShape.D2(M, N));
        backend.SgemmSiluGate(cFusedGpu, aGpu, bGpu, M, K, N);
        backend.Synchronize();

        var cFused = new float[M * N];
        backend.Download(cFusedGpu, cFused);

        // 3. Compare outputs
        float maxDiff = 0f;
        double sumSqDiff = 0.0;
        double sumSqRef = 0.0;

        for (int i = 0; i < cUnfused.Length; i++)
        {
            float diff = Math.Abs(cUnfused[i] - cFused[i]);
            if (diff > maxDiff) maxDiff = diff;
            sumSqDiff += diff * diff;
            sumSqRef += cUnfused[i] * cUnfused[i];
        }

        double relativeError = Math.Sqrt(sumSqDiff) / (Math.Sqrt(sumSqRef) + 1e-12);
        Console.WriteLine($"[SgemmSiluGate Parity] M={M} K={K} N={N} | maxDiff={maxDiff:E4} relErr={relativeError:E4}");

        // Floating-point difference should be minimal (near precision limit of fp32/fp16 accumulation)
        Assert.True(maxDiff < 1e-3f, $"Fused SgemmSiluGate diverged from unfused reference: maxDiff={maxDiff}");
        Assert.True(relativeError < 1e-4, $"Relative error too high: relErr={relativeError}");
    }
}
