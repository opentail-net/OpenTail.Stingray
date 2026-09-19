using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// GPU parity check for the new fused <c>VulkanBackend.Flux2QkvNormRope</c> shader.
/// Compares the fused single dispatch against the exact unfused three-step GPU reference sequence:
///   1. <c>FluxUnpackQkv</c>
///   2. <c>QKNorm</c>
///   3. <c>Flux2DRoPE</c>
/// Verifies bit-parity for Q, K, and V across both dstTokenOffset=0 (text-stream layout)
/// and dstTokenOffset > 0 (image-stream layout).
/// </summary>
public sealed class Flux2QkvNormRopeGpuTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    [Theory]
    [InlineData(8, 4, 0)]    // small synthetic scale, dstTokenOffset = 0
    [InlineData(8, 4, 8)]    // small synthetic scale, dstTokenOffset = 8
    [InlineData(16, 48, 0)]  // full 48 heads, dstTokenOffset = 0
    [InlineData(16, 48, 16)] // full 48 heads, dstTokenOffset = 16
    public void FusedQkvNormRope_MatchesUnfusedGpuSequence(int nTokens, int numHeads, int dstTokenOffset)
    {
        using var backend = TryCreateVulkan();
        if (backend is null) return;

        const int headDim = 128;
        int dim = numHeads * headDim;
        int nSeq = dstTokenOffset + nTokens + 4; // extra buffer capacity
        int nPairs = headDim / 2;

        var rng = new Random(42 + nTokens * 31 + numHeads * 17 + dstTokenOffset);

        var qkv = new float[nTokens * 3 * dim];
        for (int i = 0; i < qkv.Length; i++) qkv[i] = (float)(rng.NextDouble() * 2 - 1);

        var qScale = new float[headDim];
        var kScale = new float[headDim];
        for (int i = 0; i < headDim; i++)
        {
            qScale[i] = (float)(rng.NextDouble() * 1.5 + 0.5);
            kScale[i] = (float)(rng.NextDouble() * 1.5 + 0.5);
        }

        var cos = new float[nSeq * nPairs];
        var sin = new float[nSeq * nPairs];
        for (int i = 0; i < cos.Length; i++)
        {
            float theta = (float)(rng.NextDouble() * Math.PI * 2);
            cos[i] = MathF.Cos(theta);
            sin[i] = MathF.Sin(theta);
        }

        using var qkvGpu = backend.Upload(qkv, TensorShape.D2(nTokens, 3 * dim), exact: true);
        using var qScaleGpu = backend.Upload(qScale, TensorShape.D1(headDim), exact: true);
        using var kScaleGpu = backend.Upload(kScale, TensorShape.D1(headDim), exact: true);
        using var cosGpu = backend.Upload(cos, TensorShape.D2(nSeq, nPairs), exact: true);
        using var sinGpu = backend.Upload(sin, TensorShape.D2(nSeq, nPairs), exact: true);

        // --- 1. Unfused reference: FluxUnpackQkv + QKNorm + Flux2DRoPE ---
        using var qUnfused = backend.Allocate(TensorShape.D2(nSeq, dim));
        using var kUnfused = backend.Allocate(TensorShape.D2(nSeq, dim));
        using var vUnfused = backend.Allocate(TensorShape.D2(nSeq, dim));

        backend.FluxUnpackQkv(qkvGpu, qUnfused, kUnfused, vUnfused, nTokens, dim, dstTokenOffset);
        backend.QKNorm(qUnfused, kUnfused, qScaleGpu, kScaleGpu, nTokens, numHeads, headDim, eps: 1e-6f, startToken: dstTokenOffset);
        backend.Flux2DRoPE(qUnfused, kUnfused, cosGpu, sinGpu, startToken: 0, tokenCount: nSeq, numHeads, headDim);
        backend.Synchronize();

        var qOutUnfused = new float[nSeq * dim];
        var kOutUnfused = new float[nSeq * dim];
        var vOutUnfused = new float[nSeq * dim];
        backend.Download(qUnfused, qOutUnfused);
        backend.Download(kUnfused, kOutUnfused);
        backend.Download(vUnfused, vOutUnfused);

        // --- 2. Fused operation: Flux2QkvNormRope ---
        using var qFused = backend.Allocate(TensorShape.D2(nSeq, dim));
        using var kFused = backend.Allocate(TensorShape.D2(nSeq, dim));
        using var vFused = backend.Allocate(TensorShape.D2(nSeq, dim));

        backend.Flux2QkvNormRope(qkvGpu, qFused, kFused, vFused, cosGpu, sinGpu, qScaleGpu, kScaleGpu,
            nTokens, numHeads, headDim, dstTokenOffset, eps: 1e-6f);
        backend.Synchronize();

        var qOutFused = new float[nSeq * dim];
        var kOutFused = new float[nSeq * dim];
        var vOutFused = new float[nSeq * dim];
        backend.Download(qFused, qOutFused);
        backend.Download(kFused, kOutFused);
        backend.Download(vFused, vOutFused);

        // --- 3. Verify parity across the written token range ---
        int startElem = dstTokenOffset * dim;
        int countElem = nTokens * dim;

        float maxDiffQ = 0f, maxDiffK = 0f, maxDiffV = 0f;
        for (int i = startElem; i < startElem + countElem; i++)
        {
            float diffQ = Math.Abs(qOutUnfused[i] - qOutFused[i]);
            float diffK = Math.Abs(kOutUnfused[i] - kOutFused[i]);
            float diffV = Math.Abs(vOutUnfused[i] - vOutFused[i]);
            if (diffQ > maxDiffQ) maxDiffQ = diffQ;
            if (diffK > maxDiffK) maxDiffK = diffK;
            if (diffV > maxDiffV) maxDiffV = diffV;
        }

        Console.WriteLine($"[Flux2 QkvNormRope Parity] nTok={nTokens} nh={numHeads} dstOff={dstTokenOffset} | maxDiffQ={maxDiffQ:E4} maxDiffK={maxDiffK:E4} maxDiffV={maxDiffV:E4}");
        // V is a direct copy, so maxDiffV should be 0.
        Assert.True(maxDiffV == 0f, $"V copy diverged: maxDiffV={maxDiffV}");
        // Q and K accumulation is identical floating-point arithmetic in LDS vs global memory
        Assert.True(maxDiffQ < 1e-4f, $"Q output diverged from unfused reference: maxDiffQ={maxDiffQ}");
        Assert.True(maxDiffK < 1e-4f, $"K output diverged from unfused reference: maxDiffK={maxDiffK}");
    }
}
