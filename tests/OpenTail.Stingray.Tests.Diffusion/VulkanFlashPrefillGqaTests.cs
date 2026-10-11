using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// The causal GQA prefill attention kernel (<c>VulkanBackend.AttentionFlashPrefillGqa</c>, head dim 128) against the existing
/// 16-query flash kernel run in slices, on random fp32 caches. Covers chunk sizes around the 64-query tile, a non-zero base position
/// and group sizes 1, 2, 4 and 5.
/// </summary>
public sealed class VulkanFlashPrefillGqaTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    [Theory]
    [InlineData(32u, 8u, 0u, 17)]
    [InlineData(32u, 8u, 0u, 64)]
    [InlineData(32u, 8u, 0u, 100)]
    [InlineData(32u, 8u, 77u, 256)]
    [InlineData(32u, 8u, 1000u, 130)]
    [InlineData(16u, 8u, 300u, 256)]
    [InlineData(8u, 8u, 5u, 70)]
    [InlineData(40u, 8u, 63u, 65)]
    public void MatchesSlicedFlashKernel(uint numHeads, uint numKvHeads, uint basePos, int numQueries)
    {
        using var vk = TryCreateVulkan();
        Assert.SkipUnless(vk != null, "no Vulkan device");
        Assert.SkipUnless(vk!.SupportsFlashPrefillGqa(numHeads, numKvHeads, 128), "shape not served by the kernel");

        const uint headDim = 128, maxSeq = 2048;
        uint kvDim = numKvHeads * headDim, qDim = numHeads * headDim;
        var rnd = new Random(7 + (int)numHeads + numQueries);
        float[] Rand(long n) { var a = new float[n]; for (long i = 0; i < n; i++) a[i] = (float)(rnd.NextDouble() * 2 - 1); return a; }
        var kHost = Rand((long)maxSeq * kvDim);
        var vHost = Rand((long)maxSeq * kvDim);
        var qHost = Rand((long)numQueries * qDim);

        var k = vk.Upload(kHost, TensorShape.D1(kHost.Length), exact: true);
        var v = vk.Upload(vHost, TensorShape.D1(vHost.Length), exact: true);
        var q = vk.Upload(qHost, TensorShape.D1(qHost.Length), exact: true);
        var outNew = vk.Allocate(TensorShape.D1((long)numQueries * qDim));
        var outRef = vk.Allocate(TensorShape.D1((long)numQueries * qDim));

        vk.AttentionFlashPrefillGqa(q, k, v, outNew, numHeads, numKvHeads, headDim, basePos, numQueries);
        vk.Synchronize();

        // Reference: the 16-query kernel, one slice at a time on host-sliced Q.
        for (int i0 = 0; i0 < numQueries; i0 += VulkanBackend.FlashAttentionMaxQueries)
        {
            int take = Math.Min(VulkanBackend.FlashAttentionMaxQueries, numQueries - i0);
            var qs = vk.Upload(qHost.AsSpan(i0 * (int)qDim, take * (int)qDim).ToArray(), TensorShape.D1(take * (long)qDim), exact: true);
            var os = vk.Allocate(TensorShape.D1(take * (long)qDim));
            vk.AttentionBatchedFlash(qs, k, v, os, numHeads, numKvHeads, headDim, basePos + (uint)i0, take, maxSeq);
            vk.Synchronize();
            var part = new float[take * qDim];
            vk.Download(os, part);
            vk.Free(qs); vk.Free(os);
            // stash into outRef via host copy below
            Array.Copy(part, 0, _refHost ??= new float[numQueries * qDim], i0 * (int)qDim, part.Length);
        }
        var a = _refHost!; _refHost = null;
        var b = new float[numQueries * qDim];
        vk.Download(outNew, b);
        double maxAbs = 0, maxRef = 0;
        for (int i = 0; i < a.Length; i++)
        {
            Assert.False(float.IsNaN(b[i]) || float.IsInfinity(b[i]), $"non-finite at {i}");
            maxAbs = Math.Max(maxAbs, Math.Abs(a[i] - b[i]));
            maxRef = Math.Max(maxRef, Math.Abs(a[i]));
        }
        Assert.True(maxAbs <= 2e-5 * (maxRef + 1.0), $"heads={numHeads} kv={numKvHeads} base={basePos} q={numQueries}: maxAbsDiff {maxAbs:E2} (max|ref| {maxRef:E2})");
    }

    [ThreadStatic] private static float[]? _refHost;
}
