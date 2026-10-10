using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// The GQA-group split-KV decode attention kernel (<c>VulkanBackend.AttentionSplitKvGqa</c>) against the original one-key-per-thread
/// split-KV kernel (<c>AttentionSplitKv</c>) on random fp32 K/V caches. Both use online-softmax partials and the same combine pass, so they
/// agree to floating-point reordering (about 1e-7 here). Covers context lengths around the 256-key split boundary, group sizes 1, 2 and 4,
/// head dimensions 64 and 128, and a sliding-window case.
/// </summary>
public sealed class VulkanSplitKvGqaAttentionTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    [Theory]
    [InlineData(32u, 8u, 128u)]   // group 4, Qwen3-4B / Llama-3 style
    [InlineData(16u, 8u, 128u)]   // group 2
    [InlineData(8u, 8u, 128u)]    // group 1 (MHA)
    [InlineData(32u, 8u, 64u)]    // group 4, head dim 64
    [InlineData(12u, 4u, 64u)]    // group 3, head dim 64
    public void MatchesOriginalSplitKvKernel(uint numHeads, uint numKvHeads, uint headDim)
    {
        using var vk = TryCreateVulkan();
        Assert.SkipUnless(vk != null, "no Vulkan device");
        Assert.SkipUnless(VulkanBackend.SupportsSplitKvGqa(numHeads, numKvHeads, headDim), "shape not served by the GQA kernel");

        const uint maxSeq = 2560;
        uint kvDim = numKvHeads * headDim;
        var rnd = new Random(31 + (int)numHeads);
        var kHost = new float[maxSeq * kvDim];
        var vHost = new float[maxSeq * kvDim];
        for (int i = 0; i < kHost.Length; i++) { kHost[i] = (float)(rnd.NextDouble() * 2 - 1); vHost[i] = (float)(rnd.NextDouble() * 2 - 1); }
        var qHost = new float[numHeads * headDim];
        for (int i = 0; i < qHost.Length; i++) qHost[i] = (float)(rnd.NextDouble() * 2 - 1);

        var k = vk!.Upload(kHost, TensorShape.D1(kHost.Length), exact: true);
        var v = vk.Upload(vHost, TensorShape.D1(vHost.Length), exact: true);
        var q = vk.Upload(qHost, TensorShape.D1(qHost.Length), exact: true);
        uint maxSplits = (maxSeq + VulkanBackend.SplitKvChunk - 1) / VulkanBackend.SplitKvChunk;
        var partialO = vk.Allocate(TensorShape.D1((long)numHeads * maxSplits * headDim));
        var partialMeta = vk.Allocate(TensorShape.D1((long)numHeads * maxSplits * 2));
        var outOld = vk.Allocate(TensorShape.D1(numHeads * headDim));
        var outNew = vk.Allocate(TensorShape.D1(numHeads * headDim));

        foreach (var (seq, window) in new[] { (1u, 0u), (17u, 0u), (255u, 0u), (256u, 0u), (257u, 0u), (500u, 0u), (1000u, 0u), (2048u, 0u), (1000u, 300u) })
        {
            vk.AttentionSplitKv(q, k, v, outOld, partialO, partialMeta, numHeads, numKvHeads, headDim, seq, maxSeq, window);
            vk.Synchronize();
            vk.AttentionSplitKvGqa(q, k, v, outNew, partialO, partialMeta, numHeads, numKvHeads, headDim, seq, maxSeq, window);
            vk.Synchronize();
            var a = new float[numHeads * headDim];
            var b = new float[numHeads * headDim];
            vk.Download(outOld, a);
            vk.Download(outNew, b);
            double maxAbs = 0, maxRef = 0;
            for (int i = 0; i < a.Length; i++)
            {
                Assert.False(float.IsNaN(b[i]) || float.IsInfinity(b[i]), $"heads={numHeads} kv={numKvHeads} hd={headDim} seq={seq} window={window}: non-finite at {i}");
                maxAbs = Math.Max(maxAbs, Math.Abs(a[i] - b[i]));
                maxRef = Math.Max(maxRef, Math.Abs(a[i]));
            }
            Assert.True(maxAbs <= 1e-5 * (maxRef + 1.0),
                $"heads={numHeads} kv={numKvHeads} hd={headDim} seq={seq} window={window}: maxAbsDiff {maxAbs:E2} (max|ref| {maxRef:E2})");
        }
    }

    [Fact]
    public void SupportCheck_AcceptsGroupsOneToFourAndHeadDim64Or128Only()
    {
        Assert.True(VulkanBackend.SupportsSplitKvGqa(32, 8, 128));
        Assert.True(VulkanBackend.SupportsSplitKvGqa(8, 8, 64));
        Assert.True(VulkanBackend.SupportsSplitKvGqa(12, 4, 64));      // group 3
        Assert.False(VulkanBackend.SupportsSplitKvGqa(40, 8, 128));    // group 5
        Assert.False(VulkanBackend.SupportsSplitKvGqa(28, 4, 128));    // group 7
        Assert.False(VulkanBackend.SupportsSplitKvGqa(32, 8, 96));     // head dim 96
        Assert.False(VulkanBackend.SupportsSplitKvGqa(28, 8, 128));    // not a whole number of query heads per KV head
    }
}
