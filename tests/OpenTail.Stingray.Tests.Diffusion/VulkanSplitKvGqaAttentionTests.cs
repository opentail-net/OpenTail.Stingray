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
    [InlineData(16u, 4u, 256u)]   // group 4, head dim 256 (Gemma family)
    [InlineData(8u, 8u, 256u)]    // group 1, head dim 256
    [InlineData(40u, 8u, 128u)]   // group 5 (two workgroups per KV head: 4 + 1)
    [InlineData(24u, 4u, 128u)]   // group 6 (4 + 2)
    [InlineData(28u, 4u, 128u)]   // group 7, Qwen2.5-7B style (4 + 3)
    [InlineData(64u, 8u, 128u)]   // group 8, gpt-oss style (4 + 4)
    [InlineData(32u, 4u, 256u)]   // group 8, head dim 256 (hybrid Qwen3.5 style)
    [InlineData(8u, 2u, 256u)]    // group 4, head dim 256 (Gemma 4 sliding-window layers)
    [InlineData(8u, 2u, 512u)]    // group 4, head dim 512 (Gemma 4 global layers)
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


    [Theory]
    [InlineData(64u, 8u, 64u)]    // gpt-oss: 64 query heads, 8 KV heads, head dim 64
    [InlineData(32u, 8u, 128u)]
    public void WithSinksMatchesSinksKernel(uint numHeads, uint numKvHeads, uint headDim)
    {
        using var vk = TryCreateVulkan();
        Assert.SkipUnless(vk != null, "no Vulkan device");
        Assert.SkipUnless(VulkanBackend.SupportsSplitKvGqa(numHeads, numKvHeads, headDim), "shape not served by the GQA kernel");

        const uint maxSeq = 2560;
        uint kvDim = numKvHeads * headDim;
        var rnd = new Random(99 + (int)numHeads);
        float[] Rand(long n, double scale = 1.0) { var a = new float[n]; for (long i = 0; i < n; i++) a[i] = (float)((rnd.NextDouble() * 2 - 1) * scale); return a; }
        var k = vk!.Upload(Rand((long)maxSeq * kvDim), TensorShape.D1((long)maxSeq * kvDim), exact: true);
        var v = vk.Upload(Rand((long)maxSeq * kvDim), TensorShape.D1((long)maxSeq * kvDim), exact: true);
        var q = vk.Upload(Rand(numHeads * headDim), TensorShape.D1(numHeads * headDim), exact: true);
        var sinksHost = Rand(numHeads, 3.0);
        var sinks = vk.Upload(sinksHost, TensorShape.D1(numHeads), exact: true);
        uint maxSplits = (maxSeq + VulkanBackend.SplitKvChunk - 1) / VulkanBackend.SplitKvChunk;
        var partialO = vk.Allocate(TensorShape.D1((long)numHeads * maxSplits * headDim));
        var partialMeta = vk.Allocate(TensorShape.D1((long)numHeads * maxSplits * 2));
        var scratch = vk.Allocate(TensorShape.D1((long)numHeads * maxSeq));
        var outOld = vk.Allocate(TensorShape.D1(numHeads * headDim));
        var outNew = vk.Allocate(TensorShape.D1(numHeads * headDim));

        foreach (var (seq, window) in new[] { (1u, 0u), (17u, 0u), (257u, 0u), (1000u, 0u), (2048u, 0u), (1000u, 128u), (300u, 128u) })
        {
            vk.AttentionSinks(q, k, v, outOld, scratch, sinks, numHeads, numKvHeads, headDim, seq, maxSeq, window);
            vk.Synchronize();
            vk.AttentionSplitKvGqa(q, k, v, outNew, partialO, partialMeta, numHeads, numKvHeads, headDim, seq, maxSeq, window, sinks);
            vk.Synchronize();
            var a = new float[numHeads * headDim];
            var b = new float[numHeads * headDim];
            vk.Download(outOld, a);
            vk.Download(outNew, b);
            double maxAbs = 0, maxRef = 0;
            for (int i = 0; i < a.Length; i++)
            {
                Assert.False(float.IsNaN(b[i]) || float.IsInfinity(b[i]), $"seq={seq} window={window}: non-finite at {i}");
                maxAbs = Math.Max(maxAbs, Math.Abs(a[i] - b[i]));
                maxRef = Math.Max(maxRef, Math.Abs(a[i]));
            }
            Assert.True(maxAbs <= 1e-5 * (maxRef + 1.0), $"heads={numHeads} hd={headDim} seq={seq} window={window}: maxAbsDiff {maxAbs:E2} (max|ref| {maxRef:E2})");
        }
    }
    [Fact]
    public void SupportCheck_AcceptsGroupsOneToEightAndHeadDim64Or128Or256Only()
    {
        Assert.True(VulkanBackend.SupportsSplitKvGqa(32, 8, 128));
        Assert.True(VulkanBackend.SupportsSplitKvGqa(8, 8, 64));
        Assert.True(VulkanBackend.SupportsSplitKvGqa(12, 4, 64));      // group 3
        Assert.True(VulkanBackend.SupportsSplitKvGqa(40, 8, 128));     // group 5
        Assert.True(VulkanBackend.SupportsSplitKvGqa(64, 8, 128));     // group 8
        Assert.False(VulkanBackend.SupportsSplitKvGqa(72, 8, 128));    // group 9
        Assert.True(VulkanBackend.SupportsSplitKvGqa(16, 4, 256));     // head dim 256
        Assert.False(VulkanBackend.SupportsSplitKvGqa(32, 8, 96));     // head dim 96
        Assert.True(VulkanBackend.SupportsSplitKvGqa(8, 2, 512));      // head dim 512
        Assert.False(VulkanBackend.SupportsSplitKvGqa(8, 2, 1024));    // head dim 1024
        Assert.False(VulkanBackend.SupportsSplitKvGqa(28, 8, 128));    // not a whole number of query heads per KV head
    }
}
