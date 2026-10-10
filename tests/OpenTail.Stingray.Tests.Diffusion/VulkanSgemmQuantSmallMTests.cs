using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// The small-M quantized GEMM (16 activation rows x 64 weight rows per workgroup, used for M up to
/// <c>STINGRAY_VK_QGEMM_SMALLM</c>, default 32) must give exactly the same result as the large-tile kernel, because
/// both accumulate every output element over K in the same order. Synthetic data, no model files needed:
/// the same activation rows are run once at their real M (small kernel) and once zero-padded to M=128 (large kernel),
/// and the first M rows are compared bit for bit. Skips when no Vulkan device is present, or when the small-M path
/// has been switched off by the environment (then both runs use the large kernel and the comparison is vacuous).
/// </summary>
public sealed class VulkanSgemmQuantSmallMTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    // Block bytes and the offset of the two fp16 scale fields that must stay finite for random payload bytes.
    private static (int blockBytes, int[] halfOffsets) Layout(DType dt) => dt switch
    {
        DType.Q3_K => (110, [108]),
        DType.Q4_K => (144, [0, 2]),
        DType.Q5_K => (176, [0, 2]),
        _ => throw new ArgumentOutOfRangeException(nameof(dt)),
    };

    [Theory]
    [InlineData(DType.Q3_K, 3072, 3072)]
    [InlineData(DType.Q3_K, 12288, 768)]   // long K, as the Qwen MLP-down projection
    [InlineData(DType.Q4_K, 3072, 1000)]   // N not a multiple of the 64-row tile
    [InlineData(DType.Q5_K, 1024, 512)]
    public void SmallMKernel_MatchesLargeKernelExactly(DType dtype, int k, int n)
    {
        using var vk = TryCreateVulkan();
        Assert.SkipUnless(vk != null, "no Vulkan device");
        Assert.SkipUnless(Environment.GetEnvironmentVariable("STINGRAY_VK_QGEMM_SMALLM") != "0",
            "small-M path disabled by STINGRAY_VK_QGEMM_SMALLM=0");
        Assert.SkipUnless(vk!.SupportsQuantizedSgemm, "backend has no quantized Sgemm");

        var (blockBytes, halfOffsets) = Layout(dtype);
        var rnd = new Random(1234);
        long bytes = (long)n * (k / 256) * blockBytes;
        var raw = new byte[bytes];
        rnd.NextBytes(raw);
        for (long b = 0; b < bytes; b += blockBytes)
            foreach (int off in halfOffsets) { raw[b + off] = 0x00; raw[b + off + 1] = 0x3C; }   // fp16 1.0
        var weights = vk.UploadRaw(raw, TensorShape.D2(n, k), dtype);

        foreach (int m in new[] { 2, 5, 12, 16, 17, 32, 33 })
        {
            var act = new float[128 * k];                                   // rows >= m stay zero
            for (int i = 0; i < m * k; i++) act[i] = (float)(rnd.NextDouble() - 0.5);
            var a = vk.Upload(act, TensorShape.D2(128, k), exact: true);
            var cSmall = vk.Upload(new float[128 * n], TensorShape.D2(128, n), exact: true);
            var cLarge = vk.Upload(new float[128 * n], TensorShape.D2(128, n), exact: true);

            vk.Sgemm(cSmall, a, weights, m, k, n);          // routed to the small-M kernel when m <= 32
            vk.Sgemm(cLarge, a, weights, 128, k, n);        // M=128 always takes the large-tile kernel
            vk.Synchronize();

            var small = new float[128 * n]; var large = new float[128 * n];
            vk.Download(cSmall, small); vk.Download(cLarge, large);
            for (int i = 0; i < m * n; i++)
            {
                Assert.False(float.IsNaN(small[i]) || float.IsInfinity(small[i]), $"{dtype} M={m}: non-finite at {i}");
                Assert.True(small[i] == large[i], $"{dtype} K={k} N={n} M={m}: element {i} differs ({small[i]} vs {large[i]})");
            }
            vk.Free(a); vk.Free(cSmall); vk.Free(cLarge);
        }
        vk.Free(weights);
    }
}
