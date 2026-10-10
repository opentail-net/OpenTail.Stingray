using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// <c>VulkanBackend.SgemmQuant</c> (the prefill-sized tiled quantized GEMM: Q3_K, Q4_K, Q5_K, Q6_K) against the CPU dequantizer with a
/// double-precision accumulation. Synthetic blocks, no model file. M values straddle the small-tile / large-tile cutoff (32) and N is not
/// a multiple of either tile width (64 / 256).
/// </summary>
public sealed class VulkanSgemmQuantEntryTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static (int bytes, int[] halves) Layout(DType dt) => dt switch
    {
        DType.Q3_K => (110, [108]),
        DType.Q4_K => (144, [0, 2]),
        DType.Q5_K => (176, [0, 2]),
        DType.Q6_K => (210, [208]),
        _ => throw new ArgumentOutOfRangeException(nameof(dt)),
    };

    [Theory]
    [InlineData(DType.Q3_K, 768)]
    [InlineData(DType.Q4_K, 768)]
    [InlineData(DType.Q5_K, 768)]
    [InlineData(DType.Q6_K, 768)]
    [InlineData(DType.Q6_K, 2560)]
    [InlineData(DType.Q4_K, 3072)]
    public void SgemmQuant_MatchesCpuDequant(DType dtype, int k)
    {
        using var vk = TryCreateVulkan();
        Assert.SkipUnless(vk != null, "no Vulkan device");

        const int n = 130;
        var (bytes, halves) = Layout(dtype);
        int blocksPerRow = k / 256;
        var rnd = new Random(4321 + (int)dtype);
        var raw = new byte[(long)n * blocksPerRow * bytes];
        rnd.NextBytes(raw);
        for (long b = 0; b < raw.LongLength; b += bytes)
            foreach (int h in halves) { raw[b + h] = 0x00; raw[b + h + 1] = 0x3C; }   // fp16 1.0

        // Reference weights, dequantized once on the CPU.
        var w = new float[(long)n * k];
        for (int r = 0; r < n; r++)
            Dequantize.ToFloat32(new ReadOnlySpan<byte>(raw, r * blocksPerRow * bytes, blocksPerRow * bytes),
                w.AsSpan(r * k, k), dtype, k);

        var weights = vk!.UploadRaw(raw, TensorShape.D2(n, k), dtype);
        foreach (int m in new[] { 1, 5, 32, 33, 130 })
        {
            var act = new float[m * k];
            for (int i = 0; i < act.Length; i++) act[i] = (float)(rnd.NextDouble() - 0.5);
            var a = vk.Upload(act, TensorShape.D2(m, k), exact: true);
            var c = vk.Upload(new float[m * n], TensorShape.D2(m, n), exact: true);
            vk.SgemmQuant(c, a, weights, m, k, n);
            vk.Synchronize();
            var gpu = new float[m * n];
            vk.Download(c, gpu);

            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double dot = 0, mag = 0;
                    for (int t = 0; t < k; t++) { double term = (double)act[i * k + t] * w[(long)j * k + t]; dot += term; mag += Math.Abs(term); }
                    float g = gpu[i * n + j];
                    Assert.False(float.IsNaN(g) || float.IsInfinity(g), $"{dtype} M={m} ({i},{j}): non-finite");
                    Assert.True(Math.Abs(g - dot) <= 1e-5 * (mag + 1.0), $"{dtype} K={k} M={m} ({i},{j}): gpu {g} vs cpu {dot} (|terms| {mag})");
                }
            vk.Free(a); vk.Free(c);
        }
        vk.Free(weights);
    }
}
