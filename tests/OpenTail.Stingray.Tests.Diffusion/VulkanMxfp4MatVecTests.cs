using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Vulkan MXFP4 mat-vec (<c>VulkanBackend.MatVecMxfp4</c>, the gpt-oss decode kernel) against the CPU's own MXFP4
/// dequantizer with a double-precision dot product. Synthetic blocks, no model file: random packed nibbles and E8M0 scales
/// in a sane range. Covers a column count whose block count is not a multiple of the 32 lanes (2880 = 90 blocks, gpt-oss's
/// hidden size), a row count that is not a multiple of the 8 rows per workgroup, and a non-zero row offset (stacked experts).
/// </summary>
public sealed class VulkanMxfp4MatVecTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    [Theory]
    [InlineData(70, 32, 0)]
    [InlineData(70, 96, 5)]
    [InlineData(37, 1024, 0)]
    [InlineData(70, 2880, 11)]
    [InlineData(64, 14336, 3)]
    public void MatVec_MatchesCpuDequantDot(int rows, int cols, int rowOffset)
    {
        using var vk = TryCreateVulkan();
        Assert.SkipUnless(vk != null, "no Vulkan device");

        int totalRows = rowOffset + rows + 3;
        int blocksPerRow = cols / 32;
        var rnd = new Random(77);
        var raw = new byte[(long)totalRows * blocksPerRow * 17];
        rnd.NextBytes(raw);
        for (long b = 0; b < raw.LongLength; b += 17) raw[b] = (byte)rnd.Next(120, 132);   // E8M0 scale about 2^-7..2^4

        var x = new float[cols];
        for (int i = 0; i < cols; i++) x[i] = (float)(rnd.NextDouble() - 0.5);

        var weights = vk!.UploadRaw(raw, TensorShape.D2(totalRows, cols), DType.MXFP4);
        var input = vk.Upload(x, TensorShape.D1(cols), exact: true);
        var output = vk.Upload(new float[rows], TensorShape.D1(rows), exact: true);
        vk.MatVecMxfp4(output, weights, input, cols, rowOffset);
        vk.Synchronize();
        var gpu = new float[rows];
        vk.Download(output, gpu);

        var rowFloats = new float[cols];
        for (int r = 0; r < rows; r++)
        {
            var rowBytes = new ReadOnlySpan<byte>(raw, (rowOffset + r) * blocksPerRow * 17, blocksPerRow * 17);
            Dequantize.ToFloat32(rowBytes, rowFloats, DType.MXFP4, cols);
            double dot = 0, mag = 0;
            for (int i = 0; i < cols; i++) { double t = (double)rowFloats[i] * x[i]; dot += t; mag += Math.Abs(t); }
            Assert.False(float.IsNaN(gpu[r]) || float.IsInfinity(gpu[r]), $"row {r}: non-finite");
            Assert.True(Math.Abs(gpu[r] - dot) <= 1e-5 * (mag + 1.0),
                $"rows={rows} cols={cols} off={rowOffset} row {r}: gpu {gpu[r]} vs cpu {dot} (|terms| {mag})");
        }
        vk.Free(weights); vk.Free(input); vk.Free(output);
    }
}
