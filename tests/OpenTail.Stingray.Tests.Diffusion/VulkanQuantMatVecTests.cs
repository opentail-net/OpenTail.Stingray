using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Vulkan LLM-decode mat-vec kernels (<c>VulkanBackend.MatMul(output, matrix, vector, dtype)</c>) against the CPU's own
/// dequantizer with a double-precision dot product, for every raw block-quantized type this path serves. Synthetic blocks, no
/// model file: random packed payloads with the fp16 scale fields pinned to a finite value. Covers a row count that is not a
/// multiple of the 8 rows per workgroup and block counts that are not multiples of the 32 lanes (2880 = 90 blocks of 32).
/// </summary>
public sealed class VulkanQuantMatVecTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    // bytes per block, elements per block, offsets of fp16 scale fields (pinned to 1.0)
    private static (int bytes, int elems, int[] halves) Layout(DType dt) => dt switch
    {
        DType.Q8_0 => (34, 32, [0]),
        DType.Q4_0 => (18, 32, [0]),
        DType.Q3_K => (110, 256, [108]),
        DType.Q4_K => (144, 256, [0, 2]),
        DType.Q5_K => (176, 256, [0, 2]),
        DType.Q6_K => (210, 256, [208]),
        _ => throw new ArgumentOutOfRangeException(nameof(dt)),
    };

    [Theory]
    [InlineData(DType.Q8_0, 70, 32)]
    [InlineData(DType.Q8_0, 37, 96)]
    [InlineData(DType.Q8_0, 70, 2880)]
    [InlineData(DType.Q8_0, 64, 14336)]
    [InlineData(DType.Q4_0, 70, 32)]
    [InlineData(DType.Q4_0, 37, 96)]
    [InlineData(DType.Q4_0, 70, 2880)]
    [InlineData(DType.Q4_0, 64, 14336)]
    [InlineData(DType.Q3_K, 70, 768)]
    [InlineData(DType.Q3_K, 64, 14336)]
    [InlineData(DType.Q4_K, 70, 768)]
    [InlineData(DType.Q4_K, 64, 14336)]
    [InlineData(DType.Q5_K, 70, 768)]
    [InlineData(DType.Q5_K, 64, 14336)]
    [InlineData(DType.Q6_K, 70, 768)]
    [InlineData(DType.Q6_K, 64, 14336)]
    public void MatVec_MatchesCpuDequantDot(DType dtype, int rows, int cols)
    {
        using var vk = TryCreateVulkan();
        Assert.SkipUnless(vk != null, "no Vulkan device");

        var (bytes, elems, halves) = Layout(dtype);
        int blocksPerRow = cols / elems;
        var rnd = new Random(1234 + (int)dtype);
        var raw = new byte[(long)rows * blocksPerRow * bytes];
        rnd.NextBytes(raw);
        for (long b = 0; b < raw.LongLength; b += bytes)
            foreach (int h in halves) { raw[b + h] = 0x00; raw[b + h + 1] = 0x3C; }   // fp16 1.0

        var x = new float[cols];
        for (int i = 0; i < cols; i++) x[i] = (float)(rnd.NextDouble() - 0.5);

        var weights = vk!.UploadRaw(raw, TensorShape.D2(rows, cols), dtype);
        var input = vk.Upload(x, TensorShape.D1(cols), exact: true);
        var output = vk.Upload(new float[rows], TensorShape.D1(rows), exact: true);
        vk.MatMul(output, weights, input, dtype);
        vk.Synchronize();
        var gpu = new float[rows];
        vk.Download(output, gpu);

        var rowFloats = new float[cols];
        for (int r = 0; r < rows; r++)
        {
            var rowBytes = new ReadOnlySpan<byte>(raw, r * blocksPerRow * bytes, blocksPerRow * bytes);
            Dequantize.ToFloat32(rowBytes, rowFloats, dtype, cols);
            double dot = 0, mag = 0;
            for (int i = 0; i < cols; i++) { double t = (double)rowFloats[i] * x[i]; dot += t; mag += Math.Abs(t); }
            Assert.False(float.IsNaN(gpu[r]) || float.IsInfinity(gpu[r]), $"{dtype} row {r}: non-finite");
            Assert.True(Math.Abs(gpu[r] - dot) <= 1e-5 * (mag + 1.0),
                $"{dtype} rows={rows} cols={cols} row {r}: gpu {gpu[r]} vs cpu {dot} (|terms| {mag})");
        }
        vk.Free(weights); vk.Free(input); vk.Free(output);
    }
}
