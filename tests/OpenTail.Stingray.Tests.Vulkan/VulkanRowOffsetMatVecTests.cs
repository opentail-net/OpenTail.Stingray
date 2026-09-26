using System.Runtime.InteropServices;

namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// The raw-quant matvecs added for gpt-oss (MXFP4) and DeepSeek-V2-Lite Q2_K (Q2_K, IQ4_NL) against
/// the CPU reference (<see cref="Dequantize.ToFloat32"/> then a plain dot product), on random blocks.
/// The weight is a stack of three "experts" and the middle one is addressed by row offset, as the
/// MoE passes do; rows are not a multiple of the shader's 8-row workgroup.
/// </summary>
public sealed class VulkanRowOffsetMatVecTests
{
    [Theory]
    [InlineData(DType.MXFP4, 32, 17, 64)]
    [InlineData(DType.IQ4_NL, 32, 18, 96)]
    [InlineData(DType.Q2_K, 256, 84, 512)]
    public void MatchesCpuDequantDot(DType dtype, int blockElems, int blockBytes, int cols)
    {
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var _ = gpu;

        const int rows = 13, experts = 3, expert = 1;
        int totalRows = rows * experts;
        int blocksPerRow = cols / blockElems;
        var rng = new Random(4242 + (int)dtype);
        var bytes = new byte[totalRows * blocksPerRow * blockBytes];
        rng.NextBytes(bytes);
        for (int b = 0; b < totalRows * blocksPerRow; b++)
        {
            int o = b * blockBytes;
            switch (dtype)
            {
                case DType.MXFP4: bytes[o] = (byte)rng.Next(118, 130); break;          // E8M0 scale near 1
                case DType.IQ4_NL: PutHalf(bytes, o, 0.01f + 0.02f * rng.NextSingle()); break;
                case DType.Q2_K:
                    PutHalf(bytes, o + 80, 0.01f + 0.02f * rng.NextSingle());            // d
                    PutHalf(bytes, o + 82, 0.005f + 0.01f * rng.NextSingle());           // dmin
                    break;
            }
        }
        var input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = rng.NextSingle() * 2 - 1;

        var dense = new float[totalRows * cols];
        Dequantize.ToFloat32(bytes, dense, dtype, dense.Length);
        var expected = new float[rows];
        for (int r = 0; r < rows; r++)
        {
            double sum = 0;
            for (int c = 0; c < cols; c++) sum += (double)dense[(expert * rows + r) * cols + c] * input[c];
            expected[r] = (float)sum;
        }

        var raw = new float[(bytes.Length + 3) / 4];
        bytes.CopyTo(MemoryMarshal.AsBytes(raw.AsSpan()));
        var w = gpu!.Upload(raw, TensorShape.D1(raw.Length));
        var x = gpu.Upload(input, TensorShape.D1(cols));
        var y = gpu.Allocate(TensorShape.D1(rows));
        gpu.MatVecRowOffset(y, w, x, cols, expert * rows, dtype);
        var actual = new float[rows];
        gpu.Download(y, actual);

        for (int r = 0; r < rows; r++)
            Assert.True(Math.Abs(actual[r] - expected[r]) <= 1e-3f * (1 + Math.Abs(expected[r])),
                $"{dtype} row {r}: GPU {actual[r]} vs CPU {expected[r]}");
    }

    private static void PutHalf(byte[] b, int o, float v)
    {
        ushort h = BitConverter.HalfToUInt16Bits((Half)v);
        b[o] = (byte)h;
        b[o + 1] = (byte)(h >> 8);
    }
}
