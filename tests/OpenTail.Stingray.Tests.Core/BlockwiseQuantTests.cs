using System;
using OpenTail.Stingray.Cpu;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public sealed class BlockwiseQuantTests
{
    [Fact]
    public void DequantizeNf4_MatchesCodebookValues()
    {
        // 2 bytes = 4 nibbles: (0, 15), (7, 8)
        byte[] packed = [(0 << 4) | 15, (7 << 4) | 8];
        float[] absmax = [2.0f];
        float[] dst = new float[4];

        BlockwiseQuant.DequantizeNf4Blockwise(packed, absmax, dst, 4, blockSize: 64);

        // Map[0] = -1.0 * 2.0 = -2.0
        // Map[15] = 1.0 * 2.0 = 2.0
        // Map[7] = 0.0 * 2.0 = 0.0
        // Map[8] = 0.0795803 * 2.0 = 0.1591606
        Assert.Equal(-2.0f, dst[0], tolerance: 1e-5f);
        Assert.Equal(2.0f, dst[1], tolerance: 1e-5f);
        Assert.Equal(0.0f, dst[2], tolerance: 1e-5f);
        Assert.Equal(0.1591606f, dst[3], tolerance: 1e-5f);
    }

    [Fact]
    public void MatVecNf4_MatchesDequantizedDotProduct()
    {
        int rows = 2;
        int cols = 64;
        int bytesPerRow = cols / 2;
        var packed = new byte[rows * bytesPerRow];
        for (int i = 0; i < packed.Length; i++) packed[i] = (byte)((i % 16) << 4 | ((i + 3) % 16));

        float[] absmax = [1.5f, 0.8f];
        var input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = (i + 1) * 0.1f;

        // Fused MatVec
        var fusedOut = new float[rows];
        BlockwiseQuant.MatVecNf4(packed, absmax, input, fusedOut, rows, cols, blockSize: 64);

        // Dequantize then compute standard dot product
        var dequantWeights = new float[rows * cols];
        BlockwiseQuant.DequantizeNf4Blockwise(
            packed.AsSpan(0, bytesPerRow), absmax.AsSpan(0, 1), dequantWeights.AsSpan(0, cols), cols, blockSize: 64);
        BlockwiseQuant.DequantizeNf4Blockwise(
            packed.AsSpan(bytesPerRow, bytesPerRow), absmax.AsSpan(1, 1), dequantWeights.AsSpan(cols, cols), cols, blockSize: 64);

        float expectedRow0 = 0.0f;
        float expectedRow1 = 0.0f;
        for (int i = 0; i < cols; i++)
        {
            expectedRow0 += dequantWeights[i] * input[i];
            expectedRow1 += dequantWeights[cols + i] * input[i];
        }

        Assert.Equal(expectedRow0, fusedOut[0], tolerance: 1e-4f);
        Assert.Equal(expectedRow1, fusedOut[1], tolerance: 1e-4f);
    }

    [Fact]
    public void DequantizeNBits4_MatchesAwqGptqValues()
    {
        // 2 bytes = 4 nibbles: (0, 8), (15, 7)
        byte[] packed = [(8 << 4) | 0, (7 << 4) | 15]; // low nibble first
        float[] scales = [0.5f];
        byte[] zp = [8]; // zero point = 8
        float[] dst = new float[4];

        BlockwiseQuant.DequantizeNBits4(packed, scales, zp, dst, rows: 1, cols: 4, blockSize: 32);

        // (0 - 8) * 0.5 = -4.0
        // (8 - 8) * 0.5 = 0.0
        // (15 - 8) * 0.5 = 3.5
        // (7 - 8) * 0.5 = -0.5
        Assert.Equal(-4.0f, dst[0], tolerance: 1e-5f);
        Assert.Equal(0.0f, dst[1], tolerance: 1e-5f);
        Assert.Equal(3.5f, dst[2], tolerance: 1e-5f);
        Assert.Equal(-0.5f, dst[3], tolerance: 1e-5f);
    }

    [Fact]
    public void MatVecNBits4_MatchesDequantizedDotProduct()
    {
        int rows = 2;
        int cols = 32;
        int bytesPerRow = cols / 2;
        var packed = new byte[rows * bytesPerRow];
        for (int i = 0; i < packed.Length; i++) packed[i] = (byte)(i % 255);

        float[] scales = [0.25f, 0.5f];
        byte[] zp = [8, 8];
        var input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = (i + 1) * 0.05f;

        var fusedOut = new float[rows];
        BlockwiseQuant.MatVecNBits4(packed, scales, zp, input, fusedOut, rows, cols, blockSize: 32);

        var dequant = new float[rows * cols];
        BlockwiseQuant.DequantizeNBits4(packed, scales, zp, dequant, rows, cols, blockSize: 32);

        for (int r = 0; r < rows; r++)
        {
            float expected = 0.0f;
            for (int c = 0; c < cols; c++)
            {
                expected += dequant[r * cols + c] * input[c];
            }
            Assert.Equal(expected, fusedOut[r], tolerance: 1e-4f);
        }
    }

    [Fact]
    public void SkipLayerNorm_ComputesExactResidualAndNormalizedValues()
    {
        int rows = 1;
        int hidden = 4;
        float[] input = [1.0f, 2.0f, 3.0f, 4.0f];
        float[] skip = [0.5f, 0.5f, 0.5f, 0.5f];
        float[] gamma = [1.0f, 1.0f, 1.0f, 1.0f];
        float[] beta = [0.0f, 0.0f, 0.0f, 0.0f];
        float[] bias = [];
        float[] output = new float[hidden];
        float[] resOut = new float[hidden];

        SkipLayerNorm.Compute(input, skip, gamma, beta, bias, output, resOut, rows, hidden);

        // input + skip = [1.5, 2.5, 3.5, 4.5]
        // mean = 3.0, var = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25, std = sqrt(1.25 + 1e-5) = 1.11803
        Assert.Equal(1.5f, resOut[0], tolerance: 1e-5f);
        Assert.Equal(4.5f, resOut[3], tolerance: 1e-5f);

        float expectedNorm0 = (1.5f - 3.0f) / MathF.Sqrt(1.25f + 1e-5f);
        Assert.Equal(expectedNorm0, output[0], tolerance: 1e-4f);
    }

    [Fact]
    public void SkipRmsNorm_ComputesExactResidualAndNormalizedValues()
    {
        int rows = 1;
        int hidden = 4;
        float[] input = [1.0f, 2.0f, 3.0f, 4.0f];
        float[] skip = [1.0f, 1.0f, 1.0f, 1.0f];
        float[] gamma = [2.0f, 2.0f, 2.0f, 2.0f];
        float[] output = new float[hidden];
        float[] resOut = new float[hidden];

        SkipLayerNorm.ComputeRmsNorm(input, skip, gamma, output, resOut, rows, hidden);

        // input + skip = [2, 3, 4, 5]
        // sumSq = 4 + 9 + 16 + 25 = 54, meanSq = 54 / 4 = 13.5, rms = sqrt(13.5 + 1e-5) = 3.67423
        Assert.Equal(2.0f, resOut[0], tolerance: 1e-5f);
        Assert.Equal(5.0f, resOut[3], tolerance: 1e-5f);

        float expectedNorm0 = (2.0f / MathF.Sqrt(13.5f + 1e-5f)) * 2.0f;
        Assert.Equal(expectedNorm0, output[0], tolerance: 1e-4f);
    }
}
