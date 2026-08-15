using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public unsafe class MicroGemmQ4KTests
{
    private const int BytesPerBlock = 144;
    private const int ElementsPerBlock = 256;

    /// <summary>
    /// Builds one synthetic Q4_K block (144 bytes) with deterministic, varied scale/min bytes
    /// across all 12 positions -- specifically exercising sub-block indices 4-7, whose 6-bit
    /// scale/min values are cross-byte-spliced from bytes that indices 0-3 don't touch (see
    /// Dequantize.GetScaleMinK4 / ggml's get_scale_min_k4). A uniform-byte fill would pass even
    /// with the wrong unpacking, since sc/mn would coincidentally match; varied bytes make a
    /// wrong read produce a different value than the correct splice.
    /// </summary>
    private static byte[] MakeBlock(int seed)
    {
        var block = new byte[BytesPerBlock];

        // d = 1.0, dmin = 0.5 (fp16), fixed so the test isolates the scale/min unpack path.
        block[0] = 0x00; block[1] = 0x3C; // 1.0
        block[2] = 0x00; block[3] = 0x38; // 0.5

        for (int i = 0; i < 12; i++)
            block[4 + i] = (byte)((seed * 17 + i * 37 + 11) % 256);

        for (int i = 0; i < 128; i++)
            block[16 + i] = (byte)((seed * 23 + i * 13 + 5) % 256);

        return block;
    }

    private static float[] Dequant(byte[] block)
    {
        var dst = new float[ElementsPerBlock];
        Dequantize.ToFloat32(block, dst, DType.Q4_K, ElementsPerBlock);
        return dst;
    }

    [Fact]
    public void TryMatMulQ4K_MatchesNaiveMatMulAgainstDequantizedReference()
    {
        MicroGemmConfig.IsEnabled = true;
        try
        {
            const int batchSize = 2;
            const int rows = 3;
            const int cols = ElementsPerBlock; // exactly one block per row

            var weightBlocks = new byte[rows][];
            var dequantRows = new float[rows][];
            for (int r = 0; r < rows; r++)
            {
                weightBlocks[r] = MakeBlock(seed: r + 1);
                dequantRows[r] = Dequant(weightBlocks[r]);
            }

            var weights = new byte[rows * BytesPerBlock];
            for (int r = 0; r < rows; r++)
                weightBlocks[r].CopyTo(weights, r * BytesPerBlock);

            var input = new float[batchSize * cols];
            for (int i = 0; i < input.Length; i++)
                input[i] = (i % 13 - 6) * 0.1f;

            var expected = new float[batchSize * rows];
            for (int m = 0; m < batchSize; m++)
                for (int r = 0; r < rows; r++)
                {
                    float sum = 0f;
                    for (int c = 0; c < cols; c++)
                        sum += input[m * cols + c] * dequantRows[r][c];
                    expected[m * rows + r] = sum;
                }

            var actual = new float[batchSize * rows];
            fixed (float* pInput = input)
            fixed (byte* pWeights = weights)
            fixed (float* pActual = actual)
            {
                bool executed = MicroGemmQ4K.TryMatMulQ4K(pActual, pInput, pWeights, batchSize, rows, cols);
                Assert.True(executed, "TryMatMulQ4K should execute for batchSize<=16 with cols a multiple of 256.");
            }

            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], actual[i], precision: 3);
        }
        finally
        {
            MicroGemmConfig.IsEnabled = false;
        }
    }

    [Fact]
    public void TryMatMulQ4K_DisabledByDefault()
    {
        MicroGemmConfig.IsEnabled = false;

        var input = new float[ElementsPerBlock];
        var weights = new byte[BytesPerBlock];
        var output = new float[1];

        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
        fixed (float* pOutput = output)
        {
            bool executed = MicroGemmQ4K.TryMatMulQ4K(pOutput, pInput, pWeights, 1, 1, ElementsPerBlock);
            Assert.False(executed, "TryMatMulQ4K must not execute when MicroGemmConfig.IsEnabled is false.");
        }
    }
}
