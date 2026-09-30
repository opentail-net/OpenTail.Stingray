namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed unsafe class SimdKernelsQ5KQ8KTests
{
    private static byte[] BuildQ5KRow(int cols, Random random)
    {
        var row = new byte[(cols / 256) * 176];
        for (int block = 0; block < cols / 256; block++)
        {
            int offset = block * 176;
            WriteHalf(row, offset, (float)(random.NextDouble() * 0.09 + 0.01));
            WriteHalf(row, offset + 2, (float)(random.NextDouble() * 0.02 + 0.001));
            for (int i = 4; i < 176; i++)
                row[offset + i] = (byte)random.Next(256);
        }
        return row;
    }

    private static void WriteHalf(byte[] destination, int offset, float value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((Half)value);
        destination[offset] = (byte)bits;
        destination[offset + 1] = (byte)(bits >> 8);
    }

    private static void UnpackScaleMin(byte* packed, int index, out int scale, out int min)
    {
        if (index < 4)
        {
            scale = packed[index] & 63;
            min = packed[index + 4] & 63;
        }
        else
        {
            scale = (packed[index + 4] & 15) | ((packed[index - 4] >> 6) << 4);
            min = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4);
        }
    }

    private static float GgmlScalarReference(byte* row, byte* q8k, int cols)
    {
        int blocks = cols / 256;
        float* inputScales = (float*)q8k;
        sbyte* inputQuants = (sbyte*)(q8k + blocks * 4);
        short* inputSums = (short*)(q8k + blocks * 4 + blocks * 256);
        Span<float> laneSums = stackalloc float[8];
        Span<int> laneDots = stackalloc int[8];
        Span<byte> q5 = stackalloc byte[256];
        float result = 0f;

        for (int block = 0; block < blocks; block++)
        {
            byte* weights = row + block * 176;
            byte* scalesMins = weights + 4;
            byte* qh = weights + 16;
            byte* ql = weights + 48;
            sbyte* q8 = inputQuants + block * 256;
            short* bsums = inputSums + block * 16;

            for (int group = 0; group < 4; group++)
            {
                int bitLo = group * 2;
                int bitHi = bitLo + 1;
                for (int i = 0; i < 32; i++)
                {
                    byte packed = ql[group * 32 + i];
                    q5[group * 64 + i] = (byte)((packed & 15) + (((qh[i] >> bitLo) & 1) << 4));
                    q5[group * 64 + 32 + i] = (byte)((packed >> 4) + (((qh[i] >> bitHi) & 1) << 4));
                }
            }

            laneDots.Clear();
            for (int subBlock = 0; subBlock < 8; subBlock++)
            {
                UnpackScaleMin(scalesMins, subBlock, out int scale, out _);
                for (int i = 0; i < 32; i++)
                {
                    int index = subBlock * 32 + i;
                    laneDots[i & 7] += scale * q5[index] * q8[index];
                }
            }

            float d = (float)BitConverter.UInt16BitsToHalf((ushort)(weights[0] | (weights[1] << 8))) * inputScales[block];
            float dmin = (float)BitConverter.UInt16BitsToHalf((ushort)(weights[2] | (weights[3] << 8))) * inputScales[block];
            for (int lane = 0; lane < 8; lane++)
                laneSums[lane] += d * laneDots[lane];

            int minDot = 0;
            for (int subBlock = 0; subBlock < 8; subBlock++)
            {
                UnpackScaleMin(scalesMins, subBlock, out _, out int min);
                minDot += min * (bsums[subBlock * 2] + bsums[subBlock * 2 + 1]);
            }
            result -= dmin * minDot;
        }

        for (int lane = 0; lane < 8; lane++)
            result += laneSums[lane];
        return result;
    }

    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    [InlineData(2048)]
    [InlineData(4096)]
    public void DotQ5K_Q8K_MatchesGgmlScalarFormula(int cols)
    {
        const float absoluteTolerance = 2e-4f;
        const float relativeTolerance = 2e-5f;

        for (int seed = 0; seed < 8; seed++)
        {
            var random = new Random(20260929 + cols * 17 + seed);
            byte[] row = BuildQ5KRow(cols, random);
            var input = new float[cols];
            for (int i = 0; i < cols; i++)
                input[i] = (float)(random.NextDouble() * 2.0 - 1.0);

            var scratch = new byte[SimdKernels.Q8KScratchBytes(cols)];
            fixed (byte* rowPtr = row)
            fixed (float* inputPtr = input)
            fixed (byte* scratchPtr = scratch)
            {
                SimdKernels.QuantizeRowToQ8K(inputPtr, cols, scratchPtr);
                float expected = GgmlScalarReference(rowPtr, scratchPtr, cols);
                float dispatched = SimdKernels.DotQ5K_Q8K(rowPtr, scratchPtr, cols);
                float scalar = SimdKernels.DotQ5K_Q8K_Scalar(rowPtr, scratchPtr, cols);
                float floatInputOverload = SimdKernels.DotQ5K_Q8K(rowPtr, inputPtr, cols);
                float f32Activation = SimdKernels.DotQ5K(rowPtr, inputPtr, cols);
                float tolerance = absoluteTolerance + MathF.Abs(expected) * relativeTolerance;

                Assert.True(MathF.Abs(dispatched - expected) <= tolerance,
                    $"dispatched mismatch cols={cols}, seed={seed}, expected={expected:R}, actual={dispatched:R}, abs={MathF.Abs(dispatched - expected):R}, tol={tolerance:R}");
                Assert.True(MathF.Abs(scalar - expected) <= tolerance,
                    $"scalar mismatch cols={cols}, seed={seed}, expected={expected:R}, actual={scalar:R}, abs={MathF.Abs(scalar - expected):R}, tol={tolerance:R}");
                Assert.Equal(BitConverter.SingleToInt32Bits(dispatched), BitConverter.SingleToInt32Bits(floatInputOverload));
                Assert.True(float.IsFinite(f32Activation));
                Assert.True(float.IsFinite(dispatched));
            }
        }
    }
}
