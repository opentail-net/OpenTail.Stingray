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

    [Fact]
    public void Q5KDecodeGate_RoutesEveryMatVecShapeThroughQ8K_AndOffKeepsF32()
    {
        const int rows = 24, cols = 512;
        var random = new Random(20261001);
        int bpr = (cols / 256) * 176;
        var w1 = new byte[rows * bpr];
        var w2 = new byte[rows * bpr];
        for (int r = 0; r < rows; r++)
        {
            BuildQ5KRow(cols, random).CopyTo(w1, r * bpr);
            BuildQ5KRow(cols, random).CopyTo(w2, r * bpr);
        }
        var inputs = new float[4][];
        for (int k = 0; k < 4; k++)
        {
            inputs[k] = new float[cols];
            for (int i = 0; i < cols; i++) inputs[k][i] = (float)(random.NextDouble() * 2.0 - 1.0);
        }

        bool saved = SimdKernels.Q5KDecodeQ8KActivations;
        try
        {
            var q8k = new float[4][];
            var f32 = new float[4][];
            var outs = new float[8][];
            for (int k = 0; k < 4; k++) { q8k[k] = new float[rows]; f32[k] = new float[rows]; }
            for (int k = 0; k < 8; k++) outs[k] = new float[rows];
            var q8kW2 = new float[rows];

            fixed (byte* p1 = w1) fixed (byte* p2 = w2)
            fixed (float* i0 = inputs[0]) fixed (float* i1 = inputs[1]) fixed (float* i2 = inputs[2]) fixed (float* i3 = inputs[3])
            fixed (float* e0 = q8k[0]) fixed (float* e1 = q8k[1]) fixed (float* e2 = q8k[2]) fixed (float* e3 = q8k[3]) fixed (float* e2w = q8kW2)
            fixed (float* f0 = f32[0])
            fixed (float* o0 = outs[0]) fixed (float* o1 = outs[1]) fixed (float* o2 = outs[2]) fixed (float* o3 = outs[3]) fixed (float* o4 = outs[4])
            {
                // Expected: the explicit Q8_K matvec, once per (weights, input).
                SimdKernels.MatVecQ5K_Q8K(e0, p1, i0, rows, cols);
                SimdKernels.MatVecQ5K_Q8K(e1, p1, i1, rows, cols);
                SimdKernels.MatVecQ5K_Q8K(e2, p1, i2, rows, cols);
                SimdKernels.MatVecQ5K_Q8K(e3, p1, i3, rows, cols);
                SimdKernels.MatVecQ5K_Q8K(e2w, p2, i0, rows, cols);

                SimdKernels.Q5KDecodeQ8KActivations = true;
                SimdKernels.MatVec(o0, p1, i0, rows, cols, DType.Q5_K);
                AssertBits(q8k[0], outs[0]);

                SimdKernels.MatVecDual(o0, p1, o1, p2, i0, rows, cols, DType.Q5_K, DType.Q5_K);
                AssertBits(q8k[0], outs[0]);
                AssertBits(q8kW2, outs[1]);

                SimdKernels.MatVec2In(o0, o1, p1, i0, i1, rows, cols, DType.Q5_K);
                AssertBits(q8k[0], outs[0]);
                AssertBits(q8k[1], outs[1]);

                SimdKernels.MatVec4In(o0, o1, o2, o3, p1, i0, i1, i2, i3, rows, cols, DType.Q5_K);
                for (int k = 0; k < 4; k++) AssertBits(q8k[k], outs[k]);

                // Off: unchanged F32-activation path, row by row.
                SimdKernels.Q5KDecodeQ8KActivations = false;
                SimdKernels.MatVec(o4, p1, i0, rows, cols, DType.Q5_K);
                for (int r = 0; r < rows; r++)
                    Assert.Equal(BitConverter.SingleToInt32Bits(SimdKernels.DotQ5K(p1 + (long)r * bpr, i0, cols)),
                        BitConverter.SingleToInt32Bits(outs[4][r]));
                Assert.NotEqual(0, rows - Enumerable.Range(0, rows).Count(r =>
                    BitConverter.SingleToInt32Bits(outs[4][r]) == BitConverter.SingleToInt32Bits(q8k[0][r])));
            }
        }
        finally
        {
            SimdKernels.Q5KDecodeQ8KActivations = saved;
        }
    }

    private static void AssertBits(float[] expected, float[] actual)
    {
        for (int r = 0; r < expected.Length; r++)
            Assert.Equal(BitConverter.SingleToInt32Bits(expected[r]), BitConverter.SingleToInt32Bits(actual[r]));
    }
}
