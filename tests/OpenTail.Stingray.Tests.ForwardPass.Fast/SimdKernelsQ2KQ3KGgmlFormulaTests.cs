namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Pins <c>SimdKernels.DotQ2K_Q8K</c> / <c>DotQ3K_Q8K</c> against direct translations of ggml's
/// <c>ggml_vec_dot_q2_K_q8_K_generic</c> / <c>ggml_vec_dot_q3_K_q8_K_generic</c>
/// (examples/llama.cpp/llama.cpp/ggml/src/ggml-cpu/quants.c) on the SAME encoded weight row and the SAME
/// Q8_K activation buffer. Mirrors the Q5_K test: a ggml-formula oracle, not another production kernel.
/// Tolerance (declared before running): <c>2e-4 + 2e-5 * |reference|</c>, for float summation order only.
/// </summary>
public sealed unsafe class SimdKernelsQ2KQ3KGgmlFormulaTests
{
    private const float AbsTol = 2e-4f, RelTol = 2e-5f;

    private static void WriteHalf(byte[] dst, int offset, float value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((Half)value);
        dst[offset] = (byte)bits;
        dst[offset + 1] = (byte)(bits >> 8);
    }

    private static float ReadHalf(byte* p) => (float)BitConverter.UInt16BitsToHalf((ushort)(p[0] | (p[1] << 8)));

    // block_q2_K: scales[16] @0, qs[64] @16, d @80, dmin @82 (84 bytes).
    private static byte[] BuildQ2KRow(int cols, Random rnd)
    {
        var row = new byte[(cols / 256) * 84];
        for (int b = 0; b < cols / 256; b++)
        {
            int o = b * 84;
            for (int i = 0; i < 80; i++) row[o + i] = (byte)rnd.Next(256);
            WriteHalf(row, o + 80, (float)(rnd.NextDouble() * 0.09 + 0.01));
            WriteHalf(row, o + 82, (float)(rnd.NextDouble() * 0.02 + 0.001));
        }
        return row;
    }

    // block_q3_K: hmask[32] @0, qs[64] @32, scales[12] @96, d @108 (110 bytes).
    private static byte[] BuildQ3KRow(int cols, Random rnd)
    {
        var row = new byte[(cols / 256) * 110];
        for (int b = 0; b < cols / 256; b++)
        {
            int o = b * 110;
            for (int i = 0; i < 108; i++) row[o + i] = (byte)rnd.Next(256);
            WriteHalf(row, o + 108, (float)(rnd.NextDouble() * 0.09 + 0.01));
        }
        return row;
    }

    private static float RefQ2K(byte* row, byte* q8k, int cols)
    {
        int nb = cols / 256;
        float* yd = (float*)q8k;
        sbyte* yqs = (sbyte*)(q8k + nb * 4);
        short* ybs = (short*)(q8k + nb * 4 + nb * 256);
        float sumf = 0;
        for (int i = 0; i < nb; i++)
        {
            byte* x = row + i * 84;
            byte* sc = x;
            byte* q2 = x + 16;
            sbyte* q8 = yqs + i * 256;
            short* bsums = ybs + i * 16;
            int summs = 0;
            for (int j = 0; j < 16; j++) summs += bsums[j] * (sc[j] >> 4);
            float dall = yd[i] * ReadHalf(x + 80);
            float dmin = yd[i] * ReadHalf(x + 82);
            int isum = 0, isIdx = 0;
            for (int k = 0; k < 2; k++)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    int d = sc[isIdx++] & 0xF;
                    int isuml = 0;
                    for (int l = 0; l < 16; l++) isuml += q8[l] * ((q2[l] >> shift) & 3);
                    isum += d * isuml;
                    d = sc[isIdx++] & 0xF;
                    isuml = 0;
                    for (int l = 16; l < 32; l++) isuml += q8[l] * ((q2[l] >> shift) & 3);
                    isum += d * isuml;
                    shift += 2;
                    q8 += 32;
                }
                q2 += 32;
            }
            sumf += dall * isum - dmin * summs;
        }
        return sumf;
    }

    private static float RefQ3K(byte* row, byte* q8k, int cols)
    {
        int nb = cols / 256;
        float* yd = (float*)q8k;
        sbyte* yqs = (sbyte*)(q8k + nb * 4);
        const uint kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
        Span<sbyte> aux8 = stackalloc sbyte[256];
        Span<int> aux32 = stackalloc int[8];
        Span<float> sums = stackalloc float[8];
        Span<uint> auxs = stackalloc uint[4];
        Span<byte> scales = stackalloc byte[16];
        sums.Clear();
        for (int i = 0; i < nb; i++)
        {
            byte* x = row + i * 110;
            byte* hm = x;
            byte* q3 = x + 32;
            sbyte* q8 = yqs + i * 256;
            aux32.Clear();
            int a = 0;
            byte m = 1;
            for (int j = 0; j < 256; j += 128)
            {
                for (int s = 0; s < 4; s++)
                {
                    for (int l = 0; l < 32; l++)
                    {
                        int v = (q3[l] >> (2 * s)) & 3;
                        v -= (hm[l] & m) != 0 ? 0 : 4;
                        aux8[a + l] = (sbyte)v;
                    }
                    a += 32;
                    m <<= 1;
                }
                q3 += 32;
            }
            a = 0;
            uint a0 = *(uint*)(x + 96), a1 = *(uint*)(x + 100), a2 = *(uint*)(x + 104);
            uint tmp = a2;
            auxs[2] = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            auxs[3] = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            auxs[0] = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
            auxs[1] = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
            for (int w = 0; w < 4; w++)
                for (int bte = 0; bte < 4; bte++) scales[w * 4 + bte] = (byte)(auxs[w] >> (8 * bte));
            for (int j = 0; j < 16; j++)
            {
                int sc = (sbyte)scales[j] - 32;
                for (int half = 0; half < 2; half++)
                {
                    for (int l = 0; l < 8; l++) aux32[l] += sc * (q8[l] * aux8[a + l]);
                    q8 += 8; a += 8;
                }
            }
            float d = yd[i] * ReadHalf(x + 108);
            for (int l = 0; l < 8; l++) sums[l] += d * aux32[l];
        }
        float sumf = 0;
        for (int l = 0; l < 8; l++) sumf += sums[l];
        return sumf;
    }

    private delegate float RefDot(byte* row, byte* q8k, int cols);
    private delegate float KernelDot(byte* row, byte* q8k, int cols);

    private static void Check(string name, int cols, Func<int, Random, byte[]> build, RefDot reference, KernelDot kernel)
    {
        for (int seed = 0; seed < 16; seed++)
        {
            var rnd = new Random(20261001 + cols * 31 + seed);
            byte[] row = build(cols, rnd);
            var input = new float[cols];
            for (int i = 0; i < cols; i++) input[i] = (float)(rnd.NextDouble() * 2.0 - 1.0);
            var scratch = new byte[SimdKernels.Q8KScratchBytes(cols)];
            fixed (byte* r = row) fixed (float* x = input) fixed (byte* s = scratch)
            {
                SimdKernels.QuantizeRowToQ8K(x, cols, s);
                float expected = reference(r, s, cols);
                float actual = kernel(r, s, cols);
                float tol = AbsTol + MathF.Abs(expected) * RelTol;
                Assert.True(MathF.Abs(actual - expected) <= tol,
                    $"{name} cols={cols} seed={seed}: ggml formula {expected:R}, kernel {actual:R}, abs {MathF.Abs(actual - expected):R} > tol {tol:R}");
            }
        }
    }

    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(2048)]
    [InlineData(4096)]
    public void DotQ2K_Q8K_MatchesGgmlGenericFormula(int cols) =>
        Check("Q2_K", cols, BuildQ2KRow, RefQ2K, (r, s, c) => SimdKernels.DotQ2K_Q8K(r, s, c));

    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(2048)]
    [InlineData(4096)]
    public void DotQ3K_Q8K_MatchesGgmlGenericFormula(int cols) =>
        Check("Q3_K", cols, BuildQ3KRow, RefQ3K, (r, s, c) => SimdKernels.DotQ3K_Q8K(r, s, c));
}
