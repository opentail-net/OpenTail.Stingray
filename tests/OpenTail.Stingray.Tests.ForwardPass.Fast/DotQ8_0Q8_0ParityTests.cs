
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// docs/bugstofix.md (ModelCompatibility.cs:461, deepseek2 investigation): MatVecQ8_0 previously
/// dequantized the weight to F32 and dotted against the raw F32 activation, never quantizing the
/// activation at all -- unlike every other quantized dtype's primary MatVec, and unlike ggml's real
/// Q8_0 kernel (ggml_vec_dot_q8_0_q8_0), which always pairs a Q8_0 weight with a Q8_0-quantized
/// activation. This pins that the new <see cref="SimdKernels.DotQ8_0_Q8_0"/> family is a correct
/// int8-domain dot product (matches a naive scalar reference within quantization-noise tolerance,
/// and the AVX2 port agrees with its own scalar reference to a tight tolerance) -- not proving
/// ggml bit-parity, which needs ggml itself as the oracle.
/// </summary>
public sealed unsafe class DotQ8_0Q8_0ParityTests
{
    private static void WriteHalf(byte[] buffer, int offset, float value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((Half)value);
        buffer[offset] = (byte)(bits & 0xFF);
        buffer[offset + 1] = (byte)(bits >> 8);
    }

    /// <summary>One Q8_0 weight row: numBlocks * 34 bytes, [fp16 d][32 x sbyte qs] each.</summary>
    private static byte[] MakeQ8_0Row(int numBlocks, int seed)
    {
        var rng = new Random(seed);
        var bytes = new byte[numBlocks * 34];
        for (int b = 0; b < numBlocks; b++)
        {
            int off = b * 34;
            WriteHalf(bytes, off, 0.02f);
            for (int i = 0; i < 32; i++)
                bytes[off + 2 + i] = unchecked((byte)(sbyte)rng.Next(-127, 128));
        }
        return bytes;
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(8, 3)]
    public void AvxMatchesScalar_AndBothMatchNaiveF32Reference(int numBlocks, int seed)
    {
        Assert.SkipUnless(System.Runtime.Intrinsics.X86.Avx2.IsSupported
            && System.Runtime.Intrinsics.X86.Fma.IsSupported,
            "AVX2/FMA not available in this environment");

        int cols = numBlocks * 32;
        byte[] row = MakeQ8_0Row(numBlocks, seed);

        var rng = new Random(seed + 1000);
        float[] input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        // Naive reference: dequantize the weight exactly, dot against F32 input directly (no
        // activation quantization) -- the OLD behavior, and a reasonable upper-precision oracle
        // since quantizing the activation can only add noise relative to this.
        double naive = 0;
        for (int b = 0; b < numBlocks; b++)
        {
            float d = (float)BitConverter.UInt16BitsToHalf(
                (ushort)(row[b * 34] | (row[b * 34 + 1] << 8)));
            for (int i = 0; i < 32; i++)
            {
                sbyte q = unchecked((sbyte)row[b * 34 + 2 + i]);
                naive += (double)q * d * input[b * 32 + i];
            }
        }

        int scratchBytes = SimdKernels.Q8_0ScratchBytes(cols);
        byte[] scratch = new byte[scratchBytes];

        fixed (byte* pRow = row)
        fixed (float* pInput = input)
        fixed (byte* pScratch = scratch)
        {
            SimdKernels.QuantizeRowToQ8_0(pInput, cols, pScratch);
            float scalarResult = SimdKernels.DotQ8_0_Q8_0_Scalar(pRow, pScratch, numBlocks);
            float avx2Result = SimdKernels.DotQ8_0_Q8_0_Avx2(pRow, pScratch, numBlocks);

            float scale = Math.Max(1e-3f, Math.Abs(scalarResult));
            Assert.True(Math.Abs(avx2Result - scalarResult) / scale < 1e-4f,
                $"scalar={scalarResult:G9} avx2={avx2Result:G9} (numBlocks={numBlocks}, seed={seed})");

            // Activation quantization to int8 (127 symmetric levels) is real, bounded noise --
            // loose tolerance appropriate for that, not for a bit-parity check.
            double relErr = Math.Abs((double)scalarResult - naive) / Math.Max(1e-3, Math.Abs(naive));
            Assert.True(relErr < 0.02,
                $"scalar={scalarResult:G9} naiveF32={naive:G9} relErr={relErr:P2} (numBlocks={numBlocks}, seed={seed})");
        }
    }
}
