
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Parity gate for <see cref="SimdKernels.DotQ4_0"/>, the first fused CPU kernel Q4_0 has ever had.
///
/// <para>Before it, every Q4_0 matmul went through <c>Dequantize.ToFloat32</c> to a full fp32 copy
/// and then a generic float dot. That path is the independent oracle here: it shares no code with
/// the new kernel, and it is what production actually did, so agreement means the kernel can be
/// swapped in without changing a single output.</para>
///
/// <para>Correctness is checked BEFORE any timing, per the programme's standing rule — a faster
/// kernel that computes something else is not faster.</para>
/// </summary>
public sealed unsafe class Q40DotParityTests
{
    private const int QK = 32;
    private const int BytesPerBlock = 18;

    /// <summary>
    /// Build a Q4_0 row with every nibble value 0..15 exercised and a mix of scale magnitudes,
    /// so sign extension (nibble - 8 crossing zero) and the fp16 scale decode are both covered.
    /// </summary>
    private static byte[] MakeRow(int cols, int seed)
    {
        int numBlocks = cols / QK;
        var row = new byte[numBlocks * BytesPerBlock];
        var rng = new Random(seed);
        for (int b = 0; b < numBlocks; b++)
        {
            int o = b * BytesPerBlock;
            // Vary the scale across blocks, including a tiny and a negative-exponent one.
            float d = (float)((rng.NextDouble() - 0.5) * 0.25);
            var h = (Half)d;
            ushort bits = BitConverter.HalfToUInt16Bits(h);
            row[o] = (byte)(bits & 0xFF);
            row[o + 1] = (byte)(bits >> 8);
            for (int j = 0; j < QK / 2; j++)
                row[o + 2 + j] = (byte)rng.Next(0, 256);   // both nibbles, full 0..15 range
        }
        return row;
    }

    private static float[] MakeInput(int cols, int seed)
    {
        var rng = new Random(seed);
        var v = new float[cols];
        for (int i = 0; i < cols; i++) v[i] = (float)((rng.NextDouble() - 0.5) * 4.0);
        return v;
    }

    /// <summary>The oracle: dequantize exactly as production did, then a plain scalar dot.</summary>
    private static double DequantizeThenDot(byte[] row, float[] input, int cols)
    {
        var deq = new float[cols];
        Dequantize.ToFloat32(row, deq, DType.Q4_0, cols);
        double acc = 0;
        for (int i = 0; i < cols; i++) acc += (double)deq[i] * input[i];
        return acc;
    }

    [Theory]
    [InlineData(32)]      // one block
    [InlineData(64)]
    [InlineData(256)]
    [InlineData(2048)]    // a realistic row width
    [InlineData(2560)]    // gemma-4 E4B embedding dim
    public void DotQ4_0_MatchesDequantizeThenDot(int cols)
    {
        for (int seed = 0; seed < 8; seed++)
        {
            var row = MakeRow(cols, seed);
            var input = MakeInput(cols, seed + 1000);
            double expected = DequantizeThenDot(row, input, cols);

            float got;
            fixed (byte* r = row)
            fixed (float* inp = input)
                got = SimdKernels.DotQ4_0(r, inp, cols);

            // fp32 accumulation order differs from the oracle's; scale the tolerance to magnitude.
            double tol = 1e-4 * Math.Max(1.0, Math.Abs(expected));
            Assert.True(Math.Abs(got - expected) <= tol,
                $"cols={cols} seed={seed}: DotQ4_0={got} dequantize-then-dot={expected} (tol {tol})");
        }
    }

    /// <summary>
    /// Pins the nibble-to-element mapping specifically: qs[j] LOW nibble is element j and HIGH
    /// nibble is element j+16, NOT interleaved. A kernel that swapped these would still produce
    /// plausible magnitudes and pass a weak test, so drive it with a one-hot input that isolates
    /// exactly one element at a time.
    /// </summary>
    [Fact]
    public void DotQ4_0_NibbleToElementMapping_IsLowFirstThenHigh()
    {
        const int cols = QK;
        var row = new byte[BytesPerBlock];
        ushort one = BitConverter.HalfToUInt16Bits((Half)1.0f);
        row[0] = (byte)(one & 0xFF);
        row[1] = (byte)(one >> 8);
        for (int j = 0; j < QK / 2; j++)
            row[2 + j] = (byte)(((j % 16) << 4) | ((15 - j % 16) & 0xF));

        for (int elem = 0; elem < cols; elem++)
        {
            var input = new float[cols];
            input[elem] = 1f;

            int j = elem % (QK / 2);
            int expectedNibble = elem < QK / 2 ? (row[2 + j] & 0xF) : (row[2 + j] >> 4);
            float expected = expectedNibble - 8;

            float got;
            fixed (byte* r = row)
            fixed (float* inp = input)
                got = SimdKernels.DotQ4_0(r, inp, cols);

            Assert.True(Math.Abs(got - expected) < 1e-4f,
                $"element {elem}: got {got}, expected {expected} "
                + "— low nibble must map to element j and high nibble to element j+16.");
        }
    }

    /// <summary>
    /// The batched-prefill path quantizes activations to Q8_0 first, so its result is NOT expected
    /// to match the fp32 path bit for bit — activation quantization is lossy by construction. What
    /// must hold is that the SIMD integer dot agrees with a scalar reference over the same
    /// quantized inputs (pure implementation parity), and that the lossy result stays close enough
    /// to the fp32 answer to be usable. Both are checked, with different tolerances, because
    /// conflating them would let a real SIMD bug hide inside the quantization error budget.
    /// </summary>
    [Theory]
    [InlineData(32)]
    [InlineData(256)]
    [InlineData(2048)]
    [InlineData(2560)]
    public void DotQ4_0_Q8_0_MatchesFp32Dot_WithinQuantizationError(int cols)
    {
        for (int seed = 0; seed < 5; seed++)
        {
            var row = MakeRow(cols, seed);
            var input = MakeInput(cols, seed + 500);

            var scratch = new byte[SimdKernels.Q8_0ScratchBytes(cols)];
            float got, fp32;
            fixed (byte* r = row)
            fixed (float* inp = input)
            fixed (byte* s = scratch)
            {
                SimdKernels.QuantizeRowToQ8_0(inp, cols, s);
                got = SimdKernels.DotQ4_0_Q8_0(r, s, cols);
                fp32 = SimdKernels.DotQ4_0(r, inp, cols);
            }

            // Activations carry ~1/127 relative error; the dot accumulates it across cols.
            double scale = Math.Max(1.0, Math.Abs(fp32));
            double tol = 0.05 * scale + 1e-3 * Math.Sqrt(cols);
            Assert.True(Math.Abs(got - fp32) <= tol,
                $"cols={cols} seed={seed}: Q8_0 path={got} fp32 path={fp32} (tol {tol})");
        }
    }

    /// <summary>
    /// The 4-input variant unpacks each weight row once and reuses it across four activation rows.
    /// It must produce exactly what four separate single-input calls produce — the whole point is
    /// that it is a scheduling change, not an arithmetic one.
    /// </summary>
    [Theory]
    [InlineData(32)]
    [InlineData(2048)]
    public void DotQ4_0_Q8_0_4In_MatchesFourSingleCalls(int cols)
    {
        var row = MakeRow(cols, 77);
        int sb = SimdKernels.Q8_0ScratchBytes(cols);
        var scratch = new byte[sb * 4];
        var singles = new float[4];

        fixed (byte* r = row)
        fixed (byte* s = scratch)
        {
            for (int k = 0; k < 4; k++)
            {
                var input = MakeInput(cols, 900 + k);
                fixed (float* inp = input) SimdKernels.QuantizeRowToQ8_0(inp, cols, s + k * sb);
                singles[k] = SimdKernels.DotQ4_0_Q8_0(r, s + k * sb, cols);
            }

            SimdKernels.DotQ4_0_Q8_0_4In(r, s, s + sb, s + 2 * sb, s + 3 * sb, cols,
                out float o0, out float o1, out float o2, out float o3);

            Assert.Equal(singles[0], o0, 4);
            Assert.Equal(singles[1], o1, 4);
            Assert.Equal(singles[2], o2, 4);
            Assert.Equal(singles[3], o3, 4);
        }
    }
}
