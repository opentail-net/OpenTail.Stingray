namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// A9b of docs/00-current-work.md: how far does the activation quantiser move a dot product, per weight type? The first type is Q4_K.
///
/// <para>For each synthetic row (random Q4_K blocks, dequantised by <c>Dequantize</c>, against an activation vector with optional outlier channels)
/// three numbers are compared: the exact dot product in double; Stingray's <c>DotQ4K_Q8KS</c> kernel on the Q8_KS-quantised activations (the
/// production path); and the same dot with the activations quantised the way ggml's <c>quantize_row_q8_K_ref</c> does (one scale per 256 values,
/// the reference port in <c>Q8ActivationQuantizationTests</c>). The error is normalised by the sum of absolute products, so rows with a near-zero
/// dot do not dominate the mean.</para>
///
/// <para>The test asserts only that the errors are finite and that the means are small enough to be int8 activation noise. The ordering of the
/// two quantisers is printed and recorded in docs/00-current-work.md, not enforced, so a result that goes the other way is data, not a red test.</para>
/// </summary>
public sealed unsafe class Q4KDotActivationErrorTests
{
    private const int BlockSize = 256;
    private const int Q4KBlockBytes = 144;
    private const int Cols = 2048;
    private const int Rows = 64;

    private static byte[] RandomQ4KRow(Random rng, int cols)
    {
        int blocks = cols / BlockSize;
        var bytes = new byte[blocks * Q4KBlockBytes];
        rng.NextBytes(bytes);
        for (int b = 0; b < blocks; b++)
        {
            int off = b * Q4KBlockBytes;
            // Super-block scales (fp16 d and dmin) kept small and finite; the rest of the block is random quants and scale bytes.
            ushort d = BitConverter.HalfToUInt16Bits((Half)(0.005 + 0.02 * rng.NextDouble()));
            ushort dmin = BitConverter.HalfToUInt16Bits((Half)(0.001 + 0.01 * rng.NextDouble()));
            bytes[off] = (byte)(d & 0xFF); bytes[off + 1] = (byte)(d >> 8);
            bytes[off + 2] = (byte)(dmin & 0xFF); bytes[off + 3] = (byte)(dmin >> 8);
        }
        return bytes;
    }

    private static float[] OutlierRow(Random rng, int n, double amaxOverRms, int outliers)
    {
        var x = new float[n];
        for (int i = 0; i < n; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            x[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        for (int k = 0; k < outliers; k++)
            x[rng.Next(n)] = (float)(amaxOverRms * (rng.Next(2) == 0 ? 1 : -1) * (0.7 + 0.3 * rng.NextDouble()));
        return x;
    }

    // ggml quantize_row_q8_K_ref: one scale per 256 values, scale taken from the signed value of largest magnitude.
    private static void GgmlQ8K(float[] x, out float[] d, out sbyte[] qs)
    {
        int nb = x.Length / BlockSize;
        d = new float[nb];
        qs = new sbyte[x.Length];
        for (int b = 0; b < nb; b++)
        {
            float max = 0f, amax = 0f;
            for (int j = 0; j < BlockSize; j++)
            {
                float ax = MathF.Abs(x[b * BlockSize + j]);
                if (ax > amax) { amax = ax; max = x[b * BlockSize + j]; }
            }
            if (amax == 0f) continue;
            float iscale = -127.0f / max;
            for (int j = 0; j < BlockSize; j++)
            {
                int v = (int)MathF.Round(iscale * x[b * BlockSize + j], MidpointRounding.ToEven);
                qs[b * BlockSize + j] = (sbyte)Math.Min(127, v);
            }
            d[b] = 1.0f / iscale;
        }
    }

    [Theory]
    [InlineData(3.0)]
    [InlineData(25.0)]
    [InlineData(90.0)]
    public void Q4K_DotError_Q8KS_Versus_GgmlQ8K(double amaxOverRms)
    {
        var rng = new Random(23);
        int scratchBytes = SimdKernels.Q8KSScratchBytes(Cols);
        double sumKs = 0, sumK = 0, maxKs = 0, maxK = 0;
        for (int r = 0; r < Rows; r++)
        {
            byte[] row = RandomQ4KRow(rng, Cols);
            float[] x = OutlierRow(rng, Cols, amaxOverRms, outliers: 3);
            var w = new float[Cols];
            Dequantize.ToFloat32(row, w, DType.Q4_K, Cols);

            double exact = 0, norm = 0;
            for (int i = 0; i < Cols; i++)
            {
                exact += (double)w[i] * x[i];
                norm += Math.Abs((double)w[i] * x[i]);
            }

            GgmlQ8K(x, out var d, out var qs);
            double dotK = 0;
            for (int i = 0; i < Cols; i++) dotK += (double)w[i] * (d[i / BlockSize] * qs[i]);

            float dotKs;
            var scratch = new byte[scratchBytes];
            fixed (byte* rowPtr = row)
            fixed (byte* sPtr = scratch)
            fixed (float* xPtr = x)
            {
                SimdKernels.QuantizeRowToQ8KS(xPtr, Cols, sPtr);
                dotKs = SimdKernels.DotQ4K_Q8KS(rowPtr, sPtr, Cols);
            }

            double eKs = Math.Abs(dotKs - exact) / norm;
            double eK = Math.Abs(dotK - exact) / norm;
            Assert.True(double.IsFinite(eKs) && double.IsFinite(eK), $"row {r}: non-finite dot (Q8_KS {dotKs}, Q8_K {dotK}, exact {exact})");
            sumKs += eKs; sumK += eK;
            maxKs = Math.Max(maxKs, eKs); maxK = Math.Max(maxK, eK);
        }
        Console.WriteLine($"Q4_K dot, amax/rms={amaxOverRms}, {Rows} rows x {Cols} cols: normalised error mean Q8_KS {sumKs / Rows:E3} (max {maxKs:E3}), "
            + $"ggml Q8_K {sumK / Rows:E3} (max {maxK:E3})");
        Assert.True(sumKs / Rows < 0.05 && sumK / Rows < 0.05, "normalised dot error too large to be int8 activation noise");
    }
}
