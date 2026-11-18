namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Extends the A9b activation-error comparison to Q3_K, Q5_K, Q6_K and Q8_0.
/// It compares identical dequantised weight rows and activation vectors against a double
/// precision dot: Q8_KS uses Stingray's production quantizer, while Q8_K mirrors ggml's
/// one-scale-per-256 quantizer. These are measurements, not assertions about which layout
/// should win; results are recorded in docs/00-current-work.md.
/// </summary>
public sealed unsafe class PerWeightActivationQuantizationErrorTests
{
    private const int Cols = 2048;
    private const int Rows = 64;
    private const int SuperBlock = 256;

    [Theory]
    [InlineData(DType.Q3_K, 3.0)]
    [InlineData(DType.Q3_K, 25.0)]
    [InlineData(DType.Q3_K, 90.0)]
    [InlineData(DType.Q5_K, 3.0)]
    [InlineData(DType.Q5_K, 25.0)]
    [InlineData(DType.Q5_K, 90.0)]
    [InlineData(DType.Q6_K, 3.0)]
    [InlineData(DType.Q6_K, 25.0)]
    [InlineData(DType.Q6_K, 90.0)]
    [InlineData(DType.Q8_0, 3.0)]
    [InlineData(DType.Q8_0, 25.0)]
    [InlineData(DType.Q8_0, 90.0)]
    public void ActivationQuantization_DotErrorByWeightType(DType dtype, double amaxOverRms)
    {
        var rng = new Random(23);
        double sumKs = 0, sumK = 0, maxKs = 0, maxK = 0;
        int bytesPerRow = Cols / DTypeInfo.BlockSize(dtype) * DTypeInfo.BytesPerBlock(dtype);
        int q8KsBytes = SimdKernels.Q8KSScratchBytes(Cols);
        int q8KBytes = SimdKernels.Q8KScratchBytes(Cols);

        for (int r = 0; r < Rows; r++)
        {
            byte[] row = RandomWeightRow(rng, dtype, bytesPerRow);
            float[] x = OutlierRow(rng, Cols, amaxOverRms, outliers: 3);
            var w = new float[Cols];
            Dequantize.ToFloat32(row, w, dtype, Cols);

            double exact = 0, norm = 0;
            for (int i = 0; i < Cols; i++)
            {
                exact += (double)w[i] * x[i];
                norm += Math.Abs((double)w[i] * x[i]);
            }

            var ksScratch = new byte[q8KsBytes];
            var kScratch = new byte[q8KBytes];
            fixed (float* xPtr = x)
            fixed (byte* ksPtr = ksScratch)
            fixed (byte* kPtr = kScratch)
            {
                SimdKernels.QuantizeRowToQ8KS(xPtr, Cols, ksPtr);
                SimdKernels.QuantizeRowToQ8K(xPtr, Cols, kPtr);
                double dotKs = DotFromQ8KS(w, ksPtr);
                double dotK = DotFromQ8K(w, kPtr);

                double eKs = Math.Abs(dotKs - exact) / norm;
                double eK = Math.Abs(dotK - exact) / norm;
                Assert.True(double.IsFinite(eKs) && double.IsFinite(eK),
                    $"{dtype} row {r}: non-finite dot (Q8_KS {dotKs}, Q8_K {dotK}, exact {exact})");
                sumKs += eKs;
                sumK += eK;
                maxKs = Math.Max(maxKs, eKs);
                maxK = Math.Max(maxK, eK);
            }
        }

        Console.WriteLine($"{dtype} dot, amax/rms={amaxOverRms}, {Rows} rows x {Cols} cols: normalized error mean Q8_KS {sumKs / Rows:E3} (max {maxKs:E3}), "
            + $"ggml Q8_K {sumK / Rows:E3} (max {maxK:E3})");
        Assert.True(sumKs / Rows < 0.05 && sumK / Rows < 0.05,
            $"{dtype} normalized dot error too large to be int8 activation noise");
    }

    private static byte[] RandomWeightRow(Random rng, DType dtype, int bytesPerRow)
    {
        int blockSize = DTypeInfo.BlockSize(dtype);
        int blockBytes = DTypeInfo.BytesPerBlock(dtype);
        var row = new byte[bytesPerRow];
        rng.NextBytes(row);

        int scaleOffset = dtype switch
        {
            DType.Q3_K => 108,
            DType.Q5_K => 0,
            DType.Q6_K => 208,
            DType.Q8_0 => 0,
            _ => throw new ArgumentOutOfRangeException(nameof(dtype))
        };
        for (int block = 0; block < Cols / blockSize; block++)
        {
            int offset = block * blockBytes + scaleOffset;
            ushort scale = BitConverter.HalfToUInt16Bits((Half)(0.005 + 0.02 * rng.NextDouble()));
            row[offset] = (byte)scale;
            row[offset + 1] = (byte)(scale >> 8);
            if (dtype == DType.Q5_K)
            {
                ushort min = BitConverter.HalfToUInt16Bits((Half)(0.001 + 0.01 * rng.NextDouble()));
                row[offset + 2] = (byte)min;
                row[offset + 3] = (byte)(min >> 8);
            }
        }

        return row;
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

    private static double DotFromQ8KS(float[] weights, byte* scratch)
    {
        int blocks = weights.Length / SuperBlock;
        float* scales = (float*)scratch;
        sbyte* qs = (sbyte*)(scratch + blocks * 32);
        double dot = 0;
        for (int block = 0; block < blocks; block++)
        for (int i = 0; i < SuperBlock; i++)
        {
            float d = scales[block * 8 + i / 32];
            sbyte q = qs[block * SuperBlock + i];
            dot += (double)weights[block * SuperBlock + i] * (d * q);
        }
        return dot;
    }

    private static double DotFromQ8K(float[] weights, byte* scratch)
    {
        int blocks = weights.Length / SuperBlock;
        float* scales = (float*)scratch;
        sbyte* qs = (sbyte*)(scratch + blocks * sizeof(float));
        double dot = 0;
        for (int block = 0; block < blocks; block++)
        for (int i = 0; i < SuperBlock; i++)
            dot += (double)weights[block * SuperBlock + i] * (scales[block] * qs[block * SuperBlock + i]);
        return dot;
    }
}
