
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// AVX2-vs-scalar equivalence and real timing for the BFloat16 fast kernel added 2026-08-28
/// (docs/05-cpu-architecture-kernel-opportunities.md, Backlog B). Admitted with a working
/// dequantizer but fell through to <c>MatVecDequantFallback</c> with no dedicated kernel — unlike
/// every other kernel in this backlog, no block/scale structure or activation requantization is
/// involved (raw 2-byte values, activation stays full F32), so the test fixture is simpler too.
/// </summary>
public sealed unsafe class SimdKernelsBF16Tests
{
    private static byte[] BuildMatrix(int rows, int cols, Random rng)
    {
        var bytes = new byte[rows * cols * 2];
        rng.NextBytes(bytes);
        return bytes;
    }

    [Fact]
    public void DotBF16_Avx2MatchesScalar()
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        foreach ((int rows, int cols) in new[] { (4, 8), (5, 17), (8, 64), (3, 129) })
        {
            var rng = new Random(unchecked((int)0xB715AA00) ^ (rows * 131 + cols));
            byte[] weightBytes = BuildMatrix(rows, cols, rng);
            var input = new float[cols];
            for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

            var avxOut = new float[rows];
            var scalarOut = new float[rows];

            fixed (byte* wPtr = weightBytes)
            fixed (float* iPtr = input)
            {
                for (int r = 0; r < rows; r++)
                {
                    ushort* rowPtr = (ushort*)(wPtr + (long)r * cols * 2);
                    avxOut[r] = SimdKernels.DotBF16(rowPtr, iPtr, cols);
                    scalarOut[r] = SimdKernels.DotBF16_Scalar(rowPtr, iPtr, cols);
                }
            }

            int mismatches = 0;
            float maxAbs = 0, maxRel = 0;
            for (int r = 0; r < rows; r++)
            {
                float diff = MathF.Abs(avxOut[r] - scalarOut[r]);
                float rel = diff / (MathF.Abs(scalarOut[r]) + 1e-6f);
                if (diff > maxAbs) maxAbs = diff;
                if (rel > maxRel) maxRel = rel;
                if (diff > 1e-3f && rel > 1e-3f) mismatches++;
            }
            Console.WriteLine(
                $"BF16 avx-vs-scalar rows={rows} cols={cols}: maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={mismatches}/{rows}");
            Assert.True(mismatches == 0,
                $"BF16 AVX2 vs scalar mismatch ({mismatches}/{rows}) rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
        }
    }

    /// <summary>Real before/after timing, same shape/rationale as the other Backlog B benchmarks.</summary>
    [Fact]
    public void FastKernel_IsFasterThanDequantFallback()
    {
        const int rows = 17408, cols = 5120;
        var rng = new Random(0x8F16);
        byte[] weightBytes = BuildMatrix(rows, cols, rng);
        var input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        var outFast = new float[rows];
        var outFallback = new float[rows];

        fixed (byte* wPtr = weightBytes)
        fixed (float* iPtr = input)
        fixed (float* ofPtr = outFast)
        fixed (float* obPtr = outFallback)
        {
            const int trials = 7;
            const int itersPerTrial = 10;
            var fastTimes = new double[trials];
            var fallbackTimes = new double[trials];

            for (int w = 0; w < 5; w++)
            {
                SimdKernels.MatVecBF16(ofPtr, wPtr, iPtr, rows, cols);
                SimdKernels.MatVecDequantFallback(obPtr, wPtr, iPtr, rows, cols, DType.BFloat16);
            }

            for (int t = 0; t < trials; t++)
            {
                var swFast = System.Diagnostics.Stopwatch.StartNew();
                for (int it = 0; it < itersPerTrial; it++) SimdKernels.MatVecBF16(ofPtr, wPtr, iPtr, rows, cols);
                swFast.Stop();
                fastTimes[t] = swFast.Elapsed.TotalMilliseconds / itersPerTrial;

                var swFallback = System.Diagnostics.Stopwatch.StartNew();
                for (int it = 0; it < itersPerTrial; it++) SimdKernels.MatVecDequantFallback(obPtr, wPtr, iPtr, rows, cols, DType.BFloat16);
                swFallback.Stop();
                fallbackTimes[t] = swFallback.Elapsed.TotalMilliseconds / itersPerTrial;
            }

            Array.Sort(fastTimes);
            Array.Sort(fallbackTimes);
            double fastMs = fastTimes[0]; // min, not median: noise only ever slows a trial down, never speeds it up
            double fallbackMs = fallbackTimes[0];
            double speedup = fallbackMs / fastMs;
            string report = $"BF16 rows={rows} cols={cols}: fallback={fallbackMs:F3}ms fast={fastMs:F3}ms " +
                $"speedup={speedup:F2}x (min of {trials} trials, {itersPerTrial} iters each)";
            Console.WriteLine(report);
            Assert.True(speedup > 1.0, $"BF16: median speedup={speedup:F2}x is not > 1.0\n{report}");
        }
    }
}
