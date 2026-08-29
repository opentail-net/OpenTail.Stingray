
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Correctness and timing for the TQ2_0/TQ1_0 ternary formats added 2026-08-28
/// (docs/05-cpu-architecture-kernel-opportunities.md, Backlog C). Neither format had any
/// implementation before this session. TQ2_0 got a full AVX2 kernel (mirrors Q2_0's already-shipped
/// shape); TQ1_0 is deliberately scalar-only for now (its base-3 digit-extraction packing is novel
/// in this codebase and AVX2 is gated on TQ2_0's real result), so it's verified the same way
/// IQ1_M was: dequantizer cross-checked against the int-domain dot kernel, no AVX2-vs-scalar
/// comparison to lean on.
/// </summary>
public sealed unsafe class SimdKernelsTqTests
{
    private static ushort HalfToUshort(Half h) => BitConverter.HalfToUInt16Bits(h);

    private static byte[] BuildTq2_0Matrix(int rows, int cols, Random rng)
    {
        const int elementsPerBlock = 256, bytesPerBlock = 66;
        if (cols % elementsPerBlock != 0)
            throw new ArgumentException("cols must be a multiple of 256.");
        int blocksPerRow = cols / elementsPerBlock;
        int bytesPerRow = blocksPerRow * bytesPerBlock;
        var bytes = new byte[rows * bytesPerRow];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * bytesPerBlock;
                for (int i = 0; i < 64; i++)
                    bytes[off + i] = (byte)rng.Next(256);
                float d = (float)(rng.NextDouble() * 0.09 + 0.01);
                ushort dHalf = HalfToUshort((Half)d);
                bytes[off + 64] = (byte)(dHalf & 0xFF);
                bytes[off + 65] = (byte)(dHalf >> 8);
            }
        }
        return bytes;
    }

    private static byte[] BuildTq1_0Block(Random rng)
    {
        const int bytesPerBlock = 54;
        var bytes = new byte[bytesPerBlock];
        for (int i = 0; i < 52; i++) bytes[i] = (byte)rng.Next(256);
        float d = (float)(rng.NextDouble() * 0.09 + 0.01);
        ushort dHalf = HalfToUshort((Half)d);
        bytes[52] = (byte)(dHalf & 0xFF);
        bytes[53] = (byte)(dHalf >> 8);
        return bytes;
    }

    [Fact]
    public void DotTq2_0_Q8K_Avx2MatchesScalar()
    {
        if (!Avx2.IsSupported || !Avx.IsSupported) return;

        foreach ((int rows, int cols) in new[] { (4, 256), (5, 512), (8, 2048), (3, 4096) })
        {
            var rng = new Random(unchecked((int)0x7702005) ^ (rows * 131 + cols));
            byte[] weightBytes = BuildTq2_0Matrix(rows, cols, rng);
            var input = new float[cols];
            for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

            int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
            var scratch = new byte[scratchBytes];
            var avxOut = new float[rows];
            var scalarOut = new float[rows];

            fixed (byte* wPtr = weightBytes)
            fixed (byte* sPtr = scratch)
            fixed (float* iPtr = input)
            {
                SimdKernels.QuantizeRowToQ8K(iPtr, cols, sPtr);
                int bytesPerRow = (cols / 256) * 66;
                for (int r = 0; r < rows; r++)
                {
                    byte* rowPtr = wPtr + (long)r * bytesPerRow;
                    avxOut[r] = SimdKernels.DotTq2_0_Q8K(rowPtr, sPtr, cols);
                    scalarOut[r] = SimdKernels.DotTq2_0_Q8K_Scalar(rowPtr, sPtr, cols);
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
                $"TQ2_0 avx-vs-scalar rows={rows} cols={cols}: maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={mismatches}/{rows}");
            Assert.True(mismatches == 0,
                $"TQ2_0 AVX2 vs scalar mismatch ({mismatches}/{rows}) rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
        }
    }

    /// <summary>
    /// Independent cross-check for TQ2_0 too, same rationale as the IQ1_S/IQ1_M sibling tests:
    /// the dequantizer and dot kernel are both novel, hand-derived formulas here, so an
    /// AVX2-vs-scalar-of-itself check alone can't catch a shared mistake in either.
    /// </summary>
    [Fact]
    public void DotTq2_0_Q8K_AgreesWithDequantThenF32Dot()
    {
        const int cols = 256;
        var rng = new Random(0x7702);
        byte[] weightBytes = BuildTq2_0Matrix(1, cols, rng);
        var input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var dequantWeights = new float[cols];
        Dequantize.ToFloat32(weightBytes, dequantWeights, DType.TQ2_0, cols);
        float reference = 0f;
        for (int i = 0; i < cols; i++) reference += dequantWeights[i] * input[i];

        int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
        var scratch = new byte[scratchBytes];
        float kernelResult;
        fixed (byte* wPtr = weightBytes)
        fixed (byte* sPtr = scratch)
        fixed (float* iPtr = input)
        {
            SimdKernels.QuantizeRowToQ8K(iPtr, cols, sPtr);
            kernelResult = SimdKernels.DotTq2_0_Q8K(wPtr, sPtr, cols);
        }

        float diff = MathF.Abs(kernelResult - reference);
        float relDiff = diff / (MathF.Abs(reference) + 1e-6f);
        Console.WriteLine($"TQ2_0 dot-vs-dequant cross-check: kernel={kernelResult:F6} " +
            $"dequant+f32dot={reference:F6} diff={diff:E3} relDiff={relDiff:E3}");
        Assert.True(relDiff < 0.05f,
            $"TQ2_0 dot kernel disagrees with dequant+F32 reference by {relDiff:P1} " +
            $"(kernel={kernelResult:F6}, reference={reference:F6})");
    }

    [Fact]
    public void DotTq1_0_Q8K_AgreesWithDequantThenF32Dot()
    {
        const int cols = 256;
        var rng = new Random(0x1710);

        for (int trial = 0; trial < 8; trial++)
        {
            byte[] weightBytes = BuildTq1_0Block(rng);
            var input = new float[cols];
            for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

            var dequantWeights = new float[cols];
            Dequantize.ToFloat32(weightBytes, dequantWeights, DType.TQ1_0, cols);
            float reference = 0f;
            for (int i = 0; i < cols; i++) reference += dequantWeights[i] * input[i];

            int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
            var scratch = new byte[scratchBytes];
            float kernelResult;
            fixed (byte* wPtr = weightBytes)
            fixed (byte* sPtr = scratch)
            fixed (float* iPtr = input)
            {
                SimdKernels.QuantizeRowToQ8K(iPtr, cols, sPtr);
                kernelResult = SimdKernels.DotTq1_0_Q8K_Scalar(wPtr, sPtr, cols);
            }

            float diff = MathF.Abs(kernelResult - reference);
            float relDiff = diff / (MathF.Abs(reference) + 1e-6f);
            Console.WriteLine($"TQ1_0 dot-vs-dequant cross-check trial={trial}: kernel={kernelResult:F6} " +
                $"dequant+f32dot={reference:F6} diff={diff:E3} relDiff={relDiff:E3}");
            Assert.True(relDiff < 0.05f,
                $"TQ1_0 dot kernel disagrees with dequant+F32 reference by {relDiff:P1} on trial {trial} " +
                $"(kernel={kernelResult:F6}, reference={reference:F6})");
        }
    }

    [Fact]
    public void DequantTq1_0_ProducesFiniteValues()
    {
        var rng = new Random(0x1711);
        for (int trial = 0; trial < 20; trial++)
        {
            byte[] block = BuildTq1_0Block(rng);
            var y = new float[256];
            Dequantize.ToFloat32(block, y, DType.TQ1_0, 256);
            foreach (float v in y)
                Assert.True(float.IsFinite(v), $"TQ1_0 dequant produced non-finite value {v} on trial {trial}");
        }
    }

    /// <summary>
    /// Real before/after timing for TQ2_0, min of 7 interleaved trials, same shape as the other
    /// Backlog B/C benchmarks. Measured result is NOT a stable win (one isolated run showed 1.19x,
    /// but 3/4 runs embedded in the full test suite under realistic system-load noise came back
    /// 0.89-0.98x) — so TQ2_0 is deliberately NOT wired into SimdKernels.MatVec's dispatch switch
    /// (see the comment there and docs/05's Backlog C). This test only reports the number; it does
    /// not assert a win, matching the Q5_0/Q5_1/IQ1_S precedent in the sibling test files.
    /// </summary>
    [Fact]
    public void Tq2_0_FastKernel_TimingReport()
    {
        const int rows = 17408, cols = 5120;
        var rng = new Random(0x7703);
        byte[] weightBytes = BuildTq2_0Matrix(rows, cols, rng);
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
                SimdKernels.MatVecTq2_0(ofPtr, wPtr, iPtr, rows, cols);
                SimdKernels.MatVecDequantFallback(obPtr, wPtr, iPtr, rows, cols, DType.TQ2_0);
            }

            for (int t = 0; t < trials; t++)
            {
                var swFast = System.Diagnostics.Stopwatch.StartNew();
                for (int it = 0; it < itersPerTrial; it++) SimdKernels.MatVecTq2_0(ofPtr, wPtr, iPtr, rows, cols);
                swFast.Stop();
                fastTimes[t] = swFast.Elapsed.TotalMilliseconds / itersPerTrial;

                var swFallback = System.Diagnostics.Stopwatch.StartNew();
                for (int it = 0; it < itersPerTrial; it++) SimdKernels.MatVecDequantFallback(obPtr, wPtr, iPtr, rows, cols, DType.TQ2_0);
                swFallback.Stop();
                fallbackTimes[t] = swFallback.Elapsed.TotalMilliseconds / itersPerTrial;
            }

            Array.Sort(fastTimes);
            Array.Sort(fallbackTimes);
            // Min, not median: see docs/05's benchmark-robustness note.
            double fastMs = fastTimes[0];
            double fallbackMs = fallbackTimes[0];
            double speedup = fallbackMs / fastMs;
            string report = $"TQ2_0 rows={rows} cols={cols}: fallback={fallbackMs:F3}ms fast={fastMs:F3}ms " +
                $"speedup={speedup:F2}x (min of {trials} trials, {itersPerTrial} iters each)";
            Console.WriteLine(report);
        }
    }
}
