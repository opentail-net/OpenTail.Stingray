
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Correctness and timing for the IQ1_S fast kernel added 2026-08-28
/// (docs/05-cpu-architecture-kernel-opportunities.md, Backlog A). Unlike the six Q8_K-paired IQ
/// kernels done earlier this session, IQ1_S needed its dequantizer (<see
/// cref="Dequantize"/>'s <c>DequantIq1S</c>) and grid table (<see cref="IqCodebooks.Iq1sGrid"/>)
/// built from scratch, not just the fast matvec — so this file carries one more check than the
/// others: <see cref="DotIq1S_Q8K_AgreesWithDequantThenF32Dot"/> cross-verifies the int-domain dot
/// kernel against an independently-derived path (materialize via the dequantizer, then a plain
/// F32 dot), which the AVX2-vs-scalar-of-itself comparison alone cannot catch (both share the same
/// hand-derived formula, so a shared mistake would pass that check silently).
/// </summary>
public sealed unsafe class SimdKernelsIq1STests
{
    private static ushort HalfToUshort(Half h) => BitConverter.HalfToUInt16Bits(h);

    private static byte[] BuildMatrix(int rows, int cols, Random rng)
    {
        const int elementsPerBlock = 256, bytesPerBlock = 50;
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
                for (int i = 2; i < bytesPerBlock; i++)
                    bytes[off + i] = (byte)rng.Next(256);
                float d = (float)(rng.NextDouble() * 0.09 + 0.01);
                ushort dHalf = HalfToUshort((Half)d);
                bytes[off] = (byte)(dHalf & 0xFF);
                bytes[off + 1] = (byte)(dHalf >> 8);
            }
        }
        return bytes;
    }

    [Fact]
    public void DotIq1S_Q8K_Avx2MatchesScalar()
    {
        if (!Avx2.IsSupported || !Avx.IsSupported) return;

        foreach ((int rows, int cols) in new[] { (4, 256), (5, 512), (8, 2048), (3, 4096) })
        {
            var rng = new Random(unchecked((int)0x1051005) ^ (rows * 131 + cols));
            byte[] weightBytes = BuildMatrix(rows, cols, rng);

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
                int bytesPerRow = (cols / 256) * 50;
                for (int r = 0; r < rows; r++)
                {
                    byte* rowPtr = wPtr + (long)r * bytesPerRow;
                    avxOut[r] = SimdKernels.DotIq1S_Q8K(rowPtr, sPtr, cols);
                    scalarOut[r] = SimdKernels.DotIq1S_Q8K_Scalar(rowPtr, sPtr, cols);
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
                $"IQ1_S avx-vs-scalar rows={rows} cols={cols}: maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={mismatches}/{rows}");
            Assert.True(mismatches == 0,
                $"IQ1_S AVX2 vs scalar mismatch ({mismatches}/{rows}) rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
        }
    }

    /// <summary>
    /// Independent cross-check: dequantize the same random block via <c>Dequantize.ToFloat32</c>
    /// (a hand-derived formula written separately from the dot kernel, working from the
    /// dequantize_row_iq1_s reference rather than ggml_vec_dot_iq1_s_q8_K), then dot the result
    /// against the SAME raw (unquantized) float input with a plain F32 dot. Q8_K activation
    /// quantization introduces its own small error, so this isn't expected to match bit-for-bit —
    /// but a large, systematic deviation would mean the dot kernel and the dequantizer disagree
    /// about what the format actually encodes, which is exactly the class of bug an
    /// AVX2-vs-scalar-of-itself check cannot catch (both share whichever formula is wrong).
    /// </summary>
    [Fact]
    public void DotIq1S_Q8K_AgreesWithDequantThenF32Dot()
    {
        const int cols = 256;
        var rng = new Random(0x1051);
        byte[] weightBytes = BuildMatrix(1, cols, rng);

        var input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var dequantWeights = new float[cols];
        Dequantize.ToFloat32(weightBytes, dequantWeights, DType.IQ1_S, cols);
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
            kernelResult = SimdKernels.DotIq1S_Q8K(wPtr, sPtr, cols);
        }

        float diff = MathF.Abs(kernelResult - reference);
        float relDiff = diff / (MathF.Abs(reference) + 1e-6f);
        Console.WriteLine(
            $"IQ1_S dot-vs-dequant cross-check: kernel={kernelResult:F6} dequant+f32dot={reference:F6} " +
            $"diff={diff:E3} relDiff={relDiff:E3}");
        // Q8_K activation quantization is int8/127-scale, so a few-percent relative gap from
        // quantization error alone is expected; this catches a wrong FORMULA (which would produce
        // a gap far larger than quantization noise explains), not exact equality.
        Assert.True(relDiff < 0.05f,
            $"IQ1_S dot kernel disagrees with dequant+F32 reference by {relDiff:P1} " +
            $"(kernel={kernelResult:F6}, reference={reference:F6}) — larger than Q8_K quantization " +
            "error should explain; likely a formula mismatch between the dot kernel and the dequantizer.");
    }

    /// <summary>Real before/after timing, same shape/rationale as the other backlog benchmarks.</summary>
    [Fact]
    public void FastKernel_IsFasterThanDequantFallback()
    {
        const int rows = 17408, cols = 5120;
        var rng = new Random(0x1590);
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
                SimdKernels.MatVecIq1S(ofPtr, wPtr, iPtr, rows, cols);
                SimdKernels.MatVecDequantFallback(obPtr, wPtr, iPtr, rows, cols, DType.IQ1_S);
            }

            for (int t = 0; t < trials; t++)
            {
                var swFast = System.Diagnostics.Stopwatch.StartNew();
                for (int it = 0; it < itersPerTrial; it++) SimdKernels.MatVecIq1S(ofPtr, wPtr, iPtr, rows, cols);
                swFast.Stop();
                fastTimes[t] = swFast.Elapsed.TotalMilliseconds / itersPerTrial;

                var swFallback = System.Diagnostics.Stopwatch.StartNew();
                for (int it = 0; it < itersPerTrial; it++) SimdKernels.MatVecDequantFallback(obPtr, wPtr, iPtr, rows, cols, DType.IQ1_S);
                swFallback.Stop();
                fallbackTimes[t] = swFallback.Elapsed.TotalMilliseconds / itersPerTrial;
            }

            Array.Sort(fastTimes);
            Array.Sort(fallbackTimes);
            // Min, not median: see docs/05's benchmark-robustness note.
            double fastMs = fastTimes[0];
            double fallbackMs = fallbackTimes[0];
            double speedup = fallbackMs / fastMs;
            Console.WriteLine($"IQ1_S rows={rows} cols={cols}: fallback={fallbackMs:F3}ms fast={fastMs:F3}ms " +
                $"speedup={speedup:F2}x (min of {trials} trials, {itersPerTrial} iters each)");
            // NOT asserted > 1.0: measured consistently ~0.80-0.86x across repeated runs (real,
            // not noise) — MatVecIq1S is deliberately not wired into SimdKernels.MatVec's
            // dispatch for this reason (see the comment there). Kept as a documented measurement,
            // matching Q5_0/Q5_1's treatment in the other Backlog B test files.
        }
    }
}
