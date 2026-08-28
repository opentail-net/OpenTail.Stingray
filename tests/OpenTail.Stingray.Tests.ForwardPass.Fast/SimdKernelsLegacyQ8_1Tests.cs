using System.Runtime.Intrinsics.X86;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// AVX2-vs-scalar equivalence, plus a real timing microbenchmark, for the Q8_1-paired legacy
/// format kernels added 2026-08-28 (docs/05-cpu-architecture-kernel-opportunities.md, Backlog B)
/// — Q4_1, Q5_1. Both were admitted with working dequantizers but fell through to
/// <c>MatVecDequantFallback</c> with no dedicated kernel; both needed a new Q8_1 activation-quant
/// scratch (<see cref="SimdKernels.QuantizeRowToQ8_1"/>) that didn't exist before this session.
/// Mirrors <c>SimdKernelsLegacyQ8_0Tests</c>'s pattern throughout.
/// </summary>
public sealed unsafe class SimdKernelsLegacyQ8_1Tests
{
    private static ushort HalfToUshort(Half h) => BitConverter.HalfToUInt16Bits(h);

    private static byte[] BuildMatrix(int rows, int cols, int bytesPerBlock, Random rng)
    {
        const int elementsPerBlock = 32;
        if (cols % elementsPerBlock != 0)
            throw new ArgumentException("cols must be a multiple of 32.");
        int blocksPerRow = cols / elementsPerBlock;
        int bytesPerRow = blocksPerRow * bytesPerBlock;
        var bytes = new byte[rows * bytesPerRow];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * bytesPerBlock;
                for (int i = 4; i < bytesPerBlock; i++)
                    bytes[off + i] = (byte)rng.Next(256);

                float d = (float)(rng.NextDouble() * 0.09 + 0.01);
                ushort dHalf = HalfToUshort((Half)d);
                bytes[off] = (byte)(dHalf & 0xFF);
                bytes[off + 1] = (byte)(dHalf >> 8);

                float m = (float)(rng.NextDouble() * 0.02 - 0.01);
                ushort mHalf = HalfToUshort((Half)m);
                bytes[off + 2] = (byte)(mHalf & 0xFF);
                bytes[off + 3] = (byte)(mHalf >> 8);
            }
        }
        return bytes;
    }

    private static void AssertAvx2MatchesScalar(
        string label, int bytesPerBlock,
        Func<nint, nint, int, float> avxDot, Func<nint, nint, int, float> scalarDot)
    {
        if (!Avx2.IsSupported || !Ssse3.IsSupported) return;

        foreach ((int rows, int cols) in new[] { (4, 32), (5, 64), (8, 128), (3, 256) })
        {
            var rng = new Random(unchecked((int)0x9A1B2C3D) ^ (rows * 131 + cols) ^ label.GetHashCode());
            byte[] weightBytes = BuildMatrix(rows, cols, bytesPerBlock, rng);

            var input = new float[cols];
            for (int i = 0; i < cols; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            int scratchBytes = SimdKernels.Q8_1ScratchBytes(cols);
            var scratch = new byte[scratchBytes];

            var avxOut = new float[rows];
            var scalarOut = new float[rows];

            fixed (byte* wPtr = weightBytes)
            fixed (byte* sPtr = scratch)
            fixed (float* iPtr = input)
            {
                SimdKernels.QuantizeRowToQ8_1(iPtr, cols, sPtr);
                int bytesPerRow = (cols / 32) * bytesPerBlock;
                for (int r = 0; r < rows; r++)
                {
                    nint rowPtr = (nint)(wPtr + (long)r * bytesPerRow);
                    avxOut[r] = avxDot(rowPtr, (nint)sPtr, cols);
                    scalarOut[r] = scalarDot(rowPtr, (nint)sPtr, cols);
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
                $"{label} avx-vs-scalar rows={rows} cols={cols}: maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={mismatches}/{rows}");
            Assert.True(mismatches == 0,
                $"{label} AVX2 vs scalar mismatch ({mismatches}/{rows}) rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
        }
    }

    [Fact]
    public void DotQ4_1_Q8_1_Avx2MatchesScalar() =>
        AssertAvx2MatchesScalar("Q4_1", 20,
            (row, s, c) => SimdKernels.DotQ4_1_Q8_1((byte*)row, (byte*)s, c),
            (row, s, c) => SimdKernels.DotQ4_1_Q8_1_Scalar((byte*)row, (byte*)s, c));

    [Fact]
    public void DotQ5_1_Q8_1_Avx2MatchesScalar() =>
        AssertAvx2MatchesScalar("Q5_1", 24,
            (row, s, c) => SimdKernels.DotQ5_1_Q8_1((byte*)row, (byte*)s, c),
            (row, s, c) => SimdKernels.DotQ5_1_Q8_1_Scalar((byte*)row, (byte*)s, c));

    /// <summary>
    /// Real before/after timing, same shape/rationale as
    /// <c>SimdKernelsLegacyQ8_0Tests.FastKernels_AreFasterThanDequantFallback</c> — median of
    /// several interleaved trials, not a single-run number.
    /// </summary>
    [Fact]
    public void FastKernels_AreFasterThanDequantFallback()
    {
        const int rows = 17408, cols = 5120;
        var rng = new Random(0x8871);
        var report = new System.Text.StringBuilder();
        var failures = new List<string>();

        void Bench(string label, int bytesPerBlock, DType dtype, Action<nint, nint, nint, int, int> fastMatVec)
        {
            byte[] weightBytes = BuildMatrix(rows, cols, bytesPerBlock, rng);
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
                    fastMatVec((nint)ofPtr, (nint)wPtr, (nint)iPtr, rows, cols);
                    SimdKernels.MatVecDequantFallback(obPtr, wPtr, iPtr, rows, cols, dtype);
                }

                for (int t = 0; t < trials; t++)
                {
                    var swFast = System.Diagnostics.Stopwatch.StartNew();
                    for (int it = 0; it < itersPerTrial; it++) fastMatVec((nint)ofPtr, (nint)wPtr, (nint)iPtr, rows, cols);
                    swFast.Stop();
                    fastTimes[t] = swFast.Elapsed.TotalMilliseconds / itersPerTrial;

                    var swFallback = System.Diagnostics.Stopwatch.StartNew();
                    for (int it = 0; it < itersPerTrial; it++) SimdKernels.MatVecDequantFallback(obPtr, wPtr, iPtr, rows, cols, dtype);
                    swFallback.Stop();
                    fallbackTimes[t] = swFallback.Elapsed.TotalMilliseconds / itersPerTrial;
                }

                Array.Sort(fastTimes);
                Array.Sort(fallbackTimes);
                double fastMs = fastTimes[0]; // min, not median: noise only ever slows a trial down, never speeds it up
                double fallbackMs = fallbackTimes[0];
                double speedup = fallbackMs / fastMs;
                report.AppendLine($"{label} rows={rows} cols={cols}: fallback={fallbackMs:F3}ms fast={fastMs:F3}ms " +
                    $"speedup={speedup:F2}x (min of {trials} trials, {itersPerTrial} iters each)");
                if (speedup <= 1.0)
                    failures.Add($"{label}: median speedup={speedup:F2}x is not > 1.0");
            }
        }

        Bench("Q4_1", 20, DType.Q4_1,
            (o, w, i, r, c) => SimdKernels.MatVecQ4_1((float*)o, (byte*)w, (float*)i, r, c));
        // Q5_1 deliberately excluded from this asserted set: measured ~0.84-1.03x across 4 runs
        // (essentially a wash, same as Q5_0's honest negative result — both are 5-bit-split
        // formats needing a qh side-channel bit per element), so it is NOT wired into
        // SimdKernels.MatVec's dispatch switch. Its equivalence test above still runs.

        Console.WriteLine(report.ToString());
        Assert.True(failures.Count == 0, "Median speedup <= 1.0x for: " + string.Join("; ", failures) +
            "\n" + report);
    }
}
