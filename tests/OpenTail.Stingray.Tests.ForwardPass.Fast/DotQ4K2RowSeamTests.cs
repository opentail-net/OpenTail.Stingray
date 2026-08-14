using OpenTail.Stingray.Cpu;
using System.Linq;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Correctness gate for <see cref="SimdKernels.DotQ4K_2Row"/> (perf-loop-progress.md
/// iteration 24, Next-up item 4 "memory-level parallelism"), per this codebase's established
/// discipline: never extend a kernel without a matching test verified against a HAND-COMPUTED
/// reference, not just against other kernel code.
/// </summary>
public sealed unsafe class DotQ4K2RowSeamTests
{
    /// <summary>
    /// Same hand-computed single-superblock construction as
    /// <c>DotQ4KWide8SeamTests.DotQ4K_Wide8_MatchesHandComputedReference_SingleUniformBlock</c>
    /// (dmin=0, d=1, every scale byte traced by hand to yield scale=5, every nibble=5, every
    /// input=2.0 -> expected 25 dequant/element * 2.0 input = 50/element * 256 elements = 12800),
    /// applied identically to both rows (row1 built as a byte-for-byte copy of row0, with a
    /// distinct expected value only insofar as it's independently verifiable, not derived from
    /// row0's result).
    /// </summary>
    [Fact]
    public void DotQ4K_2Row_MatchesHandComputedReference_SingleUniformBlock_BothRowsIdentical()
    {
        const int cols = 256;
        byte[] block = new byte[144];

        block[0] = 0x00; block[1] = 0x3C; // d = 1.0
        block[2] = 0x00; block[3] = 0x00; // dmin = 0.0
        block[4] = 5; block[5] = 5; block[6] = 5; block[7] = 5;
        block[8] = 0; block[9] = 0; block[10] = 0; block[11] = 0;
        block[12] = 5; block[13] = 5; block[14] = 5; block[15] = 5;
        for (int i = 16; i < 144; i++) block[i] = 0x55;

        float[] input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = 2.0f;

        const float expected = 256 * 50.0f; // 12800

        fixed (byte* rowPtr = block)
        fixed (float* inputPtr = input)
        {
            SimdKernels.DotQ4K_2Row(rowPtr, rowPtr, inputPtr, cols, out float out0, out float out1);
            Assert.True(Math.Abs(out0 - expected) < 0.5f,
                $"DotQ4K_2Row row0 hand-computed mismatch: expected {expected}, got {out0}");
            Assert.True(Math.Abs(out1 - expected) < 0.5f,
                $"DotQ4K_2Row row1 hand-computed mismatch: expected {expected}, got {out1}");
        }
    }

    /// <summary>
    /// Distinct-value variant of the hand-computed reference: row1's quant nibbles are all 3
    /// instead of 5 (scale bytes unchanged, still traced to scale=5), so its expected value is
    /// independently computable (d*scale*nibble = 1*5*3 = 15, *2.0 input = 30/element * 256 =
    /// 7680) and must differ from row0's -- catches any bug where the two rows' accumulators get
    /// cross-wired (e.g. row1's dequant multiplied into row0's accumulator by a copy-paste slip).
    /// </summary>
    [Fact]
    public void DotQ4K_2Row_MatchesHandComputedReference_DistinctRows()
    {
        const int cols = 256;
        byte[] block0 = new byte[144];
        block0[0] = 0x00; block0[1] = 0x3C;
        block0[2] = 0x00; block0[3] = 0x00;
        block0[4] = 5; block0[5] = 5; block0[6] = 5; block0[7] = 5;
        block0[8] = 0; block0[9] = 0; block0[10] = 0; block0[11] = 0;
        block0[12] = 5; block0[13] = 5; block0[14] = 5; block0[15] = 5;
        for (int i = 16; i < 144; i++) block0[i] = 0x55; // nibble=5 both lo/hi

        byte[] block1 = (byte[])block0.Clone();
        for (int i = 16; i < 144; i++) block1[i] = 0x33; // nibble=3 both lo/hi

        float[] input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = 2.0f;

        const float expected0 = 256 * (1f * 5f * 5f) * 2.0f; // 12800
        const float expected1 = 256 * (1f * 5f * 3f) * 2.0f; // 7680

        fixed (byte* row0Ptr = block0)
        fixed (byte* row1Ptr = block1)
        fixed (float* inputPtr = input)
        {
            SimdKernels.DotQ4K_2Row(row0Ptr, row1Ptr, inputPtr, cols, out float out0, out float out1);
            Assert.True(Math.Abs(out0 - expected0) < 0.5f,
                $"row0 mismatch: expected {expected0}, got {out0}");
            Assert.True(Math.Abs(out1 - expected1) < 0.5f,
                $"row1 mismatch: expected {expected1}, got {out1}");
        }
    }

    /// <summary>
    /// Secondary safety net (not the primary correctness gate): DotQ4K_2Row's per-row outputs
    /// must closely match calling the already-deeply-trusted DotQ4K on each row independently,
    /// on realistic random multi-block/multi-row data. Not bit-exact -- the accumulation order
    /// differs slightly (interleaved with the other row's chain vs standalone), so FP
    /// reassociation gives a tight-but-not-zero tolerance, same bar as DotQ4K_Wide8's own
    /// cross-check.
    /// </summary>
    [Theory]
    [InlineData(256)]  // 1 block
    [InlineData(2048)] // 8 blocks -- real embDim shape (QKV/O projections)
    [InlineData(8192)] // 32 blocks -- real intermDim shape (FFN gate/up/down)
    public void DotQ4K_2Row_CloselyMatchesTwoSeparateDotQ4KCalls_OnRandomData(int cols)
    {
        var rng = new Random(97240);
        int numBlocks = cols / 256;
        byte[] row0 = new byte[numBlocks * 144];
        byte[] row1 = new byte[numBlocks * 144];
        rng.NextBytes(row0);
        rng.NextBytes(row1);
        foreach (var row in new[] { row0, row1 })
        {
            for (int b = 0; b < numBlocks; b++)
            {
                int off = b * 144;
                var dBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
                var dminBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
                row[off] = dBits[0]; row[off + 1] = dBits[1];
                row[off + 2] = dminBits[0]; row[off + 3] = dminBits[1];
            }
        }

        float[] input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        fixed (byte* r0 = row0)
        fixed (byte* r1 = row1)
        fixed (float* inputPtr = input)
        {
            float trusted0 = SimdKernels.DotQ4K(r0, inputPtr, cols);
            float trusted1 = SimdKernels.DotQ4K(r1, inputPtr, cols);
            SimdKernels.DotQ4K_2Row(r0, r1, inputPtr, cols, out float grouped0, out float grouped1);

            double relDiff0 = Math.Abs(grouped0 - trusted0) / Math.Max(1e-3, Math.Abs(trusted0));
            double relDiff1 = Math.Abs(grouped1 - trusted1) / Math.Max(1e-3, Math.Abs(trusted1));
            Assert.True(relDiff0 < 0.001,
                $"cols={cols}: row0 DotQ4K_2Row={grouped0} vs DotQ4K={trusted0}, relDiff={relDiff0:P4}");
            Assert.True(relDiff1 < 0.001,
                $"cols={cols}: row1 DotQ4K_2Row={grouped1} vs DotQ4K={trusted1}, relDiff={relDiff1:P4}");
        }
    }
}

/// <summary>
/// Performance comparison, two separate DotQ4K calls vs one DotQ4K_2Row call (shared input
/// loads), n=6 per side per iteration 5's established minimum sample size for a trustworthy
/// verdict on this box.
/// </summary>
public sealed unsafe class DotQ4K2RowPerfTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(2048)] // real embDim shape (QKV/O projections)
    [InlineData(8192)] // real intermDim shape (FFN gate/up/down)
    public void PerfGauge_DotQ4K_2Row_VsTwoSeparateDotQ4KCalls(int cols)
    {
        var rng = new Random(31415);
        int numBlocks = cols / 256;
        byte[] row0 = new byte[numBlocks * 144];
        byte[] row1 = new byte[numBlocks * 144];
        rng.NextBytes(row0);
        rng.NextBytes(row1);
        foreach (var row in new[] { row0, row1 })
        {
            for (int b = 0; b < numBlocks; b++)
            {
                int off = b * 144;
                var dBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
                var dminBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
                row[off] = dBits[0]; row[off + 1] = dBits[1];
                row[off + 2] = dminBits[0]; row[off + 3] = dminBits[1];
            }
        }
        float[] input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        const int warmup = 2000;
        const int timedCalls = 2000;

        fixed (byte* r0 = row0)
        fixed (byte* r1 = row1)
        fixed (float* inp = input)
        {
            float dummy = 0;
            for (int i = 0; i < warmup; i++)
            {
                dummy += SimdKernels.DotQ4K(r0, inp, cols) + SimdKernels.DotQ4K(r1, inp, cols);
                SimdKernels.DotQ4K_2Row(r0, r1, inp, cols, out float o0, out float o1);
                dummy += o0 + o1;
            }

            double[] separateMs = new double[6];
            for (int run = 0; run < 6; run++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int c = 0; c < timedCalls; c++)
                    dummy += SimdKernels.DotQ4K(r0, inp, cols) + SimdKernels.DotQ4K(r1, inp, cols);
                sw.Stop();
                separateMs[run] = sw.Elapsed.TotalMilliseconds;
            }

            double[] groupedMs = new double[6];
            for (int run = 0; run < 6; run++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int c = 0; c < timedCalls; c++)
                {
                    SimdKernels.DotQ4K_2Row(r0, r1, inp, cols, out float o0, out float o1);
                    dummy += o0 + o1;
                }
                sw.Stop();
                groupedMs[run] = sw.Elapsed.TotalMilliseconds;
            }

            double sMean = separateMs.Average(), gMean = groupedMs.Average();
            double sStd = Math.Sqrt(separateMs.Select(x => (x - sMean) * (x - sMean)).Sum() / 5);
            double gStd = Math.Sqrt(groupedMs.Select(x => (x - gMean) * (x - gMean)).Sum() / 5);

            output.WriteLine(
                $"[cols={cols}] {timedCalls} calls x 6 runs each (dummy={dummy}):\n" +
                $"  2x DotQ4K (separate) : {string.Join(", ", separateMs.Select(m => m.ToString("F2")))}  mean={sMean:F3}ms stdev={sStd:F3}ms\n" +
                $"  DotQ4K_2Row (grouped): {string.Join(", ", groupedMs.Select(m => m.ToString("F2")))}  mean={gMean:F3}ms stdev={gStd:F3}ms\n" +
                $"  Speedup (separate/grouped, >1 means grouped wins): {sMean / gMean:F4}x");
        }
    }
}
