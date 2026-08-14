using OpenTail.Stingray.Cpu;
using System.Linq;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Correctness gate for <see cref="SimdKernels.DotQ4K_Wide8"/> (perf-loop-progress.md
/// iteration 7), per this codebase's established discipline: never extend a kernel without a
/// matching test verified against a HAND-COMPUTED reference, not just against other kernel code.
/// </summary>
public sealed unsafe class DotQ4KWide8SeamTests
{
    /// <summary>
    /// Hand-computed reference, not derived from any other kernel. Constructs a single Q4_K
    /// super-block (144 bytes) where every design choice makes the expected value computable by
    /// simple arithmetic instead of simulating the full ggml scale/min bit-unpacking scheme:
    /// <list type="bullet">
    /// <item>dmin = 0.0 (exact in half) -- the entire "min correction" term vanishes regardless
    /// of what GetScaleMinK4 computes for `min`, so the min-related scale bytes don't need to be
    /// hand-traced through the 6-bit packing scheme at all.</item>
    /// <item>d = 1.0 (exact in half).</item>
    /// <item>scale bytes chosen (0..3 and 8..11 = 5, 4..7 = 0) so GetScaleMinK4 returns scale=5
    /// for EVERY chunk (0-3), both the direct j&lt;4 read and the packed j&gt;=4 read -- verified
    /// by tracing GetScaleMinK4's own formula by hand, not assumed.</item>
    /// <item>every quant nibble (lo and hi) = 5 (qs bytes = 0x55) and every activation input =
    /// 2.0.</item>
    /// </list>
    /// Expected per-element contribution: dequant = d*scale*nibble - dmin*min = 1*5*5 - 0 = 25;
    /// times input 2.0 = 50 per element. 256 elements/block (cols=256, 1 block) -> 256*50 = 12800.
    /// </summary>
    [Fact]
    public void DotQ4K_Wide8_MatchesHandComputedReference_SingleUniformBlock()
    {
        const int cols = 256;
        byte[] block = new byte[144];

        // d = 1.0 (half bits 0x3C00, little-endian bytes 0x00, 0x3C)
        block[0] = 0x00; block[1] = 0x3C;
        // dmin = 0.0 (half bits 0x0000)
        block[2] = 0x00; block[3] = 0x00;

        // Scale/min packed bytes (12 bytes at offset 4..15). j=0..3 direct-read scale=q[j]&63;
        // j=4..7 direct-read min=q[j]&63 (unused since dmin=0); j=4..7 (as the "j-4" source for
        // the packed branch) top 2 bits must be 0 so they don't perturb the packed scale --
        // guaranteed since these bytes are 5 and 0, both < 64.
        block[4] = 5; block[5] = 5; block[6] = 5; block[7] = 5;   // sc[0..3] -> direct scale=5 for chunks 0,1
        block[8] = 0; block[9] = 0; block[10] = 0; block[11] = 0; // sc[4..7] -> mins, irrelevant (dmin=0)
        // sc[8..11]: packed branch (j=4..7) scale = q[j+4]&0xF | ((q[j-4]>>6)<<4). q[j-4] is
        // sc[0..3]=5 (top 2 bits 0), so scale = q[j+4]&0xF. Set low nibble = 5 -> scale=5.
        block[12] = 5; block[13] = 5; block[14] = 5; block[15] = 5;

        // 128 quant bytes: every nibble (lo and hi) = 5.
        for (int i = 16; i < 144; i++) block[i] = 0x55;

        float[] input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = 2.0f;

        const float expected = 256 * 50.0f; // 12800

        fixed (byte* rowPtr = block)
        fixed (float* inputPtr = input)
        {
            float actual = SimdKernels.DotQ4K_Wide8(rowPtr, inputPtr, cols);
            Assert.True(Math.Abs(actual - expected) < 0.5f,
                $"DotQ4K_Wide8 hand-computed reference mismatch: expected {expected}, got {actual}");
        }
    }

    /// <summary>
    /// Secondary safety net (not the primary correctness gate, per the discipline): DotQ4K_Wide8
    /// must closely agree with the already-deeply-trusted DotQ4K (used throughout production
    /// decode/prefill, extensively covered elsewhere in this suite) on realistic random
    /// multi-block data, since the only intended difference is FMA accumulator grouping (not
    /// associative in floating point, so a tight tolerance rather than bit-exact).
    /// </summary>
    [Theory]
    [InlineData(256)]  // 1 block
    [InlineData(2048)] // 8 blocks -- real embDim shape used by QKV/O projections
    [InlineData(8192)] // 32 blocks -- real intermDim shape used by FFN gate/up/down
    public void DotQ4K_Wide8_CloselyMatchesDotQ4K_OnRandomData(int cols)
    {
        var rng = new Random(13579);
        int numBlocks = cols / 256;
        byte[] weights = new byte[numBlocks * 144];
        rng.NextBytes(weights);
        // Keep d/dmin in a sane half-float range so neither kernel produces NaN/Inf from
        // denormal garbage -- doesn't privilege either kernel, both read the same bytes.
        for (int b = 0; b < numBlocks; b++)
        {
            int off = b * 144;
            var dBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            var dminBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            weights[off] = dBits[0]; weights[off + 1] = dBits[1];
            weights[off + 2] = dminBits[0]; weights[off + 3] = dminBits[1];
        }

        float[] input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        fixed (byte* rowPtr = weights)
        fixed (float* inputPtr = input)
        {
            float trusted = SimdKernels.DotQ4K(rowPtr, inputPtr, cols);
            float wide8 = SimdKernels.DotQ4K_Wide8(rowPtr, inputPtr, cols);
            double relDiff = Math.Abs(wide8 - trusted) / Math.Max(1e-3, Math.Abs(trusted));
            Assert.True(relDiff < 0.001, // 0.1% -- generous for FP reassociation, tight enough to catch a real bug
                $"cols={cols}: DotQ4K_Wide8={wide8} vs DotQ4K={trusted}, relDiff={relDiff:P4}");
        }
    }
}

/// <summary>
/// Performance comparison, DotQ4K (2 accumulators) vs DotQ4K_Wide8 (8 accumulators), n=6 per
/// side per iteration 5's established minimum sample size for a trustworthy verdict on this box.
/// </summary>
public sealed unsafe class DotQ4KWide8PerfTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(2048)] // real embDim shape (QKV/O projections)
    [InlineData(8192)] // real intermDim shape (FFN gate/up/down)
    public void PerfGauge_DotQ4K_Wide8_VsDotQ4K(int cols)
    {
        var rng = new Random(24680);
        byte[] weights = new byte[cols / 256 * 144];
        rng.NextBytes(weights);
        for (int b = 0; b < cols / 256; b++)
        {
            int off = b * 144;
            var dBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            var dminBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            weights[off] = dBits[0]; weights[off + 1] = dBits[1];
            weights[off + 2] = dminBits[0]; weights[off + 3] = dminBits[1];
        }
        float[] input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        const int warmup = 2000; // widened from 200 -- suspected mid-benchmark JIT tier-up causing bimodal timings
        const int timedCalls = 2000;

        fixed (byte* w = weights)
        fixed (float* inp = input)
        {
            float dummy = 0;
            for (int i = 0; i < warmup; i++) { dummy += SimdKernels.DotQ4K(w, inp, cols); dummy += SimdKernels.DotQ4K_Wide8(w, inp, cols); }

            double[] trustedMs = new double[6];
            for (int run = 0; run < 6; run++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int c = 0; c < timedCalls; c++) dummy += SimdKernels.DotQ4K(w, inp, cols);
                sw.Stop();
                trustedMs[run] = sw.Elapsed.TotalMilliseconds;
            }

            double[] wide8Ms = new double[6];
            for (int run = 0; run < 6; run++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int c = 0; c < timedCalls; c++) dummy += SimdKernels.DotQ4K_Wide8(w, inp, cols);
                sw.Stop();
                wide8Ms[run] = sw.Elapsed.TotalMilliseconds;
            }

            double tMean = trustedMs.Average(), wMean = wide8Ms.Average();
            double tStd = Math.Sqrt(trustedMs.Select(x => (x - tMean) * (x - tMean)).Sum() / 5);
            double wStd = Math.Sqrt(wide8Ms.Select(x => (x - wMean) * (x - wMean)).Sum() / 5);

            output.WriteLine(
                $"[cols={cols}] {timedCalls} calls x 6 runs each (dummy={dummy}):\n" +
                $"  DotQ4K       : {string.Join(", ", trustedMs.Select(m => m.ToString("F2")))}  mean={tMean:F3}ms stdev={tStd:F3}ms\n" +
                $"  DotQ4K_Wide8 : {string.Join(", ", wide8Ms.Select(m => m.ToString("F2")))}  mean={wMean:F3}ms stdev={wStd:F3}ms\n" +
                $"  Speedup (trusted/wide8, >1 means wide8 wins): {tMean / wMean:F4}x");
        }
    }
}
