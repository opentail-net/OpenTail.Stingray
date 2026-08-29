
namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Perf-loop iteration 2 (see docs/perf-loop-progress.md): the batch=256 GEMM-shape threading
/// investigation (real-avx2-gemm-port-plan.md, cpu-prefill-repack-gemm-plan.md) already ruled
/// out Parallel.For dispatch overhead as significant relative to per-call work at that scale.
/// Decode (batch=1) is a structurally different call pattern -- MANY more, MUCH smaller calls
/// (one MatVec per weight matrix per layer per token, ~5-6 calls x 24 layers x N tokens per
/// generation, confirmed by DecodeProfileTimers: FFN alone is ~65% of decode wall time). That
/// per-call dispatch overhead was never isolated at THIS call frequency and THESE row counts
/// before. This test measures it directly rather than reasoning about it further.
/// No kernel logic is touched or extended here (DotQ4K itself is untouched, already correctness-
/// verified elsewhere) -- this only compares two DISPATCH mechanisms around the same existing,
/// already-trusted kernel, so the seam-test-against-a-hand-computed-reference rule doesn't apply
/// (nothing new is being computed, just measured).
/// </summary>
public sealed unsafe class DecodeMatVecDispatchPerfTests(ITestOutputHelper output) : HeavyTestBase
{
    [Fact]
    public unsafe void PerfGauge_MatVecQ4K_ParallelForVsPersistentPool_DecodeShapes()
    {
        // Real SmolLM2-1.7B-Instruct-Q4_K_M shapes (confirmed via list-metadata):
        // embedding_length=2048, feed_forward_length=8192.
        RunShape("QKV/O projection (rows=2048, cols=2048)", rows: 2048, cols: 2048);
        RunShape("FFN gate/up (rows=8192, cols=2048)", rows: 8192, cols: 2048);
        RunShape("FFN down (rows=2048, cols=8192)", rows: 2048, cols: 8192);
    }

    private void RunShape(string label, int rows, int cols)
    {
        var rng = new Random(24680);
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 144; // Q4_K: 144 bytes/superblock
        byte[] weights = new byte[(long)rows * bytesPerRow];
        rng.NextBytes(weights);
        // Keep d/dmin (first 4 bytes of each 144-byte superblock) in a sane half-float range
        // so DotQ4K doesn't produce NaN/Inf that would slow down (or crash) on denormals --
        // doesn't affect the timing comparison, both dispatch paths call the identical kernel.
        for (int r = 0; r < rows; r++)
        for (int b = 0; b < blocksPerRow; b++)
        {
            int off = (r * blocksPerRow + b) * 144;
            var dBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            var dminBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            weights[off] = dBits[0]; weights[off + 1] = dBits[1];
            weights[off + 2] = dminBits[0]; weights[off + 3] = dminBits[1];
        }

        float[] input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] outputPf = new float[rows];
        float[] outputPersist = new float[rows];

        // §29 of cpu-prefill-repack-gemm-plan.md found Parallel.For-based paths need ~9+ calls
        // to reach tiered-JIT steady state; this test interleaves warmup between TWO dispatch
        // paths, so bump well past that per-path minimum to avoid measuring partial warmup.
        const int warmup = 60;
        const int timedCalls = 500; // matches roughly one real generation's per-layer call count

        fixed (byte* wPtr = weights)
        fixed (float* inPtr = input)
        fixed (float* outPfPtr = outputPf)
        fixed (float* outPersistPtr = outputPersist)
        {
            byte* w = wPtr; float* inp = inPtr; float* op = outPfPtr; float* opp = outPersistPtr;

            void RunParallelFor()
            {
                System.Threading.Tasks.Parallel.For(0, rows, i =>
                    op[i] = SimdKernels.DotQ4K(w + (long)i * bytesPerRow, inp, cols));
            }

            void RunPersistent()
            {
                PersistentThreadPool.For(rows, (from, to) =>
                {
                    for (int i = from; i < to; i++)
                        opp[i] = SimdKernels.DotQ4K(w + (long)i * bytesPerRow, inp, cols);
                });
            }

            for (int i = 0; i < warmup; i++) { RunParallelFor(); RunPersistent(); }

            double[] pfMs = new double[2];
            for (int run = 0; run < 2; run++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int c = 0; c < timedCalls; c++) RunParallelFor();
                sw.Stop();
                pfMs[run] = sw.Elapsed.TotalMilliseconds;
            }

            double[] persistMs = new double[2];
            for (int run = 0; run < 2; run++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int c = 0; c < timedCalls; c++) RunPersistent();
                sw.Stop();
                persistMs[run] = sw.Elapsed.TotalMilliseconds;
            }

            // Sanity: both dispatch paths must compute the SAME kernel result (same DotQ4K calls,
            // same inputs) -- not a correctness gate for DotQ4K itself (already verified
            // elsewhere), just a guard that neither dispatch wrapper corrupted the row range.
            double maxAbsDiff = 0;
            for (int i = 0; i < rows; i++)
                maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs(outputPf[i] - outputPersist[i]));
            Assert.True(maxAbsDiff < 1e-4, $"Dispatch paths disagree: maxAbsDiff={maxAbsDiff}");

            double pfCallsPerSec0 = timedCalls / (pfMs[0] / 1000.0);
            double pfCallsPerSec1 = timedCalls / (pfMs[1] / 1000.0);
            double persistCallsPerSec0 = timedCalls / (persistMs[0] / 1000.0);
            double persistCallsPerSec1 = timedCalls / (persistMs[1] / 1000.0);

            output.WriteLine(
                $"[{label}] {timedCalls} calls, {rows} rows x {cols} cols:\n" +
                $"  Parallel.For        : run1={pfMs[0]:F2}ms run2={pfMs[1]:F2}ms -> {pfCallsPerSec0:F0}/{pfCallsPerSec1:F0} calls/s ({pfMs[0] / timedCalls * 1000:F1}/{pfMs[1] / timedCalls * 1000:F1}us/call)\n" +
                $"  PersistentThreadPool: run1={persistMs[0]:F2}ms run2={persistMs[1]:F2}ms -> {persistCallsPerSec0:F0}/{persistCallsPerSec1:F0} calls/s ({persistMs[0] / timedCalls * 1000:F1}/{persistMs[1] / timedCalls * 1000:F1}us/call)\n" +
                $"  Speedup (persistent/parallel, >1 means persistent wins): run1={pfMs[0] / persistMs[0]:F3}x run2={pfMs[1] / persistMs[1]:F3}x");
        }
    }
}
