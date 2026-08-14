using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Correctness gate for the int8 batched-prefill path (docs/cpu-prefill-plan.md §6 step 4).
///
/// <para>
/// This is deliberately a SEPARATE contract from <see cref="MatMulBatchedEquivalenceTests"/>,
/// not a relaxation of it. That suite still asserts <c>MatMulBatched(N)</c> is byte-identical
/// to N × <see cref="SimdKernels.MatVec"/> for the default (F32) path, and continues to pass
/// unmodified — <see cref="SimdKernels.TryMatMulBatchedQ8"/> is only reachable when
/// <see cref="SimdKernels.Q8PrefillEnabled"/> is explicitly set, so it doesn't touch that
/// contract at all. What this suite pins down instead: given the gate IS on, the batched Q8
/// path must be bit-identical to N independent per-token Q8 dot calls — i.e. batching activity
/// must not change the *quantized* answer, even though the quantized answer differs from the
/// F32 one by construction (that numeric gap is a separate, tolerance-based/perplexity concern
/// per docs §7, not what this suite checks).
/// </para>
/// </summary>
public sealed unsafe class MatMulBatchedQ8EquivalenceTests
{
    // Most tests below call TryMatMulBatchedQ8 directly and never touch the
    // Q8PrefillEnabled static gate at all. The two tests that DO need a specific gate
    // state save and restore it themselves (see GateOff_... and
    // SmallBatch_BelowThreshold_...) rather than this constructor setting it globally --
    // Q8PrefillEnabled is a process-wide static, and mutating it ambiently for every test in
    // this class would be a real cross-test race if this project's xunit.runner.json ever
    // turns parallelizeTestCollections back on (it's explicitly false today).

    private static byte[] MakeQ4KWeights(int rows, int cols, int seed)
    {
        int bytesPerRow = (cols / 256) * 144;
        var rng = new Random(seed);
        var bytes = new byte[rows * bytesPerRow];
        rng.NextBytes(bytes);
        for (int off = 0; off + 144 <= bytes.Length; off += 144)
        {
            WriteHalf(bytes, off, 0.015f);
            WriteHalf(bytes, off + 2, 0.004f);
        }
        return bytes;
    }

    private static byte[] MakeQ6KWeights(int rows, int cols, int seed)
    {
        int bytesPerRow = (cols / 256) * 210;
        var rng = new Random(seed);
        var bytes = new byte[rows * bytesPerRow];
        rng.NextBytes(bytes);
        for (int off = 0; off + 210 <= bytes.Length; off += 210)
            WriteHalf(bytes, off + 208, 0.012f);
        return bytes;
    }

    private static byte[] MakeQ3KWeights(int rows, int cols, int seed)
    {
        int bytesPerRow = (cols / 256) * 110;
        var rng = new Random(seed);
        var bytes = new byte[rows * bytesPerRow];
        rng.NextBytes(bytes);
        for (int off = 0; off + 110 <= bytes.Length; off += 110)
            WriteHalf(bytes, off + 108, 0.010f); // dAll, bytes 108-109 of the 110-byte block
        return bytes;
    }

    private static byte[] MakeQ2KWeights(int rows, int cols, int seed)
    {
        int bytesPerRow = (cols / 256) * 84;
        var rng = new Random(seed);
        var bytes = new byte[rows * bytesPerRow];
        rng.NextBytes(bytes);
        for (int off = 0; off + 84 <= bytes.Length; off += 84)
        {
            WriteHalf(bytes, off + 80, 0.010f); // d
            WriteHalf(bytes, off + 82, 0.004f); // dmin
        }
        return bytes;
    }

    private static void WriteHalf(byte[] buffer, int offset, float value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((Half)value);
        buffer[offset] = (byte)(bits & 0xFF);
        buffer[offset + 1] = (byte)(bits >> 8);
    }

    private static float[] PseudoRandomFloats(int count, int seed)
    {
        var rng = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return values;
    }

    /// <summary>Per-token Q4_K reference: N independent calls to the single-input Q8 dot.</summary>
    private static float[] ReferenceQ4K(byte[] weights, float[] input, int batchSize, int rows, int cols)
    {
        int bytesPerRow = (cols / 256) * 144;
        var result = new float[batchSize * rows];
        int scratchBytes = SimdKernels.Q8KSScratchBytes(cols);
        var scratch = new byte[scratchBytes];

        fixed (byte* w = weights)
        fixed (float* inp = input)
        fixed (byte* s = scratch)
        fixed (float* r = result)
        {
            for (int n = 0; n < batchSize; n++)
            {
                SimdKernels.QuantizeRowToQ8KS(inp + (long)n * cols, cols, s);
                for (int row = 0; row < rows; row++)
                    r[(long)n * rows + row] = SimdKernels.DotQ4K_Q8KS(w + (long)row * bytesPerRow, s, cols);
            }
        }
        return result;
    }

    private static float[] ReferenceQ6K(byte[] weights, float[] input, int batchSize, int rows, int cols)
    {
        int bytesPerRow = (cols / 256) * 210;
        var result = new float[batchSize * rows];
        int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
        var scratch = new byte[scratchBytes];

        fixed (byte* w = weights)
        fixed (float* inp = input)
        fixed (byte* s = scratch)
        fixed (float* r = result)
        {
            for (int n = 0; n < batchSize; n++)
            {
                SimdKernels.QuantizeRowToQ8K(inp + (long)n * cols, cols, s);
                for (int row = 0; row < rows; row++)
                    r[(long)n * rows + row] = SimdKernels.DotQ6K_Q8K(w + (long)row * bytesPerRow, s, cols);
            }
        }
        return result;
    }

    private static float[] ReferenceQ3K(byte[] weights, float[] input, int batchSize, int rows, int cols)
    {
        int bytesPerRow = (cols / 256) * 110;
        var result = new float[batchSize * rows];
        int scratchBytes = SimdKernels.Q8KSScratchBytes(cols);
        var scratch = new byte[scratchBytes];

        fixed (byte* w = weights)
        fixed (float* inp = input)
        fixed (byte* s = scratch)
        fixed (float* r = result)
        {
            for (int n = 0; n < batchSize; n++)
            {
                SimdKernels.QuantizeRowToQ8KS(inp + (long)n * cols, cols, s);
                for (int row = 0; row < rows; row++)
                    r[(long)n * rows + row] = SimdKernels.DotQ3K_Q8KS(w + (long)row * bytesPerRow, s, cols);
            }
        }
        return result;
    }

    private static float[] RunBatched(byte[] weights, float[] input, int batchSize, int rows, int cols, DType dtype)
    {
        var result = new float[batchSize * rows];
        fixed (byte* w = weights)
        fixed (float* inp = input)
        fixed (float* r = result)
        {
            bool ok = SimdKernels.TryMatMulBatchedQ8(r, w, inp, batchSize, rows, cols, dtype);
            Assert.True(ok, $"TryMatMulBatchedQ8 returned false for {dtype} -- dispatch mapping regressed");
        }
        return result;
    }

    [Theory]
    [InlineData(4)]   // exactly one group of 4
    [InlineData(8)]
    [InlineData(5)]   // one group of 4 + a single-token remainder
    [InlineData(6)]   // one group of 4 + a 2-token remainder
    [InlineData(7)]
    [InlineData(33)]  // several groups + remainder
    public void Q4K_BatchedMatchesPerTokenQ8Reference(int batchSize)
    {
        const int rows = 64, cols = 512;
        var weights = MakeQ4KWeights(rows, cols, seed: 7);
        var input = PseudoRandomFloats(batchSize * cols, seed: 8);

        var batched = RunBatched(weights, input, batchSize, rows, cols, DType.Q4_K);
        var reference = ReferenceQ4K(weights, input, batchSize, rows, cols);

        for (int i = 0; i < reference.Length; i++)
            Assert.True(batched[i] == reference[i],
                $"Q4_K batch={batchSize} index {i}: batched={batched[i]:R} reference={reference[i]:R}");
    }

    [Theory]
    [InlineData(4)]
    [InlineData(9)]
    [InlineData(33)]
    public void Q6K_BatchedMatchesPerTokenQ8Reference(int batchSize)
    {
        const int rows = 32, cols = 512;
        var weights = MakeQ6KWeights(rows, cols, seed: 11);
        var input = PseudoRandomFloats(batchSize * cols, seed: 12);

        var batched = RunBatched(weights, input, batchSize, rows, cols, DType.Q6_K);
        var reference = ReferenceQ6K(weights, input, batchSize, rows, cols);

        for (int i = 0; i < reference.Length; i++)
            Assert.True(batched[i] == reference[i],
                $"Q6_K batch={batchSize} index {i}: batched={batched[i]:R} reference={reference[i]:R}");
    }

    [Theory]
    [InlineData(4)]
    [InlineData(8)]   // exactly one group of 8 -- exercises the new _8In dispatch path
    [InlineData(9)]   // one group of 8 + a single-token remainder
    [InlineData(33)]  // several groups of 8 + remainder
    public void Q3K_BatchedMatchesPerTokenQ8Reference(int batchSize)
    {
        const int rows = 32, cols = 512;
        var weights = MakeQ3KWeights(rows, cols, seed: 51);
        var input = PseudoRandomFloats(batchSize * cols, seed: 52);

        var batched = RunBatched(weights, input, batchSize, rows, cols, DType.Q3_K);
        var reference = ReferenceQ3K(weights, input, batchSize, rows, cols);

        for (int i = 0; i < reference.Length; i++)
            Assert.True(batched[i] == reference[i],
                $"Q3_K batch={batchSize} index {i}: batched={batched[i]:R} reference={reference[i]:R}");
    }

    // ── Gate and fallback behaviour ──────────────────────────────────────────

    [Fact]
    public void GateOff_MatMulBatchedNeverCallsTheQ8Path()
    {
        // Belt-and-suspenders alongside MatMulBatchedEquivalenceTests: with the gate off,
        // the public entry point must reproduce the plain per-token result exactly, for a
        // dtype the Q8 path DOES support (Q4_K) -- proving the gate, not just dtype coverage,
        // is what's keeping the two paths apart.
        bool priorGate = SimdKernels.Q8PrefillEnabled;
        SimdKernels.Q8PrefillEnabled = false;
        try
        {
            const int rows = 64, cols = 512, batchSize = 8;
            var weights = MakeQ4KWeights(rows, cols, seed: 21);
            var input = PseudoRandomFloats(batchSize * cols, seed: 22);

            var viaMatMulBatched = new float[batchSize * rows];
            var viaPerToken = new float[batchSize * rows];
            fixed (byte* w = weights)
            fixed (float* inp = input)
            fixed (float* a = viaMatMulBatched)
            fixed (float* b = viaPerToken)
            {
                SimdKernels.MatMulBatched(a, w, inp, batchSize, rows, cols, DType.Q4_K);
                for (int n = 0; n < batchSize; n++)
                    SimdKernels.MatVec(b + n * rows, w, inp + n * cols, rows, cols, DType.Q4_K);
            }
            Assert.Equal(viaPerToken, viaMatMulBatched);
        }
        finally
        {
            SimdKernels.Q8PrefillEnabled = priorGate; // restore exactly what was there before
        }
    }

    [Theory]
    [InlineData(DType.Float32)]
    [InlineData(DType.Q5_K)]
    [InlineData(DType.Q2_K)]
    [InlineData(DType.Q8_0)] // has single-input Q8 dots but no _4In -- must still fall back
    public void UnsupportedDtype_FallsBackRatherThanCrashing(DType dtype)
    {
        const int rows = 64, cols = 512, batchSize = 8;
        int bytesPerRow = dtype switch
        {
            DType.Float32 => cols * sizeof(float),
            DType.Q5_K => (cols / 256) * 176,
            DType.Q2_K => (cols / 256) * 84,
            DType.Q8_0 => (cols / 32) * 34,
            _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
        };
        var weights = new byte[rows * bytesPerRow];
        new Random(5).NextBytes(weights);
        var input = PseudoRandomFloats(batchSize * cols, seed: 6);

        var result = new float[batchSize * rows];
        fixed (byte* w = weights)
        fixed (float* inp = input)
        fixed (float* r = result)
        {
            bool handled = SimdKernels.TryMatMulBatchedQ8(r, w, inp, batchSize, rows, cols, dtype);
            Assert.False(handled, $"{dtype} has no _4In kernel and must report false, not attempt it");
        }
    }

    // ── >512-token chunking (TryMatMulBatchedQ8's internal L2-cache-bounded split) ──────────
    //
    // Found via review, not by these tests (which didn't exist yet): the first version of the
    // chunking wrapper discarded each chunk's own TryResolveQ8Dispatch-derived return value and
    // unconditionally returned true, so an UNSUPPORTED dtype (Q5_K/Q2_K/Float32/Q8_0) with
    // batchSize > 512 silently skipped MatMulBatched's per-token fallback entirely, leaving the
    // whole output buffer uncomputed -- confirmed via a targeted repro (Q5_K, batchSize=600,
    // 9600/9600 output values never written) before being fixed. Neither existing test above
    // caught it: the batch-512 threshold was never exercised (correctness tests top out at 33;
    // the unsupported-dtype test uses batchSize=8, below the chunking threshold entirely).

    [Fact]
    public void Q4K_BatchGreaterThan512_ChunkedMatchesPerTokenQ8Reference()
    {
        const int rows = 8, cols = 256, batchSize = 600; // crosses the 512-token chunk boundary
        var weights = MakeQ4KWeights(rows, cols, seed: 41);
        var input = PseudoRandomFloats(batchSize * cols, seed: 42);

        var batched = RunBatched(weights, input, batchSize, rows, cols, DType.Q4_K);
        var reference = ReferenceQ4K(weights, input, batchSize, rows, cols);

        for (int i = 0; i < reference.Length; i++)
            Assert.True(batched[i] == reference[i],
                $"Q4_K chunked batch={batchSize} index {i}: batched={batched[i]:R} reference={reference[i]:R}");
    }

    [Fact]
    public void UnsupportedDtype_BatchGreaterThan512_TryMatMulBatchedQ8ReportsFalse()
    {
        // TryMatMulBatchedQ8 itself never runs a fallback -- that's MatMulBatched's job, gated
        // on this function's own return value (see the next test for the end-to-end check).
        // This pins down the half of the fix that matters here: chunking must not paper over
        // an unsupported dtype's honest "false" with an unconditional "true".
        const int rows = 8, cols = 256, batchSize = 600; // crosses the 512-token chunk boundary
        var weights = new byte[rows * (cols / 256) * 176]; // Q5_K bytes-per-row
        new Random(43).NextBytes(weights);
        var input = PseudoRandomFloats(batchSize * cols, seed: 44);

        var result = new float[batchSize * rows];
        fixed (byte* w = weights)
        fixed (float* inp = input)
        fixed (float* r = result)
        {
            bool handled = SimdKernels.TryMatMulBatchedQ8(r, w, inp, batchSize, rows, cols, DType.Q5_K);
            Assert.False(handled, "Q5_K has no int8 dot family and must report false even when chunked");
        }
    }

    [Fact]
    public void UnsupportedDtype_BatchGreaterThan512_MatMulBatchedStillComputesOutput()
    {
        // The actual regression: with Q8PrefillEnabled on, MatMulBatched only runs its
        // per-token MatVec fallback if TryMatMulBatchedQ8 reports false. Before the fix, the
        // >512-token chunking wrapper always reported true regardless of dtype support, so this
        // fallback never ran and `output` was left completely uncomputed. Repro'd directly
        // against this exact scenario (Q5_K, batchSize=600) before the fix: 9600/9600 sentinel
        // values survived untouched.
        bool priorGate = SimdKernels.Q8PrefillEnabled;
        SimdKernels.Q8PrefillEnabled = true;
        try
        {
            const int rows = 8, cols = 256, batchSize = 600;
            var weights = new byte[rows * (cols / 256) * 176]; // Q5_K bytes-per-row
            new Random(45).NextBytes(weights);
            var input = PseudoRandomFloats(batchSize * cols, seed: 46);

            var result = new float[batchSize * rows];
            for (int i = 0; i < result.Length; i++) result[i] = -999f; // sentinel: must not survive
            fixed (byte* w = weights)
            fixed (float* inp = input)
            fixed (float* r = result)
            {
                SimdKernels.MatMulBatched(r, w, inp, batchSize, rows, cols, DType.Q5_K);
            }
            Assert.DoesNotContain(-999f, result);
        }
        finally
        {
            SimdKernels.Q8PrefillEnabled = priorGate;
        }
    }

    [Fact]
    public void Q2K_BatchGreaterThan512_MatMulBatchedMatchesPerTokenFallback()
    {
        // Q2_K has no Q8 multi-input dot.  This exercises the production fallback at the
        // internal 512-token split, where an earlier wrapper falsely claimed unsupported dtypes
        // were handled and left output unwritten.
        bool priorGate = SimdKernels.Q8PrefillEnabled;
        // batchSize=600 is >= MinBatchForBlas, so on a machine with OpenBLAS actually loaded,
        // MatMulBatched would dequantize-and-sgemm instead of looping MatVec — a different
        // (mathematically equivalent, not bit-identical) kernel than the per-token reference
        // this test compares against. Push the threshold out of reach so this is genuinely a
        // "batched matches per-token, dtype fallback included" check, not an OpenBLAS-availability
        // coin flip.
        int priorMinBatchForBlas = SimdKernels.MinBatchForBlas;
        SimdKernels.Q8PrefillEnabled = true;
        SimdKernels.MinBatchForBlas = int.MaxValue;
        try
        {
            const int rows = 8, cols = 256, batchSize = 600;
            var weights = MakeQ2KWeights(rows, cols, seed: 47);
            var input = PseudoRandomFloats(batchSize * cols, seed: 48);
            var batched = new float[batchSize * rows];
            var reference = new float[batchSize * rows];
            fixed (byte* w = weights)
            fixed (float* inp = input)
            fixed (float* actual = batched)
            fixed (float* expected = reference)
            {
                SimdKernels.MatMulBatched(actual, w, inp, batchSize, rows, cols, DType.Q2_K);
                for (int n = 0; n < batchSize; n++)
                    SimdKernels.MatVec(expected + (long)n * rows, w, inp + (long)n * cols,
                        rows, cols, DType.Q2_K);
            }
            Assert.Equal(reference, batched);
        }
        finally
        {
            SimdKernels.Q8PrefillEnabled = priorGate;
            SimdKernels.MinBatchForBlas = priorMinBatchForBlas;
        }
    }

    /// <summary>
    /// A batch below <see cref="SimdKernels.MinBatchForQ8Prefill"/> must route to the per-token
    /// loop without ever entering <c>TryMatMulBatchedQ8</c>, even with the gate on and
    /// <c>allowQ8</c> requested. The threshold ships at 1 (so nothing is normally below it — that
    /// is deliberate, see the property's doc: a threshold above 1 would split a chunked prompt
    /// across two different numeric paths), so this raises it to prove the mechanism still works
    /// rather than asserting the shipped value.
    /// </summary>
    [Fact]
    public void BatchBelowMinBatchForQ8Prefill_SkipsTheQ8PathEntirely()
    {
        const int rows = 64, cols = 512, batchSize = 2;
        var weights = MakeQ4KWeights(rows, cols, seed: 31);
        var input = PseudoRandomFloats(batchSize * cols, seed: 32);

        bool priorGate = SimdKernels.Q8PrefillEnabled;
        int priorMin = SimdKernels.MinBatchForQ8Prefill;
        var viaMatMulBatched = new float[batchSize * rows];
        var viaPerToken = new float[batchSize * rows];
        try
        {
            SimdKernels.Q8PrefillEnabled = true;
            SimdKernels.MinBatchForQ8Prefill = batchSize + 1; // batch is now strictly below it

            fixed (byte* w = weights)
            fixed (float* inp = input)
            fixed (float* a = viaMatMulBatched)
            fixed (float* b = viaPerToken)
            {
                SimdKernels.MatMulBatched(a, w, inp, batchSize, rows, cols, DType.Q4_K, allowQ8: true);
                for (int n = 0; n < batchSize; n++)
                    SimdKernels.MatVec(b + n * rows, w, inp + n * cols, rows, cols, DType.Q4_K);
            }
        }
        finally
        {
            SimdKernels.Q8PrefillEnabled = priorGate;
            SimdKernels.MinBatchForQ8Prefill = priorMin;
        }
        Assert.Equal(viaPerToken, viaMatMulBatched);
    }

    /// <summary>
    /// The int8 path is opt-in per call site: a caller that does not pass <c>allowQ8</c> stays on
    /// F32 even with the gate on and a batch well above the threshold. This is what keeps batched
    /// <i>decode</i> (multi-user <c>BatchForwardMulti</c>, speculative <c>BatchVerify</c>) bit-exact
    /// with single-sequence decode — see <see cref="SimdKernels.MatMulBatched"/>'s allowQ8 doc.
    /// </summary>
    [Fact]
    public void GateOnButAllowQ8NotRequested_StaysOnTheF32Path()
    {
        const int rows = 64, cols = 512, batchSize = 8;
        var weights = MakeQ4KWeights(rows, cols, seed: 41);
        var input = PseudoRandomFloats(batchSize * cols, seed: 42);

        bool priorGate = SimdKernels.Q8PrefillEnabled;
        var defaulted = new float[batchSize * rows];
        var perToken = new float[batchSize * rows];
        var optedIn = new float[batchSize * rows];
        try
        {
            SimdKernels.Q8PrefillEnabled = true;
            fixed (byte* w = weights)
            fixed (float* inp = input)
            fixed (float* d = defaulted)
            fixed (float* b = perToken)
            fixed (float* o = optedIn)
            {
                SimdKernels.MatMulBatched(d, w, inp, batchSize, rows, cols, DType.Q4_K);
                SimdKernels.MatMulBatched(o, w, inp, batchSize, rows, cols, DType.Q4_K, allowQ8: true);
                for (int n = 0; n < batchSize; n++)
                    SimdKernels.MatVec(b + n * rows, w, inp + n * cols, rows, cols, DType.Q4_K);
            }
        }
        finally
        {
            SimdKernels.Q8PrefillEnabled = priorGate;
        }

        Assert.Equal(perToken, defaulted);
        // And prove the opt-in genuinely reaches a different kernel, so the assertion above is
        // testing the gate rather than a Q8 path that happens to be unreachable for this shape.
        Assert.NotEqual(perToken, optedIn);
    }
}
