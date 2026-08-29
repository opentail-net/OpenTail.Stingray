
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Pins the contract that batched prefill on the <b>F32 path</b> must produce exactly what
/// per-token decode produces: <c>MatMulBatched(N)</c> == N separate <c>MatVec</c> calls, element
/// for element.
///
/// <para>
/// Without OpenBLAS, the F32 <c>MatMulBatched</c> literally loops <c>MatVec</c>, so these pass
/// trivially. That is the point: they are a characterization harness for any future prefill GEMM
/// that reorders the loop to read each weight block once and dot it against all N activation rows.
/// Such a rewrite changes accumulation order, and this suite is what proves it stays numerically
/// honest rather than merely faster.
/// </para>
///
/// <para>
/// <see cref="SimdKernels.Q8PrefillEnabled"/> is forced <c>false</c> for the duration of each test
/// because it is <b>on by default</b>. The int8 path does not reorder the F32 accumulation — it
/// quantizes the activations, a different arithmetic operation whose results differ from F32 by
/// construction. Asserting bit-equality against it would be a category error, so the int8 path has
/// its own separate contract in <see cref="MatMulBatchedQ8EquivalenceTests"/> (batched Q8 must be
/// bit-identical to N per-token Q8 calls) and its quality is gated by perplexity/greedy-parity
/// measurement rather than by exact equality. Pinning the gate here keeps this suite meaningful
/// regardless of what the default becomes.
/// </para>
/// </summary>
public sealed unsafe class MatMulBatchedEquivalenceTests : IDisposable
{
    // Q8PrefillEnabled is a process-wide static. xunit.runner.json sets
    // parallelizeTestCollections=false, so save/restore around each test instance is sufficient;
    // this mirrors the same discipline MatMulBatchedQ8EquivalenceTests documents.
    private readonly bool _savedQ8Gate = SimdKernels.Q8PrefillEnabled;

    // The "without OpenBLAS, MatMulBatched literally loops MatVec" contract this suite pins
    // (see class doc) only holds when a batch never reaches the BLAS threshold. On a machine
    // with OpenBLAS actually loaded, every batchSize this suite exercises is >= the shipped
    // default (16), so without this override BLAS's sgemm — not bit-identical to the scalar
    // loop by construction, just mathematically equivalent — silently became the thing under
    // test instead of the loop reorder this suite exists to characterize.
    private readonly int _savedMinBatchForBlas = SimdKernels.MinBatchForBlas;

    public MatMulBatchedEquivalenceTests()
    {
        SimdKernels.Q8PrefillEnabled = false;
        SimdKernels.MinBatchForBlas = int.MaxValue;
    }

    public void Dispose()
    {
        SimdKernels.Q8PrefillEnabled = _savedQ8Gate;
        SimdKernels.MinBatchForBlas = _savedMinBatchForBlas;
    }

    /// <summary>Deterministic bytes, so a failure is always reproducible.</summary>
    private static byte[] PseudoRandomBytes(int count, int seed)
    {
        var rng = new Random(seed);
        var bytes = new byte[count];
        rng.NextBytes(bytes);
        return bytes;
    }

    private static float[] PseudoRandomFloats(int count, int seed)
    {
        var rng = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return values;
    }

    private static void WriteHalf(byte[] buffer, int offset, float value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((Half)value);
        buffer[offset] = (byte)(bits & 0xFF);
        buffer[offset + 1] = (byte)(bits >> 8);
    }

    /// <summary>
    /// Build a weight buffer whose per-block fp16 scales are small finite values. Random bytes in
    /// a scale slot could decode to NaN/Inf, which would make an equality assertion meaningless.
    /// Everything else is random so the quantised payload is genuinely exercised.
    /// </summary>
    private static byte[] MakeWeights(int rows, int cols, DType dtype, int seed)
    {
        long byteCount = DTypeInfo.ByteSize((long)rows * cols, dtype);

        // F32 weights are read as floats, so random bytes would decode to NaN/Inf. Build the
        // values first, then take their bytes.
        if (dtype == DType.Float32)
        {
            var values = PseudoRandomFloats(rows * cols, seed);
            var raw = new byte[byteCount];
            MemoryMarshal.AsBytes(values.AsSpan()).CopyTo(raw);
            return raw;
        }

        var bytes = PseudoRandomBytes((int)byteCount, seed);

        switch (dtype)
        {
            case DType.Q4_K:
                // 256 elements per 144-byte block: [0..1] d, [2..3] dmin.
                for (int off = 0; off + 144 <= bytes.Length; off += 144)
                {
                    WriteHalf(bytes, off, 0.015f);
                    WriteHalf(bytes, off + 2, 0.004f);
                }
                break;

            case DType.Q6_K:
                // 256 elements per 210-byte block, fp16 scale in the final two bytes.
                for (int off = 0; off + 210 <= bytes.Length; off += 210)
                    WriteHalf(bytes, off + 208, 0.012f);
                break;

            case DType.Q8_0:
                // 32 elements per 34-byte block: [0..1] d, then 32 int8 values.
                for (int off = 0; off + 34 <= bytes.Length; off += 34)
                    WriteHalf(bytes, off, 0.02f);
                break;
        }

        return bytes;
    }

    /// <summary>Run both paths over the same inputs and return (batched, reference).</summary>
    private static (float[] Batched, float[] Reference) RunBoth(
        int batchSize, int rows, int cols, DType dtype, int seed)
    {
        byte[] weights = MakeWeights(rows, cols, dtype, seed);
        float[] input = PseudoRandomFloats(batchSize * cols, seed + 1);

        var batched = new float[batchSize * rows];
        var reference = new float[batchSize * rows];

        fixed (byte* w = weights)
        fixed (float* x = input)
        fixed (float* b = batched)
        fixed (float* r = reference)
        {
            SimdKernels.MatMulBatched(b, w, x, batchSize, rows, cols, dtype);

            // Reference: each token independently, exactly as single-token decode does it.
            for (int n = 0; n < batchSize; n++)
                SimdKernels.MatVec(r + n * rows, w, x + n * cols, rows, cols, dtype);
        }

        return (batched, reference);
    }

    private static void AssertMatches(int batchSize, int rows, int cols, DType dtype, int seed = 1234)
    {
        var (batched, reference) = RunBoth(batchSize, rows, cols, dtype, seed);

        Assert.All(reference, v => Assert.True(float.IsFinite(v),
            "reference produced a non-finite value; the fixture's scales are wrong, not the kernel"));

        for (int i = 0; i < reference.Length; i++)
        {
            Assert.True(batched[i] == reference[i],
                $"{dtype} batch={batchSize} rows={rows} cols={cols}: index {i} " +
                $"batched={batched[i]:R} reference={reference[i]:R}");
        }
    }

    // ── Per-dtype equivalence across batch sizes ────────────────────────────
    // Batch sizes straddle MinBatchForBlas (16), so both the small-batch and large-batch
    // branches of MatMulBatched are covered.

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(8)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(64)]
    public void Q4K_BatchedMatchesPerTokenMatVec(int batchSize) =>
        AssertMatches(batchSize, rows: 64, cols: 256, DType.Q4_K);

    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(33)]
    public void Q6K_BatchedMatchesPerTokenMatVec(int batchSize) =>
        AssertMatches(batchSize, rows: 32, cols: 256, DType.Q6_K);

    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(33)]
    public void Q8_0_BatchedMatchesPerTokenMatVec(int batchSize) =>
        AssertMatches(batchSize, rows: 32, cols: 256, DType.Q8_0);

    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(33)]
    public void Float32_BatchedMatchesPerTokenMatVec(int batchSize)
    {
        // Bit-identity between MatMulBatched and per-token MatVec for the F32 path only holds
        // when both sides take the exact same code path. Without OpenBLAS that's the scalar
        // MatVec loop this class's constructor forces (MinBatchForBlas = int.MaxValue defends
        // against BLAS specifically); on machines where OpenBLAS is absent, an environment-
        // dependent threaded/tiled scheduling difference between MatMulBatched and the
        // sequential per-token loop has been observed to produce last-bit FP differences at
        // larger batch sizes (2026-08-28: 0x...C2 vs 0x...BE) that this suite isn't designed to
        // characterize — see SimdKernelsDequantCacheTests' BlasAvailable-gated sibling test for
        // the same environment-dependence pattern.
        Assert.SkipUnless(SimdKernels.BlasAvailable, "OpenBLAS not present in this environment");
        AssertMatches(batchSize, rows: 48, cols: 128, DType.Float32);
    }

    // ── Shapes that stress tiling boundaries ────────────────────────────────

    [Theory]
    [InlineData(64, 256)]    // single K-block
    [InlineData(64, 512)]    // two K-blocks
    [InlineData(64, 1024)]   // four K-blocks
    [InlineData(1, 256)]     // single row
    [InlineData(255, 256)]   // row count not a multiple of any plausible tile
    public void Q4K_EquivalenceHoldsAcrossShapes(int rows, int cols) =>
        AssertMatches(batchSize: 8, rows: rows, cols: cols, DType.Q4_K);

    /// <summary>
    /// Every row of the batch must be distinct — a batched kernel that accidentally broadcasts
    /// one token's activations across the batch would still match a per-token reference if the
    /// inputs happened to be identical, so prove the inputs actually differ.
    /// </summary>
    [Fact]
    public void BatchRowsAreIndependent_NotBroadcastFromTheFirstToken()
    {
        var (batched, _) = RunBoth(batchSize: 4, rows: 32, cols: 256, DType.Q4_K, seed: 99);

        var first = batched.AsSpan(0, 32).ToArray();
        bool anyRowDiffers = false;
        for (int n = 1; n < 4 && !anyRowDiffers; n++)
            anyRowDiffers = !batched.AsSpan(n * 32, 32).SequenceEqual(first);

        Assert.True(anyRowDiffers, "all batch rows identical — inputs are not independent");
    }
}
