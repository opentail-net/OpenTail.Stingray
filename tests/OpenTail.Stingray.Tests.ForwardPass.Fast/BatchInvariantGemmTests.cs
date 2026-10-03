namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Batch-size invariance of the prefill GEMMs that <c>STINGRAY_CPU_DECODE_VIA_GEMM=1</c> sends decode
/// through (the TensorSharp ManagedQuantGemm idea, todo.md 2026-10-03): a token's output row must be
/// bitwise identical whether it is computed alone (N = 1, i.e. decode) or inside a batch of any size
/// (prefill, chunked prefill, speculative verify). That property is the whole point of routing decode
/// through the GEMM, so it is asserted exactly, not to a tolerance.
///
/// <para>Why it should hold: activations are quantized per row (Q8_K), ragged groups are padded with
/// zero rows that quantize to d = 0, and each output row accumulates independently of its neighbours.
/// These tests are what prove it instead of that reasoning.</para>
/// </summary>
public sealed unsafe class BatchInvariantGemmTests
{
    public static TheoryData<int> Batches() => new([2, 3, 4, 5, 7, 8, 13, 17, 33]);

    private static void FillQ4K(byte* w, int rows, int cols, Random rng)
    {
        int nb = cols / 256;
        long rowBytes = (long)nb * 144;
        for (long i = 0; i < rowBytes * rows; i++) w[i] = (byte)rng.Next(256);
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < nb; b++)
            {
                byte* blk = w + r * rowBytes + b * 144;
                short d = BitConverter.HalfToInt16Bits((Half)(rng.NextDouble() * 0.05 + 0.005));
                short m = BitConverter.HalfToInt16Bits((Half)(rng.NextDouble() * 0.02 + 0.001));
                blk[0] = (byte)(d & 0xFF); blk[1] = (byte)(d >> 8);
                blk[2] = (byte)(m & 0xFF); blk[3] = (byte)(m >> 8);
            }
    }

    private static void FillQ6K(byte* w, int rows, int cols, Random rng)
    {
        // block_q6_K: ql[128], qh[64], scales[16] (int8), d (fp16) = 210 bytes.
        int nb = cols / 256;
        long rowBytes = (long)nb * 210;
        for (long i = 0; i < rowBytes * rows; i++) w[i] = (byte)rng.Next(256);
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < nb; b++)
            {
                byte* blk = w + r * rowBytes + b * 210;
                short d = BitConverter.HalfToInt16Bits((Half)(rng.NextDouble() * 0.01 + 0.001));
                blk[208] = (byte)(d & 0xFF); blk[209] = (byte)(d >> 8);
            }
    }

    private delegate bool Gemm(float* output, float* input, int batch);

    /// <summary>Runs the batch once, then every row alone, and demands bit equality per row.</summary>
    private static void AssertRowInvariant(Gemm gemm, int batch, int rows, int cols, Random rng, string what)
    {
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * (long)cols * batch), 64);
        float* full = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * (long)rows * batch), 64);
        float* one = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * rows), 64);
        try
        {
            for (long i = 0; i < (long)cols * batch; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            x[3] = 7.5f; // an outlier, so rows differ in quantization scale
            Assert.True(gemm(full, x, batch), $"{what}: declined batch={batch}");
            int nonZero = 0;
            for (long i = 0; i < (long)rows * batch; i++) if (full[i] != 0f && float.IsFinite(full[i])) nonZero++;
            Assert.True(nonZero > rows * batch / 2, $"{what}: output mostly zero/non-finite ({nonZero}); the check would be vacuous");
            for (int t = 0; t < batch; t++)
            {
                Assert.True(gemm(one, x + (long)t * cols, 1), $"{what}: declined batch=1");
                for (int r = 0; r < rows; r++)
                {
                    float a = full[(long)t * rows + r], b = one[r];
                    if (BitConverter.SingleToInt32Bits(a) != BitConverter.SingleToInt32Bits(b))
                        Assert.Fail($"{what}: batch={batch} row t={t} out={r}: in-batch {a:R} vs alone {b:R}");
                }
            }
        }
        finally { NativeMemory.AlignedFree(x); NativeMemory.AlignedFree(full); NativeMemory.AlignedFree(one); }
    }

    [Theory]
    [MemberData(nameof(Batches))]
    public void Q4Kx8_RowResult_IsIndependentOfBatchSize(int batch)
    {
        const int rows = 64, cols = 512;
        if (!SimdKernels.CanRepackQ4Kx8(rows, cols)) return; // needs AVX2+FMA
        var rng = new Random(20261003 + batch);
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)((long)cols / 256 * 144 * rows), 64);
        byte* packed = (byte*)NativeMemory.AlignedAlloc((nuint)SimdKernels.Q4Kx8PackedBytes(rows, cols), 64);
        try
        {
            FillQ4K(w, rows, cols, rng);
            SimdKernels.RepackQ4KMatrix(w, packed, rows, cols);
            AssertRowInvariant((o, x, n) => SimdKernels.TryMatMulBatchedQ4Kx8(o, packed, x, n, rows, cols),
                batch, rows, cols, rng, "Q4Kx8");
        }
        finally { NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(packed); }
    }

    [Theory]
    [MemberData(nameof(Batches))]
    public void Q6K_RowResult_IsIndependentOfBatchSize(int batch)
    {
        const int rows = 64, cols = 512;
        var rng = new Random(20261004 + batch);
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)((long)cols / 256 * 210 * rows), 64);
        try
        {
            FillQ6K(w, rows, cols, rng);
            float* probe = stackalloc float[rows];
            float* px = stackalloc float[cols];
            if (!Q6KPrefillGemm.TryMatMulBatched(probe, w, px, 1, rows, cols)) return; // kernel unavailable here
            AssertRowInvariant((o, x, n) => Q6KPrefillGemm.TryMatMulBatched(o, w, x, n, rows, cols),
                batch, rows, cols, rng, "Q6K");
        }
        finally { NativeMemory.AlignedFree(w); }
    }

    /// <summary>The gate+up dual entry (one quantisation, one dispatch) must equal two single calls
    /// bit for bit at every batch size, including N = 1 (decode) and ragged groups.</summary>
    [Theory]
    [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(4)] [InlineData(5)] [InlineData(13)] [InlineData(64)]
    public void Q4Kx8Dual_IsBitIdenticalToTwoSingleCalls(int batch)
    {
        const int rows = 128, cols = 768;
        if (!SimdKernels.CanRepackQ4Kx8(rows, cols)) return;
        var rng = new Random(31 + batch);
        long wBytes = (long)cols / 256 * 144 * rows, pBytes = SimdKernels.Q4Kx8PackedBytes(rows, cols);
        byte* w1 = (byte*)NativeMemory.AlignedAlloc((nuint)wBytes, 64), w2 = (byte*)NativeMemory.AlignedAlloc((nuint)wBytes, 64);
        byte* p1 = (byte*)NativeMemory.AlignedAlloc((nuint)pBytes, 64), p2 = (byte*)NativeMemory.AlignedAlloc((nuint)pBytes, 64);
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * cols * batch), 64);
        var outs = new float*[4];
        for (int i = 0; i < 4; i++) outs[i] = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * rows * batch), 64);
        try
        {
            FillQ4K(w1, rows, cols, rng); FillQ4K(w2, rows, cols, rng);
            SimdKernels.RepackQ4KMatrix(w1, p1, rows, cols); SimdKernels.RepackQ4KMatrix(w2, p2, rows, cols);
            for (int i = 0; i < cols * batch; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

            Assert.True(SimdKernels.TryMatMulBatchedQ4Kx8Dual(outs[0], p1, outs[1], p2, x, batch, rows, cols), "dual declined");
            Assert.True(SimdKernels.TryMatMulBatchedQ4Kx8(outs[2], p1, x, batch, rows, cols));
            Assert.True(SimdKernels.TryMatMulBatchedQ4Kx8(outs[3], p2, x, batch, rows, cols));
            for (long i = 0; i < (long)rows * batch; i++)
            {
                Assert.Equal(BitConverter.SingleToInt32Bits(outs[2][i]), BitConverter.SingleToInt32Bits(outs[0][i]));
                Assert.Equal(BitConverter.SingleToInt32Bits(outs[3][i]), BitConverter.SingleToInt32Bits(outs[1][i]));
            }
            Assert.Contains(Enumerable.Range(0, rows * batch), i => outs[0][i] != 0f);
        }
        finally
        {
            NativeMemory.AlignedFree(w1); NativeMemory.AlignedFree(w2); NativeMemory.AlignedFree(p1);
            NativeMemory.AlignedFree(p2); NativeMemory.AlignedFree(x);
            foreach (var o in outs) NativeMemory.AlignedFree(o);
        }
    }

    /// <summary>Re-entrancy: GEMMs issued concurrently from many threads (as MoE expert loops do)
    /// must each get their own quantised activations; results must match the serial calls.</summary>
    [Fact]
    public void Q4Kx8_ConcurrentCalls_FromParallelLoop_MatchSerial()
    {
        const int rows = 64, cols = 512, jobs = 32;
        if (!SimdKernels.CanRepackQ4Kx8(rows, cols)) return;
        var rng = new Random(99);
        long wBytes = (long)cols / 256 * 144 * rows;
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)wBytes, 64);
        byte* p = (byte*)NativeMemory.AlignedAlloc((nuint)SimdKernels.Q4Kx8PackedBytes(rows, cols), 64);
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * cols * jobs * 3), 64);
        float* serial = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * rows * jobs * 3), 64);
        float* par = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * rows * jobs * 3), 64);
        try
        {
            FillQ4K(w, rows, cols, rng);
            SimdKernels.RepackQ4KMatrix(w, p, rows, cols);
            for (int i = 0; i < cols * jobs * 3; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            nint pp = (nint)p, px = (nint)x, ps = (nint)serial, ppar = (nint)par;
            // Job j: a batch of (j % 3) + 1 rows at its own offset.
            for (int j = 0; j < jobs; j++)
                SimdKernels.TryMatMulBatchedQ4Kx8((float*)ps + (long)j * 3 * rows, (byte*)pp, (float*)px + (long)j * 3 * cols, j % 3 + 1, rows, cols);
            for (int round = 0; round < 5; round++)
                Parallel.For(0, jobs, j =>
                    SimdKernels.TryMatMulBatchedQ4Kx8((float*)ppar + (long)j * 3 * rows, (byte*)pp, (float*)px + (long)j * 3 * cols, j % 3 + 1, rows, cols));
            for (int j = 0; j < jobs; j++)
                for (long i = 0; i < (long)(j % 3 + 1) * rows; i++)
                    Assert.Equal(BitConverter.SingleToInt32Bits(serial[(long)j * 3 * rows + i]),
                                 BitConverter.SingleToInt32Bits(par[(long)j * 3 * rows + i]));
        }
        finally
        {
            NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(p); NativeMemory.AlignedFree(x);
            NativeMemory.AlignedFree(serial); NativeMemory.AlignedFree(par);
        }
    }

    [Fact]
    public void Q4Kx8_RowInvariant_AtTrunkShape()
    {
        const int rows = 2048, cols = 2048, batch = 11;
        if (!SimdKernels.CanRepackQ4Kx8(rows, cols)) return;
        var rng = new Random(7);
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)((long)cols / 256 * 144 * rows), 64);
        byte* packed = (byte*)NativeMemory.AlignedAlloc((nuint)SimdKernels.Q4Kx8PackedBytes(rows, cols), 64);
        try
        {
            FillQ4K(w, rows, cols, rng);
            SimdKernels.RepackQ4KMatrix(w, packed, rows, cols);
            AssertRowInvariant((o, x, n) => SimdKernels.TryMatMulBatchedQ4Kx8(o, packed, x, n, rows, cols),
                batch, rows, cols, rng, "Q4Kx8 trunk");
        }
        finally { NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(packed); }
    }
}
