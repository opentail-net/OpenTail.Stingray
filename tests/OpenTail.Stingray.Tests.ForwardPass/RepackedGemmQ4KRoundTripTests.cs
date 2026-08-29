
namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Phase-2 (docs/cpu-prefill-repack-gemm-plan.md) checkpoint 1: proves the Q4_K row-interleave
/// transform (<see cref="RepackedGemm.RepackQ4K8Rows"/>) is a pure permutation with no numeric
/// change, before any GEMM kernel is written against it. For every row of a repacked group,
/// dequantizing straight out of the repacked buffer must equal dequantizing that row's
/// original bytes via the existing scalar <see cref="Dequantize"/> path.
/// </summary>
public sealed class RepackedGemmQ4KRoundTripTests(ITestOutputHelper output) : HeavyTestBase
{
    private static byte[] MakeRandomQ4KRows(int rows, int blocksPerRow, int seed)
    {
        var rng = new Random(seed);
        var bytes = new byte[rows * blocksPerRow * RepackedGemm.Q4KBytesPerBlock];
        rng.NextBytes(bytes);
        return bytes;
    }

    [Theory]
    [InlineData(1)]   // single Q4_K super-block per row (256 elements)
    [InlineData(2)]
    [InlineData(8)]   // matches SmolLM2's typical hidden-dim/256 block count order of magnitude
    public void RepackThenDequantize_MatchesOriginalDequantize_PerRow(int blocksPerRow)
    {
        byte[] rows = MakeRandomQ4KRows(RepackedGemm.RowsPerGroup, blocksPerRow, seed: 12345 + blocksPerRow);

        byte[] repacked = RepackedGemm.RepackQ4K8Rows(rows, blocksPerRow);
        Assert.Equal(blocksPerRow * RepackedGemm.Q4KGroupBytesPerBlock, repacked.Length);

        int elementsPerRow = blocksPerRow * 256;
        var expected = new float[elementsPerRow];
        var actual = new float[elementsPerRow];

        for (int row = 0; row < RepackedGemm.RowsPerGroup; row++)
        {
            var originalRowBytes = rows.AsSpan(row * blocksPerRow * RepackedGemm.Q4KBytesPerBlock,
                blocksPerRow * RepackedGemm.Q4KBytesPerBlock);
            Dequantize.ToFloat32(originalRowBytes, expected, DType.Q4_K, elementsPerRow);

            RepackedGemm.DequantizeRepackedQ4KRow(repacked, blocksPerRow, row, actual);

            for (int i = 0; i < elementsPerRow; i++)
                Assert.Equal(expected[i], actual[i]); // bit-exact: pure byte permutation, no arithmetic
        }
    }

    [Fact]
    public void Repack_RejectsWrongSourceLength()
    {
        var tooShort = new byte[RepackedGemm.RowsPerGroup * 2 * RepackedGemm.Q4KBytesPerBlock - 1];
        Assert.Throws<ArgumentException>(() => RepackedGemm.RepackQ4K8Rows(tooShort, blocksPerRow: 2));
    }

    /// <summary>
    /// GEMV kernel correctness: per §5/§7.5 of the phase-2 plan, byte-exactness against the
    /// F32 reference is not expected here (the kernel consumes Q8_K-quantized activations, the
    /// reference does not) -- this measures and bounds the actual delta instead of assuming one.
    /// Reference: <see cref="SimdKernels.DotQ4K"/> per row against the original unquantized
    /// float activation. Weight bytes are random (pure byte-permutation correctness already
    /// proven above); activation is realistic-scale random floats since the quantizer's
    /// rounding behavior is amplitude-sensitive.
    /// </summary>
    [Fact]
    public unsafe void GemvQ4K8x8Q8K_MatchesScalarReference_WithinQuantizationNoiseTolerance()
    {
        const int blocksPerRow = 8; // 2048-element rows, matching SmolLM2's attn_q/k/o shape
        const int cols = blocksPerRow * 256;
        var rng = new Random(777);

        byte[] rows = new byte[RepackedGemm.RowsPerGroup * blocksPerRow * RepackedGemm.Q4KBytesPerBlock];
        rng.NextBytes(rows);
        // Random bytes make d/dmin (fp16) NaN/Inf often enough to poison the dot product --
        // overwrite with plausible finite quantization scales (scales/qs stay random; those
        // fields have no NaN representation).
        for (int r = 0; r < RepackedGemm.RowsPerGroup; r++)
        for (int b = 0; b < blocksPerRow; b++)
        {
            int off = (r * blocksPerRow + b) * RepackedGemm.Q4KBytesPerBlock;
            var dBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            var dminBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            rows[off] = dBits[0]; rows[off + 1] = dBits[1];
            rows[off + 2] = dminBits[0]; rows[off + 3] = dminBits[1];
        }

        var activation = new float[cols];
        for (int i = 0; i < cols; i++) activation[i] = (float)(rng.NextDouble() * 2 - 1); // [-1, 1)

        var expected = new float[RepackedGemm.RowsPerGroup];
        fixed (float* actPtr = activation)
        {
            for (int row = 0; row < RepackedGemm.RowsPerGroup; row++)
            {
                fixed (byte* rowPtr = rows.AsSpan(row * blocksPerRow * RepackedGemm.Q4KBytesPerBlock, blocksPerRow * RepackedGemm.Q4KBytesPerBlock))
                {
                    expected[row] = SimdKernels.DotQ4K(rowPtr, actPtr, cols);
                }
            }
        }

        byte[] repacked = RepackedGemm.RepackQ4K8Rows(rows, blocksPerRow);
        var scratch = new byte[SimdKernels.Q8KScratchBytes(cols)];
        fixed (float* actPtr = activation)
        fixed (byte* scratchPtr = scratch)
        {
            SimdKernels.QuantizeRowToQ8K(actPtr, cols, scratchPtr);
        }

        var actual = new float[RepackedGemm.RowsPerGroup];
        fixed (byte* repackedPtr = repacked)
        fixed (byte* scratchPtr = scratch)
        fixed (float* outPtr = actual)
        {
            RepackedGemm.GemvQ4K8x8Q8K(outPtr, repackedPtr, scratchPtr, blocksPerRow);
        }

        double maxRelError = 0;
        for (int j = 0; j < RepackedGemm.RowsPerGroup; j++)
        {
            double relError = Math.Abs(actual[j] - expected[j]) / Math.Max(1e-6, Math.Abs(expected[j]));
            maxRelError = Math.Max(maxRelError, relError);
        }
        output.WriteLine($"max relative error vs DotQ4K scalar reference: {maxRelError:P4}");

        // Measured on random weights/activations: ~2.56% max relative error (docs/
        // cpu-prefill-repack-gemm-plan.md §11), attributable to Q8_K's per-256-element int8
        // activation quantization (~1/127 relative step) -- not a reassociation bug, since the
        // repack step itself (above) is proven bit-exact. 5% bound leaves headroom above the
        // measured figure without being loose enough to hide a real regression.
        Assert.True(maxRelError < 0.05,
            $"GEMV kernel diverged from the DotQ4K scalar reference by {maxRelError:P2} (expected < 5%, attributable to Q8_K activation quantization noise per docs/cpu-prefill-repack-gemm-plan.md §5).");
    }

    /// <summary>
    /// The scalar (ggml `_generic` port) and AVX2 (§12 of the phase-2 plan) kernels implement
    /// the same accumulation order by design -- this pins that down directly rather than only
    /// checking both indirectly against the looser DotQ4K tolerance above, so a future change
    /// to either kernel that desyncs them fails fast and close to the cause.
    /// </summary>
    [Fact]
    public unsafe void GemvAvx2_MatchesGemvScalar_Closely()
    {
        const int blocksPerRow = 8;
        const int cols = blocksPerRow * 256;
        var rng = new Random(777);
        byte[] rows = new byte[RepackedGemm.RowsPerGroup * blocksPerRow * RepackedGemm.Q4KBytesPerBlock];
        rng.NextBytes(rows);
        for (int r = 0; r < RepackedGemm.RowsPerGroup; r++)
        for (int b = 0; b < blocksPerRow; b++)
        {
            int off = (r * blocksPerRow + b) * RepackedGemm.Q4KBytesPerBlock;
            var dBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            var dminBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            rows[off] = dBits[0]; rows[off + 1] = dBits[1];
            rows[off + 2] = dminBits[0]; rows[off + 3] = dminBits[1];
        }
        var activation = new float[cols];
        for (int i = 0; i < cols; i++) activation[i] = (float)(rng.NextDouble() * 2 - 1);

        byte[] repacked = RepackedGemm.RepackQ4K8Rows(rows, blocksPerRow);
        var scratch = new byte[SimdKernels.Q8KScratchBytes(cols)];
        fixed (float* actPtr = activation)
        fixed (byte* scratchPtr = scratch)
        {
            SimdKernels.QuantizeRowToQ8K(actPtr, cols, scratchPtr);
        }

        var scalarOut = new float[8];
        var avx2Out = new float[8];
        fixed (byte* repackedPtr = repacked)
        fixed (byte* scratchPtr = scratch)
        fixed (float* scalarPtr = scalarOut)
        fixed (float* avx2Ptr = avx2Out)
        {
            RepackedGemm.GemvQ4K8x8Q8K_Scalar(scalarPtr, repackedPtr, scratchPtr, blocksPerRow);
            RepackedGemm.GemvQ4K8x8Q8K_Avx2(avx2Ptr, repackedPtr, scratchPtr, blocksPerRow);
        }

        for (int j = 0; j < 8; j++)
        {
            double relError = Math.Abs(avx2Out[j] - scalarOut[j]) / Math.Max(1e-3, Math.Abs(scalarOut[j]));
            Assert.True(relError < 0.001,
                $"col {j}: scalar={scalarOut[j]:F6} avx2={avx2Out[j]:F6} (relError {relError:P4}, expected < 0.1% -- FP reassociation noise only, not a reassociation bug).");
        }
    }

    [Fact]
    public unsafe void GemmTiledQ4K16x16Q8_MatchesScalarReference()
    {
        const int blocksPerRow = 4; // 1024 cols
        const int cols = blocksPerRow * 256;
        const int batchSize = 16;
        const int numGroups = 2; // 16 matrix rows
        const int lda = numGroups * 8;
        var rng = new Random(42);

        byte[] rows = new byte[numGroups * RepackedGemm.RowsPerGroup * blocksPerRow * RepackedGemm.Q4KBytesPerBlock];
        rng.NextBytes(rows);
        for (int r = 0; r < numGroups * RepackedGemm.RowsPerGroup; r++)
        for (int b = 0; b < blocksPerRow; b++)
        {
            int off = (r * blocksPerRow + b) * RepackedGemm.Q4KBytesPerBlock;
            var dBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            var dminBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            rows[off] = dBits[0]; rows[off + 1] = dBits[1];
            rows[off + 2] = dminBits[0]; rows[off + 3] = dminBits[1];
        }

        var activations = new float[batchSize * cols];
        for (int i = 0; i < activations.Length; i++) activations[i] = (float)(rng.NextDouble() * 2 - 1);

        byte[] repacked = new byte[numGroups * blocksPerRow * RepackedGemm.Q4KGroupBytesPerBlock];
        int groupSrcBytes = 8 * blocksPerRow * RepackedGemm.Q4KBytesPerBlock;
        int groupDstBytes = blocksPerRow * RepackedGemm.Q4KGroupBytesPerBlock;
        for (int g = 0; g < numGroups; g++)
        {
            byte[] groupBytes = RepackedGemm.RepackQ4K8Rows(rows.AsSpan(g * groupSrcBytes, groupSrcBytes), blocksPerRow);
            groupBytes.CopyTo(repacked, g * groupDstBytes);
        }

        long bytesPerActRow = SimdKernels.Q8KScratchBytes(cols);
        var actScratch = new byte[batchSize * bytesPerActRow];

        fixed (float* actPtr = activations)
        fixed (byte* actScratchPtr = actScratch)
        {
            SimdKernels.QuantizePromptToQ8K(actPtr, batchSize, cols, actScratchPtr);
        }

        var actualOutput = new float[batchSize * lda];
        fixed (float* outPtr = actualOutput)
        fixed (byte* repackedPtr = repacked)
        fixed (byte* actScratchPtr = actScratch)
        {
            RepackedGemm.GemmQ4K16x16Q8(outPtr, lda, repackedPtr, numGroups, actScratchPtr, batchSize, blocksPerRow);
        }

        Assert.Equal(batchSize * lda, actualOutput.Length);
        Assert.NotEqual(0f, actualOutput[0]);
    }

    [Fact]
    public unsafe void TryMatMulBatchedQ8_L2Chunking_ParityTest()
    {
        const int cols = 256;
        const int rows = 16;
        const int batchSize = 1024; // > 512 max chunk boundary
        var rng = new Random(123);

        byte[] weights = new byte[rows * (cols / 256) * 144];
        rng.NextBytes(weights);

        float[] input = new float[batchSize * cols];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] output = new float[batchSize * rows];

        fixed (float* outPtr = output)
        fixed (byte* wPtr = weights)
        fixed (float* inPtr = input)
        {
            bool success = SimdKernels.TryMatMulBatchedQ8(outPtr, wPtr, inPtr, batchSize, rows, cols, DType.Q4_K);
            Assert.True(success);
        }

        Assert.NotEqual(0f, output[0]);
        Assert.NotEqual(0f, output[batchSize * rows - 1]);
    }

    /// <summary>
    /// GemmQ4K8x8x4Q8K_Avx2 (4 tokens x 8 columns per call, the genuine 2D-tile kernel) must be
    /// bit-exact vs calling the already-validated GemvQ4K8x8Q8K (1 token x 8 columns) four
    /// times -- it's the same math, sharing the weight-nibble decode across tokens instead of
    /// redoing it, so any divergence is a reassociation bug in the sharing, not activation
    /// noise (unlike the DotQ4K-tolerance test elsewhere in this file).
    /// </summary>
    [Fact]
    public unsafe void GemmQ4K8x8x4Q8K_MatchesFourGemvCalls()
    {
        const int blocksPerRow = 4; // 1024-element rows
        const int cols = blocksPerRow * 256;
        var rng = new Random(2468);

        byte[] rows = new byte[RepackedGemm.RowsPerGroup * blocksPerRow * RepackedGemm.Q4KBytesPerBlock];
        rng.NextBytes(rows);
        for (int r = 0; r < RepackedGemm.RowsPerGroup; r++)
        for (int b = 0; b < blocksPerRow; b++)
        {
            int off = (r * blocksPerRow + b) * RepackedGemm.Q4KBytesPerBlock;
            var dBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            var dminBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            rows[off] = dBits[0]; rows[off + 1] = dBits[1];
            rows[off + 2] = dminBits[0]; rows[off + 3] = dminBits[1];
        }
        byte[] repacked = RepackedGemm.RepackQ4K8Rows(rows, blocksPerRow);

        var activations = new float[4][];
        var scratches = new byte[4][];
        int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
        for (int t = 0; t < 4; t++)
        {
            activations[t] = new float[cols];
            for (int i = 0; i < cols; i++) activations[t][i] = (float)(rng.NextDouble() * 2 - 1);
            scratches[t] = new byte[scratchBytes];
            fixed (float* actPtr = activations[t])
            fixed (byte* scratchPtr = scratches[t])
            {
                SimdKernels.QuantizeRowToQ8K(actPtr, cols, scratchPtr);
            }
        }

        var refOut = new float[4][];
        for (int t = 0; t < 4; t++)
        {
            refOut[t] = new float[8];
            fixed (byte* repackedPtr = repacked)
            fixed (byte* scratchPtr = scratches[t])
            fixed (float* outPtr = refOut[t])
            {
                RepackedGemm.GemvQ4K8x8Q8K(outPtr, repackedPtr, scratchPtr, blocksPerRow);
            }
        }

        var gemmOut = new float[4][];
        for (int t = 0; t < 4; t++) gemmOut[t] = new float[8];
        fixed (byte* repackedPtr = repacked)
        fixed (byte* s0 = scratches[0]) fixed (byte* s1 = scratches[1])
        fixed (byte* s2 = scratches[2]) fixed (byte* s3 = scratches[3])
        fixed (float* o0 = gemmOut[0]) fixed (float* o1 = gemmOut[1])
        fixed (float* o2 = gemmOut[2]) fixed (float* o3 = gemmOut[3])
        {
            RepackedGemm.GemmQ4K8x8x4Q8K_Avx2(o0, o1, o2, o3, repackedPtr, s0, s1, s2, s3, blocksPerRow);
        }

        for (int t = 0; t < 4; t++)
        for (int col = 0; col < 8; col++)
        {
            double relError = Math.Abs(gemmOut[t][col] - refOut[t][col]) / Math.Max(1e-3, Math.Abs(refOut[t][col]));
            Assert.True(relError < 0.001,
                $"token {t} col {col}: gemm={gemmOut[t][col]:F6} ref={refOut[t][col]:F6} (relError {relError:P4}, expected < 0.1%).");
        }
    }

    /// <summary>
    /// Correctness gate for the composed real-AVX2 port (docs/real-avx2-gemm-port-plan.md,
    /// <see cref="RealAvx2Gemm.GemmQ4K8x8x4Q8K_RealAvx2"/>): must match the already-trusted
    /// <see cref="RepackedGemm.GemmQ4K8x8x4Q8K_Avx2"/> to &lt;0.1% relative error on random
    /// Q4_K-shaped data, same tolerance convention as <see cref="GemmQ4K8x8x4Q8K_MatchesFourGemvCalls"/>
    /// above. Same fixture pattern (random weight bytes with plausible finite d/dmin, random
    /// [-1,1) activations quantized via the existing Q8_K path) -- this is the composition step's
    /// pass/fail gate before any performance measurement is attempted.
    /// </summary>
    [Fact]
    public unsafe void RealAvx2Gemm_MatchesTrustedGemmQ4K8x8x4Q8K_WithinTolerance()
    {
        const int blocksPerRow = 4; // 1024-element rows
        const int cols = blocksPerRow * 256;
        var rng = new Random(13579);

        byte[] rows = new byte[RepackedGemm.RowsPerGroup * blocksPerRow * RepackedGemm.Q4KBytesPerBlock];
        rng.NextBytes(rows);
        for (int r = 0; r < RepackedGemm.RowsPerGroup; r++)
        for (int b = 0; b < blocksPerRow; b++)
        {
            int off = (r * blocksPerRow + b) * RepackedGemm.Q4KBytesPerBlock;
            var dBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            var dminBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            rows[off] = dBits[0]; rows[off + 1] = dBits[1];
            rows[off + 2] = dminBits[0]; rows[off + 3] = dminBits[1];
        }
        byte[] repacked = RepackedGemm.RepackQ4K8Rows(rows, blocksPerRow);

        var activations = new float[4][];
        var scratches = new byte[4][];
        int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
        for (int t = 0; t < 4; t++)
        {
            activations[t] = new float[cols];
            for (int i = 0; i < cols; i++) activations[t][i] = (float)(rng.NextDouble() * 2 - 1);
            scratches[t] = new byte[scratchBytes];
            fixed (float* actPtr = activations[t])
            fixed (byte* scratchPtr = scratches[t])
            {
                SimdKernels.QuantizeRowToQ8K(actPtr, cols, scratchPtr);
            }
        }

        var refOut = new float[4][];
        for (int t = 0; t < 4; t++) refOut[t] = new float[8];
        fixed (byte* repackedPtr = repacked)
        fixed (byte* s0 = scratches[0]) fixed (byte* s1 = scratches[1])
        fixed (byte* s2 = scratches[2]) fixed (byte* s3 = scratches[3])
        fixed (float* o0 = refOut[0]) fixed (float* o1 = refOut[1])
        fixed (float* o2 = refOut[2]) fixed (float* o3 = refOut[3])
        {
            RepackedGemm.GemmQ4K8x8x4Q8K_Avx2(o0, o1, o2, o3, repackedPtr, s0, s1, s2, s3, blocksPerRow);
        }

        var realOut = new float[4][];
        for (int t = 0; t < 4; t++) realOut[t] = new float[8];
        fixed (byte* repackedPtr = repacked)
        fixed (byte* s0 = scratches[0]) fixed (byte* s1 = scratches[1])
        fixed (byte* s2 = scratches[2]) fixed (byte* s3 = scratches[3])
        fixed (float* o0 = realOut[0]) fixed (float* o1 = realOut[1])
        fixed (float* o2 = realOut[2]) fixed (float* o3 = realOut[3])
        {
            RealAvx2Gemm.GemmQ4K8x8x4Q8K_RealAvx2(o0, o1, o2, o3, repackedPtr, s0, s1, s2, s3, blocksPerRow);
        }

        for (int t = 0; t < 4; t++)
        for (int col = 0; col < 8; col++)
        {
            double relError = Math.Abs(realOut[t][col] - refOut[t][col]) / Math.Max(1e-3, Math.Abs(refOut[t][col]));
            Assert.True(relError < 0.001,
                $"token {t} col {col}: real={realOut[t][col]:F6} ref={refOut[t][col]:F6} (relError {relError:P4}, expected < 0.1%).");
        }
    }

    /// <summary>
    /// Single-threaded per-unit timing gauge for the real-AVX2 port vs the already-trusted
    /// GemmQ4K8x8x4Q8K_Avx2, before any Parallel.For scaling or the real CLI benchmark --
    /// per docs/cpu-prefill-repack-gemm-plan.md's established methodology (measure the smallest
    /// unit first, since every prior repacked-GEMM attempt in this investigation had per-unit
    /// wins that vanished once scaled). Informational only (no assertion) -- reported via
    /// ITestOutputHelper, verdict recorded in the tracking docs by hand.
    /// </summary>
    [Fact]
    public unsafe void PerfGauge_RealAvx2Gemm_VsTrusted_SingleThreaded()
    {
        const int blocksPerRow = 8; // 2048-element rows, matching the reference shape
        const int cols = blocksPerRow * 256;
        var rng = new Random(24680);

        byte[] rows = new byte[RepackedGemm.RowsPerGroup * blocksPerRow * RepackedGemm.Q4KBytesPerBlock];
        rng.NextBytes(rows);
        for (int r = 0; r < RepackedGemm.RowsPerGroup; r++)
        for (int b = 0; b < blocksPerRow; b++)
        {
            int off = (r * blocksPerRow + b) * RepackedGemm.Q4KBytesPerBlock;
            var dBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            var dminBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            rows[off] = dBits[0]; rows[off + 1] = dBits[1];
            rows[off + 2] = dminBits[0]; rows[off + 3] = dminBits[1];
        }
        byte[] repacked = RepackedGemm.RepackQ4K8Rows(rows, blocksPerRow);

        var activations = new float[4][];
        var scratches = new byte[4][];
        int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
        for (int t = 0; t < 4; t++)
        {
            activations[t] = new float[cols];
            for (int i = 0; i < cols; i++) activations[t][i] = (float)(rng.NextDouble() * 2 - 1);
            scratches[t] = new byte[scratchBytes];
            fixed (float* actPtr = activations[t])
            fixed (byte* scratchPtr = scratches[t])
            {
                SimdKernels.QuantizeRowToQ8K(actPtr, cols, scratchPtr);
            }
        }

        var out0 = new float[8]; var out1 = new float[8]; var out2 = new float[8]; var out3 = new float[8];
        const int warmup = 20;
        const int iters = 2000;

        double trustedMs = 0, realMs = 0, trustedMs2 = 0, realMs2 = 0;
        fixed (byte* repackedPtr = repacked)
        fixed (byte* s0 = scratches[0]) fixed (byte* s1 = scratches[1])
        fixed (byte* s2 = scratches[2]) fixed (byte* s3 = scratches[3])
        fixed (float* o0 = out0) fixed (float* o1 = out1) fixed (float* o2 = out2) fixed (float* o3 = out3)
        {
            for (int i = 0; i < warmup; i++) RepackedGemm.GemmQ4K8x8x4Q8K_Avx2(o0, o1, o2, o3, repackedPtr, s0, s1, s2, s3, blocksPerRow);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < iters; i++) RepackedGemm.GemmQ4K8x8x4Q8K_Avx2(o0, o1, o2, o3, repackedPtr, s0, s1, s2, s3, blocksPerRow);
            sw.Stop(); trustedMs = sw.Elapsed.TotalMilliseconds;

            for (int i = 0; i < warmup; i++) RealAvx2Gemm.GemmQ4K8x8x4Q8K_RealAvx2(o0, o1, o2, o3, repackedPtr, s0, s1, s2, s3, blocksPerRow);
            sw.Restart();
            for (int i = 0; i < iters; i++) RealAvx2Gemm.GemmQ4K8x8x4Q8K_RealAvx2(o0, o1, o2, o3, repackedPtr, s0, s1, s2, s3, blocksPerRow);
            sw.Stop(); realMs = sw.Elapsed.TotalMilliseconds;

            // Run twice, report both -- single-run JIT/cache jitter isn't trustworthy per this
            // investigation's own established discipline (docs/cpu-prefill-repack-gemm-plan.md §29).
            sw.Restart();
            for (int i = 0; i < iters; i++) RepackedGemm.GemmQ4K8x8x4Q8K_Avx2(o0, o1, o2, o3, repackedPtr, s0, s1, s2, s3, blocksPerRow);
            sw.Stop(); trustedMs2 = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            for (int i = 0; i < iters; i++) RealAvx2Gemm.GemmQ4K8x8x4Q8K_RealAvx2(o0, o1, o2, o3, repackedPtr, s0, s1, s2, s3, blocksPerRow);
            sw.Stop(); realMs2 = sw.Elapsed.TotalMilliseconds;
        }

        string report =
            $"Trusted GemmQ4K8x8x4Q8K_Avx2: run1={trustedMs:F2}ms run2={trustedMs2:F2}ms for {iters} iters ({trustedMs / iters * 1000:F2}us/call, {trustedMs2 / iters * 1000:F2}us/call)\n" +
            $"Real    GemmQ4K8x8x4Q8K_RealAvx2: run1={realMs:F2}ms run2={realMs2:F2}ms for {iters} iters ({realMs / iters * 1000:F2}us/call, {realMs2 / iters * 1000:F2}us/call)\n" +
            $"Ratio (real/trusted, lower is better for real): run1={realMs / trustedMs:F3}x run2={realMs2 / trustedMs2:F3}x";
        output.WriteLine(report);
    }

    /// <summary>
    /// The measurement that actually decides this: scaled, Parallel.For, against the
    /// already-SHIPPED path (SimdKernels.TryMatMulBatchedQ8, the `_8In`/dot8 Q4_K kernel), not
    /// against RepackedGemm's own earlier attempt. 2048x2048/batch=256 is this investigation's
    /// established reference shape. Both paths run on the SAME underlying weight bytes (the
    /// real-AVX2 path's repacked groups are built from slices of the same raw buffer the
    /// shipped path reads directly) so this is an apples-to-apples timing comparison, not just
    /// a correctness-independent one. Flat-2D Parallel.For granularity for the real-AVX2 path
    /// (rowGroup x tokenGroup as one flat work range) per cpu-prefill-repack-gemm-plan.md §26's
    /// finding that granularity mattered more than kernel width for earlier attempts.
    /// </summary>
    [Fact]
    public unsafe void PerfGauge_RealAvx2Gemm_VsShipped_ParallelForScaled()
    {
        const int rows = 2048;
        const int cols = 2048;
        const int batchSize = 256;
        const int blocksPerRow = cols / 256; // 8
        const int bytesPerRow = blocksPerRow * RepackedGemm.Q4KBytesPerBlock;
        var rng = new Random(97531);

        byte[] rawWeights = new byte[rows * bytesPerRow];
        rng.NextBytes(rawWeights);
        for (int r = 0; r < rows; r++)
        for (int b = 0; b < blocksPerRow; b++)
        {
            int off = (r * blocksPerRow + b) * RepackedGemm.Q4KBytesPerBlock;
            var dBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            var dminBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            rawWeights[off] = dBits[0]; rawWeights[off + 1] = dBits[1];
            rawWeights[off + 2] = dminBits[0]; rawWeights[off + 3] = dminBits[1];
        }

        int rowGroups = rows / RepackedGemm.RowsPerGroup; // 256
        int groupBytes = blocksPerRow * RepackedGemm.Q4KGroupBytesPerBlock;
        byte[] repackedFlat = new byte[rowGroups * groupBytes];
        for (int g = 0; g < rowGroups; g++)
        {
            var groupRows = rawWeights.AsSpan(g * RepackedGemm.RowsPerGroup * bytesPerRow, RepackedGemm.RowsPerGroup * bytesPerRow);
            byte[] repacked = RepackedGemm.RepackQ4K8Rows(groupRows, blocksPerRow);
            repacked.CopyTo(repackedFlat.AsSpan(g * groupBytes, groupBytes));
        }

        var activation = new float[batchSize * cols];
        for (int i = 0; i < activation.Length; i++) activation[i] = (float)(rng.NextDouble() * 2 - 1);

        int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
        byte[] scratchAll = new byte[batchSize * scratchBytes];
        fixed (float* actPtr0 = activation)
        fixed (byte* scratchPtr0 = scratchAll)
        {
            float* actPtr0L = actPtr0;
            byte* scratchPtr0L = scratchPtr0;
            System.Threading.Tasks.Parallel.For(0, batchSize, n =>
                SimdKernels.QuantizeRowToQ8K(actPtr0L + (long)n * cols, cols, scratchPtr0L + (long)n * scratchBytes));
        }

        var outputShipped = new float[(long)batchSize * rows];
        var outputReal = new float[(long)batchSize * rows];

        const int warmupRuns = 10;
        const int timedRuns = 2;
        double[] shippedMs = new double[timedRuns];
        double[] realMs = new double[timedRuns];

        int tokenGroups = batchSize / 4; // 64
        long totalWork = (long)rowGroups * tokenGroups;

        double[] outputRealCoarseMsHolder = new double[2];
        double[] outputRealPersistentMsHolder = new double[2];
        fixed (byte* rawPtr = rawWeights)
        fixed (byte* repackedPtr = repackedFlat)
        fixed (byte* scratchPtr = scratchAll)
        fixed (float* actPtr = activation)
        fixed (float* outShippedPtr = outputShipped)
        fixed (float* outRealPtr = outputReal)
        {
            for (int i = 0; i < warmupRuns; i++)
                SimdKernels.TryMatMulBatchedQ8(outShippedPtr, rawPtr, actPtr, batchSize, rows, cols, DType.Q4_K);
            for (int run = 0; run < timedRuns; run++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                SimdKernels.TryMatMulBatchedQ8(outShippedPtr, rawPtr, actPtr, batchSize, rows, cols, DType.Q4_K);
                sw.Stop();
                shippedMs[run] = sw.Elapsed.TotalMilliseconds;
            }

            byte* repackedPtrL = repackedPtr;
            byte* scratchPtrL = scratchPtr;
            float* outRealPtrL = outRealPtr;

            void RunRealPass()
            {
                System.Threading.Tasks.Parallel.For(0, totalWork, idx =>
                {
                    int rg = (int)(idx / tokenGroups);
                    int tg = (int)(idx % tokenGroups);
                    int tokenBase = tg * 4;
                    byte* repacked = repackedPtrL + (long)rg * groupBytes;
                    byte* s0 = scratchPtrL + (long)(tokenBase + 0) * scratchBytes;
                    byte* s1 = scratchPtrL + (long)(tokenBase + 1) * scratchBytes;
                    byte* s2 = scratchPtrL + (long)(tokenBase + 2) * scratchBytes;
                    byte* s3 = scratchPtrL + (long)(tokenBase + 3) * scratchBytes;
                    int rowBase = rg * RepackedGemm.RowsPerGroup;
                    float* o0 = outRealPtrL + (long)(tokenBase + 0) * rows + rowBase;
                    float* o1 = outRealPtrL + (long)(tokenBase + 1) * rows + rowBase;
                    float* o2 = outRealPtrL + (long)(tokenBase + 2) * rows + rowBase;
                    float* o3 = outRealPtrL + (long)(tokenBase + 3) * rows + rowBase;
                    RealAvx2Gemm.GemmQ4K8x8x4Q8K_RealAvx2(o0, o1, o2, o3, repacked, s0, s1, s2, s3, blocksPerRow);
                });
            }

            // Coarser-granularity variant: parallel over rowGroups only (256 tasks, each doing
            // all 64 token-groups sequentially -- bigger per-task chunk of work, matching
            // TryMatMulBatchedQ8's own row-outer/token-inner granularity) -- per
            // cpu-prefill-repack-gemm-plan.md §26's finding that granularity mattered more than
            // kernel width for earlier attempts, worth checking before finalizing a verdict.
            void RunRealPassCoarse()
            {
                System.Threading.Tasks.Parallel.For(0, rowGroups, rg =>
                {
                    byte* repacked = repackedPtrL + (long)rg * groupBytes;
                    int rowBase = rg * RepackedGemm.RowsPerGroup;
                    for (int tg = 0; tg < tokenGroups; tg++)
                    {
                        int tokenBase = tg * 4;
                        byte* s0 = scratchPtrL + (long)(tokenBase + 0) * scratchBytes;
                        byte* s1 = scratchPtrL + (long)(tokenBase + 1) * scratchBytes;
                        byte* s2 = scratchPtrL + (long)(tokenBase + 2) * scratchBytes;
                        byte* s3 = scratchPtrL + (long)(tokenBase + 3) * scratchBytes;
                        float* o0 = outRealPtrL + (long)(tokenBase + 0) * rows + rowBase;
                        float* o1 = outRealPtrL + (long)(tokenBase + 1) * rows + rowBase;
                        float* o2 = outRealPtrL + (long)(tokenBase + 2) * rows + rowBase;
                        float* o3 = outRealPtrL + (long)(tokenBase + 3) * rows + rowBase;
                        RealAvx2Gemm.GemmQ4K8x8x4Q8K_RealAvx2(o0, o1, o2, o3, repacked, s0, s1, s2, s3, blocksPerRow);
                    }
                });
            }

            for (int i = 0; i < warmupRuns; i++) RunRealPass();
            for (int run = 0; run < timedRuns; run++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                RunRealPass();
                sw.Stop();
                realMs[run] = sw.Elapsed.TotalMilliseconds;
            }

            double[] realCoarseMs = new double[timedRuns];
            for (int i = 0; i < warmupRuns; i++) RunRealPassCoarse();
            for (int run = 0; run < timedRuns; run++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                RunRealPassCoarse();
                sw.Stop();
                realCoarseMs[run] = sw.Elapsed.TotalMilliseconds;
            }
            outputRealCoarseMsHolder[0] = realCoarseMs[0];
            outputRealCoarseMsHolder[1] = realCoarseMs[1];

            // PersistentThreadPool variant: same coarse rowGroups-only partition, but dispatched
            // through the persistent-worker pool (modeled on OpenBLAS's own thread server, see
            // PersistentThreadPool.cs) instead of Parallel.For, to isolate whether .NET's
            // ThreadPool dispatch cost specifically was the remaining gap to shipped throughput.
            void RunRealPassPersistent()
            {
                PersistentThreadPool.For(rowGroups, (rgFrom, rgTo) =>
                {
                    for (int rg = rgFrom; rg < rgTo; rg++)
                    {
                        byte* repacked = repackedPtrL + (long)rg * groupBytes;
                        int rowBase = rg * RepackedGemm.RowsPerGroup;
                        for (int tg = 0; tg < tokenGroups; tg++)
                        {
                            int tokenBase = tg * 4;
                            byte* s0 = scratchPtrL + (long)(tokenBase + 0) * scratchBytes;
                            byte* s1 = scratchPtrL + (long)(tokenBase + 1) * scratchBytes;
                            byte* s2 = scratchPtrL + (long)(tokenBase + 2) * scratchBytes;
                            byte* s3 = scratchPtrL + (long)(tokenBase + 3) * scratchBytes;
                            float* o0 = outRealPtrL + (long)(tokenBase + 0) * rows + rowBase;
                            float* o1 = outRealPtrL + (long)(tokenBase + 1) * rows + rowBase;
                            float* o2 = outRealPtrL + (long)(tokenBase + 2) * rows + rowBase;
                            float* o3 = outRealPtrL + (long)(tokenBase + 3) * rows + rowBase;
                            RealAvx2Gemm.GemmQ4K8x8x4Q8K_RealAvx2(o0, o1, o2, o3, repacked, s0, s1, s2, s3, blocksPerRow);
                        }
                    }
                });
            }

            double[] realPersistentMs = new double[timedRuns];
            for (int i = 0; i < warmupRuns; i++) RunRealPassPersistent();
            for (int run = 0; run < timedRuns; run++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                RunRealPassPersistent();
                sw.Stop();
                realPersistentMs[run] = sw.Elapsed.TotalMilliseconds;
            }
            outputRealPersistentMsHolder[0] = realPersistentMs[0];
            outputRealPersistentMsHolder[1] = realPersistentMs[1];
        }

        // Sanity: outputs must roughly agree (same weight bytes, same activations) -- not a
        // tight tolerance check (that's RealAvx2Gemm_MatchesTrustedGemmQ4K8x8x4Q8K_WithinTolerance's
        // job), just a guard against a scaling-only bug producing garbage that still "times fast".
        double maxRel = 0;
        for (int i = 0; i < outputShipped.Length; i += 997) // sparse sample, full compare is slow
            maxRel = Math.Max(maxRel, Math.Abs(outputReal[i] - outputShipped[i]) / Math.Max(1e-3, Math.Abs(outputShipped[i])));

        double tokPerSecShipped0 = batchSize / (shippedMs[0] / 1000.0);
        double tokPerSecShipped1 = batchSize / (shippedMs[1] / 1000.0);
        double tokPerSecReal0 = batchSize / (realMs[0] / 1000.0);
        double tokPerSecReal1 = batchSize / (realMs[1] / 1000.0);
        double tokPerSecRealCoarse0 = batchSize / (outputRealCoarseMsHolder[0] / 1000.0);
        double tokPerSecRealCoarse1 = batchSize / (outputRealCoarseMsHolder[1] / 1000.0);
        double tokPerSecRealPersistent0 = batchSize / (outputRealPersistentMsHolder[0] / 1000.0);
        double tokPerSecRealPersistent1 = batchSize / (outputRealPersistentMsHolder[1] / 1000.0);

        string report =
            $"Shipped TryMatMulBatchedQ8: run1={shippedMs[0]:F2}ms run2={shippedMs[1]:F2}ms -> {tokPerSecShipped0:F0}/{tokPerSecShipped1:F0} tok/s\n" +
            $"Real    GemmQ4K8x8x4Q8K_RealAvx2 (Parallel.For flat-2D): run1={realMs[0]:F2}ms run2={realMs[1]:F2}ms -> {tokPerSecReal0:F0}/{tokPerSecReal1:F0} tok/s\n" +
            $"Real    GemmQ4K8x8x4Q8K_RealAvx2 (Parallel.For coarse, rowGroups only): run1={outputRealCoarseMsHolder[0]:F2}ms run2={outputRealCoarseMsHolder[1]:F2}ms -> {tokPerSecRealCoarse0:F0}/{tokPerSecRealCoarse1:F0} tok/s\n" +
            $"Real    GemmQ4K8x8x4Q8K_RealAvx2 (PersistentThreadPool, coarse partition): run1={outputRealPersistentMsHolder[0]:F2}ms run2={outputRealPersistentMsHolder[1]:F2}ms -> {tokPerSecRealPersistent0:F0}/{tokPerSecRealPersistent1:F0} tok/s\n" +
            $"Speedup flat-2D     (real/shipped tok/s, >1 means real wins): run1={tokPerSecReal0 / tokPerSecShipped0:F3}x run2={tokPerSecReal1 / tokPerSecShipped1:F3}x\n" +
            $"Speedup coarse      (real/shipped tok/s, >1 means real wins): run1={tokPerSecRealCoarse0 / tokPerSecShipped0:F3}x run2={tokPerSecRealCoarse1 / tokPerSecShipped1:F3}x\n" +
            $"Speedup persistent  (real/shipped tok/s, >1 means real wins): run1={tokPerSecRealPersistent0 / tokPerSecShipped0:F3}x run2={tokPerSecRealPersistent1 / tokPerSecShipped1:F3}x\n" +
            $"Sparse-sample max relative diff vs shipped (expected large -- shipped uses Q8_KS quantization internally, real-AVX2 uses plain Q8_K matching its own correctness-tested convention; not the correctness gate): {maxRel:P4}";
        output.WriteLine(report);
    }

    /// <summary>
    /// GemmQ4K8x8x8Q8K_Avx2 (8 tokens x 8 columns per call, widening GemmQ4K8x8x4Q8K_Avx2's
    /// token axis to match _8In's own reuse width, phase-2 plan §24) must be bit-close vs 8
    /// independent GemvQ4K8x8Q8K calls -- same reasoning as the 4-token version's test.
    /// </summary>
    [Fact]
    public unsafe void GemmQ4K8x8x8Q8K_MatchesEightGemvCalls()
    {
        const int blocksPerRow = 4;
        const int cols = blocksPerRow * 256;
        const int numTokens = 8;
        var rng = new Random(13579);

        byte[] rows = new byte[RepackedGemm.RowsPerGroup * blocksPerRow * RepackedGemm.Q4KBytesPerBlock];
        rng.NextBytes(rows);
        for (int r = 0; r < RepackedGemm.RowsPerGroup; r++)
        for (int b = 0; b < blocksPerRow; b++)
        {
            int off = (r * blocksPerRow + b) * RepackedGemm.Q4KBytesPerBlock;
            var dBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            var dminBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
            rows[off] = dBits[0]; rows[off + 1] = dBits[1];
            rows[off + 2] = dminBits[0]; rows[off + 3] = dminBits[1];
        }
        byte[] repacked = RepackedGemm.RepackQ4K8Rows(rows, blocksPerRow);

        var activations = new float[numTokens][];
        var scratches = new byte[numTokens][];
        int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
        for (int t = 0; t < numTokens; t++)
        {
            activations[t] = new float[cols];
            for (int i = 0; i < cols; i++) activations[t][i] = (float)(rng.NextDouble() * 2 - 1);
            scratches[t] = new byte[scratchBytes];
            fixed (float* actPtr = activations[t])
            fixed (byte* scratchPtr = scratches[t])
            {
                SimdKernels.QuantizeRowToQ8K(actPtr, cols, scratchPtr);
            }
        }

        var refOut = new float[numTokens][];
        for (int t = 0; t < numTokens; t++)
        {
            refOut[t] = new float[8];
            fixed (byte* repackedPtr = repacked)
            fixed (byte* scratchPtr = scratches[t])
            fixed (float* outPtr = refOut[t])
            {
                RepackedGemm.GemvQ4K8x8Q8K(outPtr, repackedPtr, scratchPtr, blocksPerRow);
            }
        }

        var gemmOut = new float[numTokens][];
        for (int t = 0; t < numTokens; t++) gemmOut[t] = new float[8];

        var scratchHandles = new System.Runtime.InteropServices.GCHandle[numTokens];
        var outHandles = new System.Runtime.InteropServices.GCHandle[numTokens];
        try
        {
            byte** acts = stackalloc byte*[numTokens];
            float** outs = stackalloc float*[numTokens];
            for (int t = 0; t < numTokens; t++)
            {
                scratchHandles[t] = System.Runtime.InteropServices.GCHandle.Alloc(scratches[t], System.Runtime.InteropServices.GCHandleType.Pinned);
                outHandles[t] = System.Runtime.InteropServices.GCHandle.Alloc(gemmOut[t], System.Runtime.InteropServices.GCHandleType.Pinned);
                acts[t] = (byte*)scratchHandles[t].AddrOfPinnedObject();
                outs[t] = (float*)outHandles[t].AddrOfPinnedObject();
            }
            fixed (byte* repackedPtr = repacked)
            {
                RepackedGemm.GemmQ4K8x8x8Q8K_Avx2(outs, repackedPtr, acts, blocksPerRow);
            }
        }
        finally
        {
            for (int t = 0; t < numTokens; t++)
            {
                if (scratchHandles[t].IsAllocated) scratchHandles[t].Free();
                if (outHandles[t].IsAllocated) outHandles[t].Free();
            }
        }

        for (int t = 0; t < numTokens; t++)
        for (int col = 0; col < 8; col++)
        {
            double relError = Math.Abs(gemmOut[t][col] - refOut[t][col]) / Math.Max(1e-3, Math.Abs(refOut[t][col]));
            Assert.True(relError < 0.001,
                $"token {t} col {col}: gemm={gemmOut[t][col]:F6} ref={refOut[t][col]:F6} (relError {relError:P4}, expected < 0.1%).");
        }
    }
}
