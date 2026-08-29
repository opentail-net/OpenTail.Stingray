
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Correctness gate for the <c>block_q4_Kx8</c> 8-row weight repack (perf-loop iteration 39).
///
/// <para>The repack exists so a single AVX2 load can hold 8 bytes from each of 4 weight rows,
/// letting one <c>maddubs</c> cover 8 elements × 4 rows and four of them chain into a 32-element
/// sub-block — the instruction-efficiency advantage llama.cpp's kernel has over a row-major
/// layout. Every vectorised kernel built on this layout inherits its indexing, so the layout is
/// pinned here FIRST, against the existing row-major path, before any SIMD is written against it.</para>
///
/// <para>The reference used is <c>DotQ4Kx8_Q8KS_Scalar</c>, which walks the interleaved bytes
/// element by element by construction — if the interleave or the decoded scale/min tables are
/// wrong it diverges immediately and by a large margin, rather than in the last mantissa bits.</para>
/// </summary>
public sealed unsafe class Q4Kx8RepackTests
{
    private static void WriteHalf(byte[] buf, int off, float v)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((Half)v);
        buf[off] = (byte)(bits & 0xFF);
        buf[off + 1] = (byte)(bits >> 8);
    }

    /// <summary>Deterministic Q4_K rows with finite scales, so a mismatch is always a layout bug.</summary>
    private static byte[] BuildRows(int rows, int cols, int seed)
    {
        int blocks = cols / 256;
        var b = new byte[(long)rows * blocks * 144];
        var rng = new Random(seed);
        rng.NextBytes(b);
        for (int off = 0; off + 144 <= b.Length; off += 144)
        {
            WriteHalf(b, off, 0.015f);
            WriteHalf(b, off + 2, 0.004f);
        }
        return b;
    }

    private static float[] RandFloats(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }

    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    public void RepackedGroupMatchesRowMajorDot(int cols)
    {
        const int rows = 8;
        int blocks = cols / 256;
        int bytesPerRow = blocks * 144;

        var weights = BuildRows(rows, cols, seed: 4242);
        var input = RandFloats(cols, seed: 99);

        var packed = new byte[(long)blocks * SimdKernels.Q4Kx8BlockBytes];
        var scratch = new byte[SimdKernels.Q8KSScratchBytes(cols)];
        var viaRepack = new float[rows];
        var viaRowMajor = new float[rows];

        fixed (byte* w = weights)
        fixed (byte* p = packed)
        fixed (byte* sc = scratch)
        fixed (float* inp = input)
        fixed (float* outR = viaRepack)
        {
            SimdKernels.QuantizeRowToQ8KS(inp, cols, sc);
            SimdKernels.RepackQ4K8Rows(w, p, blocks, bytesPerRow);
            SimdKernels.DotQ4Kx8_Q8KS_Scalar(p, sc, blocks, outR);

            for (int r = 0; r < rows; r++)
                viaRowMajor[r] = SimdKernels.DotQ4K_Q8KS(w + (long)r * bytesPerRow, sc, cols);
        }

        for (int r = 0; r < rows; r++)
        {
            // Same arithmetic, different traversal order, so this is FP-noise close rather than
            // exact. A layout error shows up as an O(1) difference, not 1e-4.
            float expect = viaRowMajor[r];
            float got = viaRepack[r];
            float tol = 1e-3f * Math.Max(1f, Math.Abs(expect));
            Assert.True(Math.Abs(expect - got) < tol,
                $"cols={cols} row {r}: repacked={got:R} rowMajor={expect:R}");
        }
    }

    /// <summary>
    /// The interleave must be a pure permutation — every source nibble-byte appears exactly once.
    /// Checked directly rather than inferred from the dot, so a self-consistent but wrong mapping
    /// (e.g. one that drops a chunk and duplicates another) cannot pass.
    /// </summary>
    [Fact]
    public void InterleaveIsAPermutationOfTheSourceBytes()
    {
        const int cols = 512, rows = 8;
        int blocks = cols / 256;
        int bytesPerRow = blocks * 144;

        var weights = BuildRows(rows, cols, seed: 7);
        var packed = new byte[(long)blocks * SimdKernels.Q4Kx8BlockBytes];

        fixed (byte* w = weights)
        fixed (byte* p = packed)
            SimdKernels.RepackQ4K8Rows(w, p, blocks, bytesPerRow);

        for (int b = 0; b < blocks; b++)
        {
            var srcCounts = new Dictionary<byte, int>();
            var dstCounts = new Dictionary<byte, int>();
            for (int r = 0; r < rows; r++)
                for (int i = 0; i < 128; i++)
                {
                    byte v = weights[(long)r * bytesPerRow + (long)b * 144 + 16 + i];
                    srcCounts[v] = srcCounts.GetValueOrDefault(v) + 1;
                }
            for (int i = 0; i < 1024; i++)
            {
                byte v = packed[(long)b * SimdKernels.Q4Kx8BlockBytes + 192 + i];
                dstCounts[v] = dstCounts.GetValueOrDefault(v) + 1;
            }
            Assert.Equal(srcCounts.Count, dstCounts.Count);
            foreach (var kv in srcCounts)
                Assert.Equal(kv.Value, dstCounts[kv.Key]);
        }
    }

    /// <summary>
    /// The AVX2 kernel over the repacked layout must agree with the row-major path. This is the
    /// gate on the lane algebra: the interleave puts rows in vector lanes, <c>madd_epi16</c> folds
    /// pairs so row r lands in int32 lanes 2r/2r+1, and the per-row d/dmin are applied lane-wise.
    /// Any error in that mapping produces an O(1) divergence on some subset of rows, not FP noise.
    /// </summary>
    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    [InlineData(2048)]
    public void Avx2KernelMatchesRowMajorDot(int cols)
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported) return;

        const int rows = 8;
        int blocks = cols / 256;
        int bytesPerRow = blocks * 144;

        var weights = BuildRows(rows, cols, seed: 31337);
        var input = RandFloats(cols, seed: 4242);

        var packed = new byte[(long)blocks * SimdKernels.Q4Kx8BlockBytes];
        var scratch = new byte[SimdKernels.Q8KSScratchBytes(cols)];
        var viaAvx2 = new float[rows];
        var viaRowMajor = new float[rows];

        fixed (byte* w = weights)
        fixed (byte* p = packed)
        fixed (byte* sc = scratch)
        fixed (float* inp = input)
        fixed (float* outR = viaAvx2)
        {
            SimdKernels.QuantizeRowToQ8KS(inp, cols, sc);
            SimdKernels.RepackQ4K8Rows(w, p, blocks, bytesPerRow);
            SimdKernels.DotQ4Kx8_Q8KS_Avx2(p, sc, blocks, outR);

            for (int r = 0; r < rows; r++)
                viaRowMajor[r] = SimdKernels.DotQ4K_Q8KS(w + (long)r * bytesPerRow, sc, cols);
        }

        for (int r = 0; r < rows; r++)
        {
            float expect = viaRowMajor[r];
            float got = viaAvx2[r];
            float tol = 1e-3f * Math.Max(1f, Math.Abs(expect));
            Assert.True(Math.Abs(expect - got) < tol,
                $"cols={cols} row {r}: avx2={got:R} rowMajor={expect:R}");
        }
    }

    /// <summary>
    /// The 8-row x 8-token kernel must agree with the row-major path for every (token, row) pair.
    /// Each token gets a DIFFERENT input so a bug that broadcasts one token's activations across
    /// the batch — the most likely failure mode for this shape — cannot pass.
    /// </summary>
    [Theory]
    [InlineData(2048)]
    [InlineData(8192)]
    public void Avx2Kernel8InMatchesRowMajorDot(int cols)
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported) return;

        const int rows = 8, toks = 8;
        int blocks = cols / 256;
        int bytesPerRow = blocks * 144;
        int scratchBytes = SimdKernels.Q8KSScratchBytes(cols);

        var weights = BuildRows(rows, cols, seed: 555);
        var packed = new byte[(long)blocks * SimdKernels.Q4Kx8BlockBytes];
        var scratch = new byte[(long)scratchBytes * toks];
        var got = new float[toks * rows];
        var expect = new float[toks * rows];

        fixed (byte* w = weights)
        fixed (byte* p = packed)
        fixed (byte* scAll = scratch)
        fixed (float* outR = got)
        {
            for (int k = 0; k < toks; k++)
            {
                var input = RandFloats(cols, seed: 1000 + k);   // distinct per token
                fixed (float* inp = input)
                    SimdKernels.QuantizeRowToQ8KS(inp, cols, scAll + (long)k * scratchBytes);
            }
            SimdKernels.RepackQ4K8Rows(w, p, blocks, bytesPerRow);

            SimdKernels.DotQ4Kx8_Q8KS_8In(p,
                scAll, scAll + scratchBytes, scAll + 2L * scratchBytes, scAll + 3L * scratchBytes,
                scAll + 4L * scratchBytes, scAll + 5L * scratchBytes, scAll + 6L * scratchBytes,
                scAll + 7L * scratchBytes, blocks, outR);

            for (int k = 0; k < toks; k++)
                for (int r = 0; r < rows; r++)
                    expect[k * rows + r] = SimdKernels.DotQ4K_Q8KS(
                        w + (long)r * bytesPerRow, scAll + (long)k * scratchBytes, cols);
        }

        for (int k = 0; k < toks; k++)
            for (int r = 0; r < rows; r++)
            {
                float e = expect[k * rows + r], a = got[k * rows + r];
                float tol = 1e-3f * Math.Max(1f, Math.Abs(e));
                Assert.True(Math.Abs(e - a) < tol,
                    $"cols={cols} token {k} row {r}: kernel={a:R} rowMajor={e:R}");
            }
    }
}
