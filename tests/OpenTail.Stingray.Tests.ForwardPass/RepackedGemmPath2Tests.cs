using System.Runtime.InteropServices;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Parity tests for <b>Path 2</b> — the literal C# port of llama.cpp's AVX2
/// <c>ggml_gemm_q4_K_8x8_q8_K</c> (see docs/repack-gemm/port-log.md).
///
/// <para><b>Why these compare against a scalar reference rather than against Path 1.</b> The two
/// paths quantise activations differently — Path 1 uses Q8_KS (eight sub-scales per super-block),
/// Path 2 uses Q8_K (one). They are therefore <i>not</i> bit-identical by construction, and a test
/// asserting equality between them could never pass. Both are instead compared to an exact-float
/// dot computed from the raw Q4_K bytes, and judged on relative error.</para>
///
/// <para>These are pure-kernel tests: they build synthetic Q4_K weights and need no model file.</para>
/// </summary>
public sealed unsafe class RepackedGemmPath2Tests
{
    /// <summary>Standard K-quant 6-bit scale/min unpack (llama.cpp <c>get_scale_min_k4</c>).</summary>
    private static void GetScaleMinK4(int j, byte* q, out byte d, out byte m)
    {
        if (j < 4) { d = (byte)(q[j] & 63); m = (byte)(q[j + 4] & 63); }
        else
        {
            d = (byte)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
            m = (byte)((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
        }
    }

    private static float HalfToF(byte lo, byte hi)
        => (float)BitConverter.Int16BitsToHalf((short)(lo | (hi << 8)));

    /// <summary>Exact float dot of one Q4_K row (144 B per super-block) against float activations.</summary>
    private static double RefDot(byte* row, float* x, int cols)
    {
        int nb = cols / 256;
        double acc = 0;
        for (int b = 0; b < nb; b++)
        {
            byte* blk = row + b * 144;
            float d = HalfToF(blk[0], blk[1]);
            float dmin = HalfToF(blk[2], blk[3]);
            byte* sc = blk + 4;
            byte* qs = blk + 16;
            for (int sb = 0; sb < 8; sb++)
            {
                GetScaleMinK4(sb, sc, out byte s, out byte m);
                byte* qbase = qs + (sb / 2) * 32;
                bool high = (sb & 1) != 0;
                double sumqx = 0, sumx = 0;
                for (int i = 0; i < 32; i++)
                {
                    int q = high ? (qbase[i] >> 4) : (qbase[i] & 0xF);
                    float xv = x[b * 256 + sb * 32 + i];
                    sumqx += q * xv;
                    sumx += xv;
                }
                acc += d * s * sumqx - dmin * m * sumx;
            }
        }
        return acc;
    }

    /// <summary>
    /// Runs both paths at one shape and returns their worst relative error against the reference.
    /// </summary>
    private static (double p1Rel, double p2Rel, bool p2Handled) Measure(int batch, int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        int nb = cols / 256;
        long rowBytes = (long)nb * 144;

        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)(rowBytes * rows), 64);
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * cols * batch), 64);
        byte* packed = (byte*)NativeMemory.AlignedAlloc((nuint)SimdKernels.Q4Kx8PackedBytes(rows, cols), 64);
        float* o1 = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * rows * batch), 64);
        float* o2 = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * rows * batch), 64);
        var saved = GemmPathConfig.Current;
        try
        {
            for (long i = 0; i < rowBytes * rows; i++) w[i] = (byte)rng.Next(256);
            // Constrain d/dmin so the reference stays in a sane numeric range.
            for (int r = 0; r < rows; r++)
                for (int b = 0; b < nb; b++)
                {
                    byte* blk = w + r * rowBytes + b * 144;
                    short a = BitConverter.HalfToInt16Bits((Half)(rng.NextDouble() * 0.05 + 0.005));
                    short c = BitConverter.HalfToInt16Bits((Half)(rng.NextDouble() * 0.02 + 0.001));
                    blk[0] = (byte)(a & 0xFF); blk[1] = (byte)(a >> 8);
                    blk[2] = (byte)(c & 0xFF); blk[3] = (byte)(c >> 8);
                }
            for (int i = 0; i < cols * batch; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

            SimdKernels.RepackQ4KMatrix(w, packed, rows, cols);

            GemmPathConfig.Current = GemmPath.Path1;
            SimdKernels.TryMatMulBatchedQ4Kx8(o1, packed, x, batch, rows, cols);
            GemmPathConfig.Current = GemmPath.Path2;
            bool handled = SimdKernels.TryMatMulBatchedQ4Kx8(o2, packed, x, batch, rows, cols);

            double e1 = 0, e2 = 0, scale = 0;
            for (int t = 0; t < batch; t++)
                for (int r = 0; r < rows; r++)
                {
                    double refv = RefDot(w + r * rowBytes, x + (long)t * cols, cols);
                    scale = Math.Max(scale, Math.Abs(refv));
                    e1 = Math.Max(e1, Math.Abs(o1[(long)t * rows + r] - refv));
                    e2 = Math.Max(e2, Math.Abs(o2[(long)t * rows + r] - refv));
                }
            return (e1 / scale, e2 / scale, handled);
        }
        finally
        {
            GemmPathConfig.Current = saved;
            NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(x); NativeMemory.AlignedFree(packed);
            NativeMemory.AlignedFree(o1); NativeMemory.AlignedFree(o2);
        }
    }

    /// <summary>
    /// Path 2 must stay within activation-quantisation noise of an exact reference at every batch
    /// size. The ragged values matter most: 1/2/3 exercise the zero-padded partial group, and
    /// 5/7/13/17/65 exercise a full pass followed by a short one. A wrong lane layout shows up as a
    /// relative error near 1.0, not near 0.004, so this bound is deliberately loose but decisive.
    /// </summary>
    [Theory]
    [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(4)] [InlineData(5)]
    [InlineData(7)] [InlineData(8)] [InlineData(11)] [InlineData(13)] [InlineData(16)]
    [InlineData(17)] [InlineData(20)] [InlineData(64)] [InlineData(65)]
    public void Path2_MatchesScalarReference_AtEveryBatchSize(int batch)
    {
        if (!SimdKernels.CanRepackQ4Kx8(64, 512)) return;   // needs AVX2+FMA

        var (p1Rel, p2Rel, handled) = Measure(batch, rows: 64, cols: 512, seed: 20260802);

        Assert.True(handled, $"Path 2 declined batch={batch}; it must handle every batch size " +
                             "since increment 8b (zero-padded partial groups).");
        Assert.True(p2Rel < 0.02, $"Path 2 rel error {p2Rel:E3} at batch={batch} (Path 1: {p1Rel:E3})");
    }

    /// <summary>
    /// Path 2 must be the path a default process actually takes at a production trunk shape.
    ///
    /// <para>Every other test here sets <see cref="GemmPathConfig.Current"/> explicitly, so none of
    /// them would notice if the default flipped back to Path 1 or if Path 2 silently declined the
    /// real shape — the caller falls through to Path 1 and the only symptom is lost speed. This
    /// asserts the two halves separately: the resolved default, and that a call at that default
    /// increments Path 2's own engagement counter.</para>
    ///
    /// <para>Path 2's Q8_K activations (one scale per super-block) are what let the 6-bit weight
    /// scale fold into <c>madd_epi16</c> and the int32 accumulate run across a whole super-block,
    /// which Q8_KS cannot do — see <c>AccumQ4KInput</c>'s remarks. That is the difference this
    /// pins, so it must be observed rather than assumed.</para>
    /// </summary>
    [Fact]
    public void Path2_IsTheDefault_AndEngagesAtTrunkShape()
    {
        if (Environment.GetEnvironmentVariable("STINGRAY_GEMM_PATH") is { Length: > 0 })
            return;   // an explicit override is a legitimate configuration, not a failure

        Assert.Equal(GemmPath.Path2, GemmPathConfig.ReadFromEnvironment());

        if (!SimdKernels.CanRepackQ4Kx8(2048, 2048)) return;

        // NOT via Measure(): that sets GemmPathConfig.Current itself, which would defeat the whole
        // point of this test. The dispatcher is called with the default left untouched.
        const int rows = 2048, cols = 2048, batch = 11;
        var rng = new Random(7);
        long rowBytes = (long)(cols / 256) * 144;
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)(rowBytes * rows), 64);
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * cols * batch), 64);
        byte* packed = (byte*)NativeMemory.AlignedAlloc((nuint)SimdKernels.Q4Kx8PackedBytes(rows, cols), 64);
        float* o = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * rows * batch), 64);
        try
        {
            for (long i = 0; i < rowBytes * rows; i++) w[i] = (byte)rng.Next(256);
            for (int i = 0; i < cols * batch; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            SimdKernels.RepackQ4KMatrix(w, packed, rows, cols);

            int before = RepackedGemmPath2.EngagedCalls;
            Assert.True(SimdKernels.TryMatMulBatchedQ4Kx8(o, packed, x, batch, rows, cols));

            Assert.True(RepackedGemmPath2.EngagedCalls > before,
                "Path 2 did not engage at the trunk shape under the resolved default — the caller " +
                "silently fell back to Path 1, which costs the repacked GEMM's measured 1.83x.");
        }
        finally
        {
            NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(x);
            NativeMemory.AlignedFree(packed); NativeMemory.AlignedFree(o);
        }
    }

    /// <summary>
    /// Same check at the real trunk shape. Kept separate because it is far slower, and because a
    /// kernel can be correct at cols=512 (2 super-blocks) and wrong at cols=2048 (8) if a
    /// super-block loop index is mishandled.
    /// </summary>
    [Theory]
    [InlineData(4)] [InlineData(7)] [InlineData(11)]
    public void Path2_MatchesScalarReference_AtTrunkShape(int batch)
    {
        if (!SimdKernels.CanRepackQ4Kx8(2048, 2048)) return;

        var (_, p2Rel, handled) = Measure(batch, rows: 2048, cols: 2048, seed: 7);

        Assert.True(handled);
        Assert.True(p2Rel < 0.02, $"Path 2 rel error {p2Rel:E3} at trunk shape, batch={batch}");
    }

    /// <summary>
    /// The vectorised quantiser must agree with the scalar oracle byte for byte — same <c>d</c>,
    /// same quants, same bsums. Two deliberate deviations (max/min sign recovery instead of the
    /// original's compare-mask chain, and a direct bsums derivation instead of its shuffle/blend
    /// dance) are supposed to be output-identical, and this is what proves it rather than the
    /// reasoning in their doc comments.
    /// </summary>
    [Theory]
    [InlineData(256)] [InlineData(512)] [InlineData(2048)]
    public void QuantizeMatQ8Kx4Avx2_IsBitIdenticalToScalar(int cols)
    {
        if (!SimdKernels.CanRepackQ4Kx8(64, 256)) return;

        var rng = new Random(1234 + cols);
        long bytes = RepackedGemmPath2.Q8Kx4Bytes(cols);
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * 4 * cols), 64);
        byte* a = (byte*)NativeMemory.AllocZeroed((nuint)bytes);
        byte* b = (byte*)NativeMemory.AllocZeroed((nuint)bytes);
        try
        {
            // Mixed magnitudes and signs, including a deliberately large negative outlier so the
            // "max-magnitude element is negative" branch of the sign recovery is exercised.
            for (int i = 0; i < 4 * cols; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            x[5] = -9.5f;
            x[cols + 11] = 9.5f;

            RepackedGemmPath2.QuantizeMatQ8Kx4(x, a, cols);
            RepackedGemmPath2.QuantizeMatQ8Kx4Avx2(x, b, cols);

            for (long i = 0; i < bytes; i++)
                if (a[i] != b[i])
                    Assert.Fail($"cols={cols}: byte {i} differs — scalar {a[i]}, avx2 {b[i]}");
        }
        finally { NativeMemory.AlignedFree(x); NativeMemory.Free(a); NativeMemory.Free(b); }
    }

    /// <summary>
    /// The zero-padding contract: rows appended to fill a <c>block_q8_Kx4</c> quantise to
    /// <c>d = 0</c> (because <c>amax == 0</c>) and so contribute exactly nothing. If that ever
    /// stopped holding, padded batches would silently perturb real rows — this pins it directly
    /// rather than relying on the end-to-end parity tests to notice.
    /// </summary>
    [Fact]
    public void QuantizeMatQ8Kx4_ZeroRow_ProducesZeroScale()
    {
        const int cols = 512;
        float* x = (float*)NativeMemory.AllocZeroed((nuint)(sizeof(float) * 4 * cols));
        byte* y = (byte*)NativeMemory.AllocZeroed((nuint)RepackedGemmPath2.Q8Kx4Bytes(cols));
        try
        {
            // Rows 0-1 carry real data; rows 2-3 stay zero, as the padding path leaves them.
            for (int i = 0; i < 2 * cols; i++) x[i] = (float)Math.Sin(i * 0.01) * 3f;

            RepackedGemmPath2.QuantizeMatQ8Kx4(x, y, cols);

            int nb = cols / RepackedGemmPath2.QK_K;
            for (int b = 0; b < nb; b++)
            {
                float* d = (float*)(y + (long)b * RepackedGemmPath2.Q8Kx4BlockBytes);
                Assert.NotEqual(0f, d[0]);
                Assert.NotEqual(0f, d[1]);
                Assert.Equal(0f, d[2]);
                Assert.Equal(0f, d[3]);
            }
        }
        finally { NativeMemory.Free(x); NativeMemory.Free(y); }
    }
}
