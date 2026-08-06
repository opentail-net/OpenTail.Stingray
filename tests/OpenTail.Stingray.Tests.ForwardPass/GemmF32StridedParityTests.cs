using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Parity gate for <see cref="SimdKernels.GemmF32_6x2"/>, the strided generalization of the
/// shape-hardcoded <see cref="SimdKernels.GemmF32_64x64_6x2"/>
/// (docs/cpu-architecture-kernel-opportunities.md item 3).
///
/// <para>The hardcoded kernel bakes K, N and all three row strides in as the literal 64, which is
/// exactly why flash attention refuses to run at <c>headDim = 128</c>: <c>Q·Kᵀ</c> there needs
/// <c>k=128, n=64</c> and <c>P·V</c> needs <c>k=64, n=128</c>, and feeding either through a
/// kernel that assumes 64 would read past the operands and produce silently wrong scores rather
/// than fail. The strided kernel is what unblocks it.</para>
///
/// <para><b>The contract is bit equality at the overlapping shape.</b> The two kernels share the
/// k-loop and the FMA order exactly, so at <c>k = n = lda = ldb = ldc = 64</c> there is no
/// legitimate reason for a single bit to differ. A tolerance here would accept a transposed index
/// or an off-by-one stride whose error happens to land small on random data, which is the entire
/// failure mode being guarded against.</para>
///
/// <para>Non-square shapes have no second implementation to compare against, so those are checked
/// against a plain triple-loop reference at a tolerance — different summation order, so bit
/// equality is not available there, but the reference shares no code with the kernel.</para>
/// </summary>
public sealed unsafe class GemmF32StridedParityTests
{
    private static float[] Rand(int n, int seed)
    {
        var rng = new Random(seed);
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)((rng.NextDouble() - 0.5) * 2.0);
        return v;
    }

    /// <summary>Independent oracle: no SIMD, no blocking, plain definition of a GEMM.</summary>
    private static float[] Reference(float[] a, float[] b, int m, int k, int n,
        int lda, int ldb, int ldc, float[]? initial)
    {
        var c = initial is null ? new float[m * ldc] : (float[])initial.Clone();
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double acc = initial is null ? 0 : c[i * ldc + j];
                for (int kk = 0; kk < k; kk++)
                    acc += (double)a[i * lda + kk] * b[kk * ldb + j];
                c[i * ldc + j] = (float)acc;
            }
        return c;
    }

    /// <summary>
    /// The load-bearing test: at 64x64x64 the strided kernel must reproduce the hardcoded one bit
    /// for bit, for every row count including the ragged 1-5 row tail and the empty case.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(5)]   // pure ragged tail, no 6-row block
    [InlineData(6)]   // exactly one block
    [InlineData(7)]   // one block + tail
    [InlineData(64)]
    public void Strided_MatchesHardcoded_BitExact_At64(int m)
    {
        const int D = 64;
        var a = Rand(64 * D, 11 + m);
        var b = Rand(D * D, 22 + m);

        foreach (bool accumulate in new[] { false, true })
        {
            var cHard = Rand(64 * D, 33 + m);
            var cStrided = (float[])cHard.Clone();

            fixed (float* pa = a)
            fixed (float* pb = b)
            fixed (float* ph = cHard)
            fixed (float* ps = cStrided)
            {
                SimdKernels.GemmF32_64x64_6x2(pa, pb, ph, m, accumulate);
                SimdKernels.GemmF32_6x2(pa, pb, ps, m, D, D, D, D, D, accumulate);
            }

            for (int i = 0; i < cHard.Length; i++)
                Assert.True(
                    BitConverter.SingleToInt32Bits(cHard[i]) == BitConverter.SingleToInt32Bits(cStrided[i]),
                    $"m={m} accumulate={accumulate} element {i}: hardcoded {cHard[i]} vs strided {cStrided[i]} "
                    + "— the two kernels share the k-loop and FMA order, so they must agree bit for bit");
        }
    }

    /// <summary>
    /// The two shapes flash-128 actually needs, neither of which the hardcoded kernel can express:
    /// Q·Kᵀ (k = headDim = 128, n = key tile = 64) and P·V (k = keys = 64, n = headDim = 128).
    /// Checked against the independent triple-loop reference.
    /// </summary>
    [Theory]
    [InlineData(64, 128, 64)]    // Q·Kᵀ at headDim 128
    [InlineData(64, 64, 128)]    // P·V at headDim 128
    [InlineData(64, 128, 128)]
    [InlineData(13, 128, 64)]    // ragged rows, the real tile tail
    [InlineData(6, 40, 24)]      // k and n neither 64 nor 128, n ≡ 8 (mod 16) to hit the 8-wide block
    public void Strided_MatchesReference_AtFlash128Shapes(int m, int k, int n)
    {
        var a = Rand(m * k, 101 + k + n);
        var b = Rand(k * n, 202 + k + n);

        foreach (bool accumulate in new[] { false, true })
        {
            var c0 = Rand(m * n, 303 + k + n);
            var got = (float[])c0.Clone();
            fixed (float* pa = a)
            fixed (float* pb = b)
            fixed (float* pc = got)
                SimdKernels.GemmF32_6x2(pa, pb, pc, m, k, n, k, n, n, accumulate);

            var want = Reference(a, b, m, k, n, k, n, n, accumulate ? c0 : null);

            for (int i = 0; i < want.Length; i++)
            {
                double tol = 1e-4 * Math.Max(1.0, Math.Abs(want[i]));
                Assert.True(Math.Abs(got[i] - want[i]) <= tol,
                    $"m={m} k={k} n={n} accumulate={accumulate} element {i}: got {got[i]}, want {want[i]}");
            }
        }
    }

    /// <summary>
    /// Strides that are LARGER than the logical extent — the case that separates a correct stride
    /// implementation from one that quietly assumes packed rows. If the kernel used k/n where it
    /// should use lda/ldb/ldc, this reads and writes the wrong rows; on packed data that mistake
    /// is invisible, which is why every other test here would miss it.
    /// </summary>
    [Fact]
    public void Strided_HonoursStridesWiderThanExtents()
    {
        const int m = 9, k = 40, n = 24;
        const int lda = 64, ldb = 32, ldc = 48;

        var a = Rand(m * lda, 7);
        var b = Rand(k * ldb, 8);
        var c0 = Rand(m * ldc, 9);

        var got = (float[])c0.Clone();
        fixed (float* pa = a)
        fixed (float* pb = b)
        fixed (float* pc = got)
            SimdKernels.GemmF32_6x2(pa, pb, pc, m, k, n, lda, ldb, ldc, accumulate: false);

        var want = Reference(a, b, m, k, n, lda, ldb, ldc, null);

        for (int i = 0; i < m; i++)
            for (int j = 0; j < ldc; j++)
            {
                int idx = i * ldc + j;
                if (j < n)
                {
                    double tol = 1e-4 * Math.Max(1.0, Math.Abs(want[idx]));
                    Assert.True(Math.Abs(got[idx] - want[idx]) <= tol,
                        $"[{i},{j}] got {got[idx]}, want {want[idx]}");
                }
                else
                {
                    // Columns past n live inside the destination's stride and must be untouched.
                    Assert.Equal(c0[idx], got[idx]);
                }
            }
    }

    /// <summary>n not a multiple of 8 is rejected loudly rather than silently truncated.</summary>
    [Fact]
    public void Strided_RejectsUnsupportedN()
    {
        var a = new float[64];
        var b = new float[64];
        var c = new float[64];
        fixed (float* pa = a)
        fixed (float* pb = b)
        fixed (float* pc = c)
        {
            float* qa = pa, qb = pb, qc = pc;
            Assert.Throws<ArgumentException>(() => SimdKernels.GemmF32_6x2(qa, qb, qc, 1, 8, 12, 8, 12, 12));
        }
    }
}
