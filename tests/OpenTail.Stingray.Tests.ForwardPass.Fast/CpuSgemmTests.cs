using System.Runtime.InteropServices;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Contract for <see cref="CpuSgemm"/> (the pure-C# BLIS-style F32 GEMM ported from TensorSharp to
/// replace OpenBLAS). Every supported microkernel is checked against a double-precision reference on
/// shapes that hit each driver branch: skinny (M ≤ 4), narrow dot (small N, K-contiguous operands),
/// serial packed tiles past the MC/NC blocks, parallel tiles, edge tiles and K tails, in all four
/// transpose layouts. Shapes and the tolerance model follow TensorSharp's CpuSgemmTests.
///
/// <para>Kernel pinning is process-global, so every test that pins one lives in this class (xUnit runs
/// a class's tests serially). Other callers only ever see a correct kernel.</para>
/// </summary>
public sealed unsafe class CpuSgemmTests
{
    private static string[] SupportedKernels() =>
        Enum.GetValues<CpuSgemm.KernelKind>().Where(CpuSgemm.IsSupported).Select(k => k.ToString()).ToArray();

    public static TheoryData<string> Kernels() => new(SupportedKernels());

    private sealed class KernelScope : IDisposable
    {
        private readonly CpuSgemm.KernelKind _previous = CpuSgemm.ActiveKernel;
        public KernelScope(string kind) => CpuSgemm.ActiveKernel = Enum.Parse<CpuSgemm.KernelKind>(kind);
        public void Dispose() => CpuSgemm.ActiveKernel = _previous;
    }

    private static float[] Random(int seed, long count)
    {
        var rng = new Random(seed);
        var v = new float[count];
        for (long i = 0; i < count; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    /// <summary>
    /// Asserts C == alpha·A·B + beta·C0 within a K-scaled bound on the products' magnitude (float
    /// accumulation error grows ~ K·eps·Σ|a·b|). A(i,p) = a[i*ars + p*acs], B(p,j) = b[p*brs + j*bcs].
    /// </summary>
    private static void AssertGemm(int m, int n, int k, float alpha, float beta,
        float[] a, long ars, long acs, float[] b, long brs, long bcs, float[] c0, float[] c, long ldc)
    {
        double tolFactor = 2e-7 * (k + 16);
        for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            double acc = 0, mag = 0;
            for (int p = 0; p < k; p++)
            {
                double prod = (double)a[i * ars + p * acs] * b[p * brs + j * bcs];
                acc += prod;
                mag += Math.Abs(prod);
            }
            double before = beta == 0f ? 0 : c0[i * ldc + j];
            double expected = alpha * acc + beta * before;
            double actual = c[i * ldc + j];
            double tol = tolFactor * (Math.Abs(alpha) * mag + Math.Abs(beta * before)) + 1e-6;
            Assert.True(Math.Abs(actual - expected) <= tol,
                $"C[{i},{j}] = {actual}, expected {expected} (tol {tol:E2}), m={m} n={n} k={k}");
        }
    }

    private static void Gemm(int m, int n, int k, float alpha, float[] a, long ars, long acs,
        float[] b, long brs, long bcs, float beta, float[] c, long ldc)
    {
        fixed (float* pa = a) fixed (float* pb = b) fixed (float* pc = c)
            CpuSgemm.Gemm(m, n, k, alpha, pa, ars, acs, pb, brs, bcs, beta, pc, ldc);
    }

    public static TheoryData<string, int, int, int, string> Shapes()
    {
        int[][] shapes =
        [
            [1, 1, 1], [1, 37, 64], [3, 5, 7], [4, 300, 129],
            [5, 1, 33], [8, 32, 1], [9, 33, 17], [13, 47, 256],
            [31, 17, 300], [70, 90, 128], [129, 65, 513], [200, 200, 120],
            // Serial shapes past the MC row block and the NC column block (ic > 0 / jc > 0 loops); in
            // NT, N = 40 and N = 5 take the narrow dot path (ragged rows, a K tail, a long unsplit K).
            [301, 40, 100], [290, 68, 100], [8, 3000, 64], [37, 5, 4133],
            // Parallel narrow products; the last one's B^T exceeds the dot path's L2 budget (K split).
            [1001, 3, 1152], [403, 29, 700], [21, 40, 3500],
            // Diffusion-sized: parallel packed tiles with edge rows and columns.
            [517, 333, 260],
        ];
        var data = new TheoryData<string, int, int, int, string>();
        foreach (var kernel in SupportedKernels())
            foreach (var s in shapes)
                foreach (var layout in new[] { "NN", "NT", "TN", "TT" })
                    data.Add(kernel, s[0], s[1], s[2], layout);
        return data;
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Gemm_MatchesDoubleReference_AllLayouts(string kernel, int m, int n, int k, string layout)
    {
        using var _ = new KernelScope(kernel);
        bool ta = layout[0] == 'T', tb = layout[1] == 'T';
        float[] a = Random(1, (long)m * k), b = Random(2, (long)k * n);
        long ars = ta ? 1 : k, acs = ta ? m : 1;
        long brs = tb ? 1 : n, bcs = tb ? k : 1;

        float[] c = Random(3, (long)m * n), c0 = (float[])c.Clone();
        Gemm(m, n, k, 1f, a, ars, acs, b, brs, bcs, 0f, c, n);
        AssertGemm(m, n, k, 1f, 0f, a, ars, acs, b, brs, bcs, c0, c, n);

        float[] c2 = Random(4, (long)m * n), c20 = (float[])c2.Clone();
        Gemm(m, n, k, -0.5f, a, ars, acs, b, brs, bcs, 0.25f, c2, n);
        AssertGemm(m, n, k, -0.5f, 0.25f, a, ars, acs, b, brs, bcs, c20, c2, n);
    }

    [Theory]
    [MemberData(nameof(Kernels))]
    public void BetaOne_Accumulates_AndAlphaZero_OnlyScales(string kernel)
    {
        using var _ = new KernelScope(kernel);
        const int m = 37, n = 53, k = 301;
        float[] a = Random(5, m * k), b = Random(6, k * n);

        float[] c = Random(7, m * n), c0 = (float[])c.Clone();
        Gemm(m, n, k, 1f, a, k, 1, b, n, 1, 1f, c, n);
        AssertGemm(m, n, k, 1f, 1f, a, k, 1, b, n, 1, c0, c, n);

        float[] d = Random(8, m * n), d0 = (float[])d.Clone();
        Gemm(m, n, k, 0f, a, k, 1, b, n, 1, 0.5f, d, n);
        for (int i = 0; i < d.Length; i++) Assert.Equal(0.5f * d0[i], d[i]);
    }

    [Theory]
    [MemberData(nameof(Kernels))]
    public void BetaZero_IgnoresNaNInOutput(string kernel)
    {
        using var _ = new KernelScope(kernel);
        foreach (var (m, n, k) in new[] { (2, 40, 33), (64, 48, 100), (300, 7, 129) })
        {
            float[] a = Random(9, m * k), b = Random(10, k * n);
            var c = new float[m * n];
            Array.Fill(c, float.NaN);
            Gemm(m, n, k, 1f, a, k, 1, b, 1, k, 0f, c, n);
            Assert.All(c, v => Assert.False(float.IsNaN(v)));
        }
    }

    [Theory]
    [MemberData(nameof(Kernels))]
    public void ArbitraryElementStrides_AndPaddedOutputRows(string kernel)
    {
        using var _ = new KernelScope(kernel);
        const int m = 45, n = 70, k = 90;
        // A as every other column of a [m, 2k] buffer; B column-major with a padded leading dim; C rows padded.
        float[] a = Random(11, (long)m * 2 * k), b = Random(12, (long)n * (k + 3));
        long ars = 2 * k, acs = 2, brs = 1, bcs = k + 3, ldc = n + 5;
        float[] c = Random(13, m * ldc), c0 = (float[])c.Clone();
        Gemm(m, n, k, 0.75f, a, ars, acs, b, brs, bcs, -1f, c, ldc);
        AssertGemm(m, n, k, 0.75f, -1f, a, ars, acs, b, brs, bcs, c0, c, ldc);
        for (int i = 0; i < m; i++)
            for (long j = n; j < ldc; j++)
                Assert.Equal(c0[i * ldc + j], c[i * ldc + j]); // padding untouched
    }

    [Theory]
    [MemberData(nameof(Kernels))]
    public void GemmBatched_MatchesPerItem(string kernel)
    {
        using var _ = new KernelScope(kernel);
        foreach (var (batch, m, n, k) in new[] { (8, 16, 16, 64), (3, 100, 50, 64), (12, 1, 77, 64), (5, 33, 9, 200) })
        {
            float[] a = Random(14, (long)batch * m * k), b = Random(15, (long)batch * n * k);
            var c = new float[batch * m * n];
            fixed (float* pa = a) fixed (float* pb = b) fixed (float* pc = c)
                CpuSgemm.GemmBatched(batch, m, n, k, 1f, pa, (long)m * k, k, 1, pb, (long)n * k, 1, k,
                    0f, pc, (long)m * n, n);
            for (int bi = 0; bi < batch; bi++)
            {
                float[] ai = a.AsSpan(bi * m * k, m * k).ToArray(), bItem = b.AsSpan(bi * n * k, n * k).ToArray();
                float[] ci = c.AsSpan(bi * m * n, m * n).ToArray();
                AssertGemm(m, n, k, 1f, 0f, ai, k, 1, bItem, 1, k, ci, ci, n);
            }
        }
    }

    [Theory]
    [MemberData(nameof(Kernels))]
    public void NarrowDot_TailMask_DoesNotTurnInfIntoNaN(string kernel)
    {
        using var _ = new KernelScope(kernel);
        // K = 37 is not a multiple of any lane count: the tail re-reads the last full vector with the
        // already-summed lanes masked off. An Inf in a masked lane must not come back as 0·Inf = NaN.
        const int m = 8, n = 3, k = 37;
        float[] a = Random(16, m * k), b = Random(17, n * k);
        a[29] = float.PositiveInfinity; // row 0, lane 29: summed in the main loop, masked in the tail re-read
        var c = new float[m * n];
        Gemm(m, n, k, 1f, a, k, 1, b, 1, k, 0f, c, n);
        for (int j = 0; j < n; j++)
            Assert.True(float.IsInfinity(c[j]), $"C[0,{j}] = {c[j]} (Inf input must stay Inf, not NaN)");
        for (int i = 1; i < m; i++)
            for (int j = 0; j < n; j++)
                Assert.True(float.IsFinite(c[i * n + j]));
    }

    [Theory]
    [MemberData(nameof(Kernels))]
    public void NarrowDot_MatchesPackedPath(string kernel)
    {
        using var _ = new KernelScope(kernel);
        try
        {
            foreach (var (m, n, k) in new[] { (300, 16, 288), (77, 40, 1000), (1001, 3, 1152) })
            {
                float[] a = Random(18, (long)m * k), b = Random(19, (long)n * k);
                var dot = new float[m * n];
                var packed = new float[m * n];
                CpuSgemm.NarrowDotMaxN = 64;
                Gemm(m, n, k, 1f, a, k, 1, b, 1, k, 0f, dot, n);
                CpuSgemm.NarrowDotMaxN = 0;
                Gemm(m, n, k, 1f, a, k, 1, b, 1, k, 0f, packed, n);
                AssertGemm(m, n, k, 1f, 0f, a, k, 1, b, 1, k, dot, dot, n);
                AssertGemm(m, n, k, 1f, 0f, a, k, 1, b, 1, k, packed, packed, n);
            }
        }
        finally { CpuSgemm.NarrowDotMaxN = -1; }
    }

    [Fact]
    public void KernelSelection_HonoursSupport()
    {
        Assert.True(CpuSgemm.IsSupported(CpuSgemm.KernelKind.Portable));
        Assert.True(CpuSgemm.IsSupported(CpuSgemm.DefaultKernel()));
        foreach (var kind in Enum.GetValues<CpuSgemm.KernelKind>())
            if (!CpuSgemm.IsSupported(kind))
                Assert.Throws<PlatformNotSupportedException>(() => CpuSgemm.ActiveKernel = kind);
        Assert.False(string.IsNullOrEmpty(CpuSgemm.ActiveKernelName));
    }

    [Fact]
    public void EmptyAndDegenerateShapes()
    {
        var c = new float[] { 3f, 4f };
        Gemm(0, 2, 5, 1f, [], 5, 1, new float[10], 2, 1, 0f, c, 2);   // m = 0: untouched
        Assert.Equal([3f, 4f], c);
        Gemm(1, 2, 0, 1f, [], 0, 1, [], 2, 1, 0.5f, c, 2);            // k = 0: C = beta·C
        Assert.Equal([1.5f, 2f], c);
    }

    [Fact]
    public void ThreadBuffers_ReusedAcrossShapes_StayCorrect()
    {
        // Grow the per-thread packing buffers with a large product, then run a small one on the same
        // threads: stale panel contents beyond the small shape must not leak into its result.
        foreach (var (m, n, k) in new[] { (400, 400, 700), (13, 21, 30), (250, 90, 513), (7, 7, 7) })
        {
            float[] a = Random(20 + m, (long)m * k), b = Random(21 + n, (long)k * n);
            var c = new float[m * n];
            Gemm(m, n, k, 1f, a, k, 1, b, n, 1, 0f, c, n);
            AssertGemm(m, n, k, 1f, 0f, a, k, 1, b, n, 1, c, c, n);
        }
    }

    [Fact]
    public void Gemm_IsDeterministic_AcrossRuns()
    {
        const int m = 333, n = 257, k = 300;
        float[] a = Random(30, m * k), b = Random(31, n * k);
        var first = new float[m * n];
        Gemm(m, n, k, 1f, a, k, 1, b, 1, k, 0f, first, n);
        for (int r = 0; r < 5; r++)
        {
            var again = new float[m * n];
            Gemm(m, n, k, 1f, a, k, 1, b, 1, k, 0f, again, n);
            Assert.Equal(first, again); // tiles own disjoint C regions and a fixed K order: bitwise
        }
    }
}
