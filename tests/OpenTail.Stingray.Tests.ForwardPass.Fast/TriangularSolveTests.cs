namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// docs/1-correctness/21: the generic lower-triangular solve (GGML_OP_SOLVE_TRI semantics) against an independent double-precision
/// forward-substitution written column by column in ggml's loop order, plus the self-check A·X = B.
/// </summary>
public sealed class TriangularSolveTests
{
    private static (float[] A, float[] B) Make(int n, int k, int batch, Random rnd, bool poisonUpper)
    {
        var a = new float[batch * n * n]; var b = new float[batch * n * k];
        for (int bi = 0; bi < batch; bi++)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    float v = j < i ? (float)(rnd.NextDouble() - 0.5) * 0.6f
                            : j == i ? (float)(1.0 + rnd.NextDouble()) * (rnd.Next(2) == 0 ? 1 : -1)
                            : (poisonUpper ? float.NaN : (float)rnd.NextDouble());
                    a[bi * n * n + i * n + j] = v;
                }
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rnd.NextDouble() * 4 - 2);
        return (a, b);
    }

    private static double[] Oracle(float[] a, float[] b, int n, int k, int batch)
    {
        var x = new double[batch * n * k];
        for (int bi = 0; bi < batch; bi++)
            for (int col = 0; col < k; col++)
                for (int i = 0; i < n; i++)
                {
                    double sum = 0;
                    for (int t = 0; t < i; t++) sum += (double)a[bi * n * n + i * n + t] * x[bi * n * k + t * k + col];
                    x[bi * n * k + i * k + col] = (b[bi * n * k + i * k + col] - sum) / a[bi * n * n + i * n + i];
                }
        return x;
    }

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(2, 1, 1)]
    [InlineData(5, 3, 1)]
    [InlineData(16, 7, 3)]
    [InlineData(33, 17, 2)]
    [InlineData(64, 64, 2)]
    [InlineData(7, 130, 1)]
    public void SolveLower_MatchesTheOracle_IgnoresTheUpperTriangle_AndSatisfiesAxEqualsB(int n, int k, int batch)
    {
        var (a, b) = Make(n, k, batch, new Random(n * 1000 + k * 10 + batch), poisonUpper: true);   // NaN above the diagonal must never be read
        var x = new float[b.Length];
        OpenTail.Stingray.Cpu.TriangularSolve.SolveLower(a, b, x, n, k, batch);

        var expected = Oracle(a, b, n, k, batch);
        double scale = expected.Max(Math.Abs) + 1e-6;
        for (int i = 0; i < x.Length; i++)
            Assert.True(float.IsFinite(x[i]) && Math.Abs(expected[i] - x[i]) <= 1e-4 * scale, $"x[{i}]: expected {expected[i]:G7}, got {x[i]:G7}");

        for (int bi = 0; bi < batch; bi++)                // A·X reproduces B (lower triangle only)
            for (int i = 0; i < n; i++)
                for (int c = 0; c < k; c++)
                {
                    double s = 0;
                    for (int t = 0; t <= i; t++) s += (double)a[bi * n * n + i * n + t] * x[bi * n * k + t * k + c];
                    Assert.True(Math.Abs(s - b[bi * n * k + i * k + c]) <= 1e-4 * (Math.Abs(b[bi * n * k + i * k + c]) + scale), $"A·X != B at batch {bi} row {i} col {c}");
                }
    }

    [Fact]
    public void SolveLower_InPlace_EqualsOutOfPlace()
    {
        var (a, b) = Make(20, 9, 2, new Random(5), poisonUpper: false);
        var outOfPlace = new float[b.Length];
        OpenTail.Stingray.Cpu.TriangularSolve.SolveLower(a, b, outOfPlace, 20, 9, 2);
        var inPlace = (float[])b.Clone();
        OpenTail.Stingray.Cpu.TriangularSolve.SolveLower(a, inPlace, inPlace, 20, 9, 2);
        Assert.Equal(outOfPlace, inPlace);
    }

    [Fact]
    public void SolveLower_RejectsZeroDiagonalAndShortBuffers()
    {
        var a = new float[] { 1, 0, 2, 0 };   // row 1 diagonal is 0
        Assert.Throws<ArgumentException>(() => OpenTail.Stingray.Cpu.TriangularSolve.SolveLower(a, new float[2], new float[2], 2, 1));
        Assert.Throws<ArgumentException>(() => OpenTail.Stingray.Cpu.TriangularSolve.SolveLower(new float[3], new float[2], new float[2], 2, 1));
        Assert.Throws<ArgumentException>(() => OpenTail.Stingray.Cpu.TriangularSolve.SolveLower(new float[4], new float[2], new float[1], 2, 1));
    }
}
