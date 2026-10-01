namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Generic lower-triangular solve <c>A·X = B</c>, matching ggml's <c>GGML_OP_SOLVE_TRI</c> (ops.cpp
/// <c>ggml_compute_forward_solve_tri_f32</c>): lower triangular A, non-unit diagonal, multiple right-hand sides, batched.
/// Only the lower triangle of A (including the diagonal) is read, as in ggml.
/// </summary>
public static class TriangularSolve
{
    /// <summary>
    /// Solves for every batch and every right-hand-side column. Row-major layouts: <paramref name="a"/> is [batch, n, n],
    /// <paramref name="b"/> and <paramref name="x"/> are [batch, n, k]. <paramref name="x"/> may alias <paramref name="b"/>.
    /// </summary>
    public static void SolveLower(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> x, int n, int k, int batch = 1)
    {
        if (n < 0 || k < 0 || batch < 0) throw new ArgumentOutOfRangeException(nameof(n));
        long aLen = (long)batch * n * n, bLen = (long)batch * n * k;
        if (a.Length < aLen) throw new ArgumentException("A is smaller than batch*n*n.", nameof(a));
        if (b.Length < bLen) throw new ArgumentException("B is smaller than batch*n*k.", nameof(b));
        if (x.Length < bLen) throw new ArgumentException("X is smaller than batch*n*k.", nameof(x));

        for (int bi = 0; bi < batch; bi++)
        {
            var ab = a.Slice(bi * n * n, n * n);
            var bb = b.Slice(bi * n * k, n * k);
            var xb = x.Slice(bi * n * k, n * k);
            // Forward substitution, row by row, vectorised across the k right-hand sides: x[i,:] = (b[i,:] - sum_{t<i} a[i,t]*x[t,:]) / a[i,i]
            for (int i = 0; i < n; i++)
            {
                float diag = ab[i * n + i];
                if (diag == 0f) throw new ArgumentException($"Zero diagonal in triangular matrix at row {i}.", nameof(a));
                var xi = xb.Slice(i * k, k);
                bb.Slice(i * k, k).CopyTo(xi);
                for (int t = 0; t < i; t++)
                {
                    float coef = ab[i * n + t];
                    if (coef == 0f) continue;
                    var xt = xb.Slice(t * k, k);
                    for (int c = 0; c < k; c++) xi[c] -= coef * xt[c];
                }
                float inv = 1f / diag;
                for (int c = 0; c < k; c++) xi[c] *= inv;
            }
        }
    }
}
