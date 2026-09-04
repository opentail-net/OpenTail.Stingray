
namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Opt-in feature flag configuration for the Fused Small-Batch Micro-GEMM CPU kernel.
/// Gated behind <c>STINGRAY_CPU_MICRO_GEMM=1</c>.
/// Defaults to <b>false (disabled)</b>.
/// </summary>
public static class MicroGemmConfig
{
    private static bool _isEnabled = ReadFromEnvironment();

    /// <summary>
    /// Gets or sets whether the Fused Small-Batch Micro-GEMM kernel is active for small batch sizes (M &lt;= 64).
    /// </summary>
    public static bool IsEnabled
    {
        get => _isEnabled;
        set => _isEnabled = value;
    }

    /// <summary>
    /// Maximum prompt sequence batch size (M) for which Micro-GEMM can be attempted when enabled.
    /// Default: 64 tokens.
    /// </summary>
    public static int MaxMicroBatchSize { get; set; } = 64;

    /// <summary>
    /// Reads <c>STINGRAY_CPU_MICRO_GEMM</c> from environment.
    /// </summary>
    public static bool ReadFromEnvironment()
    {
        string? val = Environment.GetEnvironmentVariable("STINGRAY_CPU_MICRO_GEMM");
        if (string.IsNullOrWhiteSpace(val)) return false;
        val = val.Trim();
        return val == "1" || val.Equals("true", StringComparison.OrdinalIgnoreCase);
    }
}

/// <summary>
/// Fused Small-Batch Micro-GEMM kernel optimized for small prompt batch sizes (M in [2, 64]).
/// Bypasses multi-threaded OpenBLAS thread-pool dispatch overhead on short prompts by executing
/// inline unrolled AVX2 SIMD matrix-multi-vector accumulators.
/// </summary>
public static unsafe class MicroGemmKernel
{
    /// <summary>
    /// Attempts to execute a fused small-batch GEMM operation for dense F32 matrix multiplication
    /// [M x K] * [K x N]^T -&gt; [M x N] when <see cref="MicroGemmConfig.IsEnabled"/> is true and M &lt;= 64.
    /// </summary>
    /// <returns>True if the micro-kernel executed; false if disabled or shape not handled.</returns>
    public static bool TryMatMulF32(
        float* a,
        float* b,
        float* c,
        int m,
        int k,
        int n)
    {
        if (!MicroGemmConfig.IsEnabled || m < 1 || m > MicroGemmConfig.MaxMicroBatchSize)
            return false;

        if (!Avx2.IsSupported)
            return false;

        // Process N columns in steps of 1 (outer col loop) and M rows in unrolled blocks of up to 4
        int mBlocks = m / 4;

        for (int j = 0; j < n; j++)
        {
            float* bCol = b + (nuint)j * (nuint)k;

            // Process 4 rows of A simultaneously per column j
            int i = 0;
            for (; i < mBlocks * 4; i += 4)
            {
                float* aRow0 = a + (nuint)(i + 0) * (nuint)k;
                float* aRow1 = a + (nuint)(i + 1) * (nuint)k;
                float* aRow2 = a + (nuint)(i + 2) * (nuint)k;
                float* aRow3 = a + (nuint)(i + 3) * (nuint)k;

                Vector256<float> acc0 = Vector256<float>.Zero;
                Vector256<float> acc1 = Vector256<float>.Zero;
                Vector256<float> acc2 = Vector256<float>.Zero;
                Vector256<float> acc3 = Vector256<float>.Zero;

                int l = 0;
                for (; l <= k - 8; l += 8)
                {
                    Vector256<float> vb = Avx.LoadVector256(bCol + l);
                    acc0 = Fma.MultiplyAdd(Avx.LoadVector256(aRow0 + l), vb, acc0);
                    acc1 = Fma.MultiplyAdd(Avx.LoadVector256(aRow1 + l), vb, acc1);
                    acc2 = Fma.MultiplyAdd(Avx.LoadVector256(aRow2 + l), vb, acc2);
                    acc3 = Fma.MultiplyAdd(Avx.LoadVector256(aRow3 + l), vb, acc3);
                }

                float sum0 = HorizontalSum256(acc0);
                float sum1 = HorizontalSum256(acc1);
                float sum2 = HorizontalSum256(acc2);
                float sum3 = HorizontalSum256(acc3);

                // Handle remaining elements in K
                for (; l < k; l++)
                {
                    float bv = bCol[l];
                    sum0 += aRow0[l] * bv;
                    sum1 += aRow1[l] * bv;
                    sum2 += aRow2[l] * bv;
                    sum3 += aRow3[l] * bv;
                }

                c[(i + 0) * n + j] = sum0;
                c[(i + 1) * n + j] = sum1;
                c[(i + 2) * n + j] = sum2;
                c[(i + 3) * n + j] = sum3;
            }

            // Remainder rows (1..3)
            for (; i < m; i++)
            {
                float* aRow = a + (nuint)i * (nuint)k;
                Vector256<float> acc = Vector256<float>.Zero;

                int l = 0;
                for (; l <= k - 8; l += 8)
                {
                    Vector256<float> va = Avx.LoadVector256(aRow + l);
                    Vector256<float> vb = Avx.LoadVector256(bCol + l);
                    acc = Fma.MultiplyAdd(va, vb, acc);
                }

                float sum = HorizontalSum256(acc);
                for (; l < k; l++)
                {
                    sum += aRow[l] * bCol[l];
                }

                c[i * n + j] = sum;
            }
        }

        return true;
    }

    private static float HorizontalSum256(Vector256<float> v)
    {
        Vector128<float> low = v.GetLower();
        Vector128<float> high = v.GetUpper();
        Vector128<float> sum128 = Sse.Add(low, high);
        Vector128<float> hsum1 = Sse3.HorizontalAdd(sum128, sum128);
        Vector128<float> hsum2 = Sse3.HorizontalAdd(hsum1, hsum1);
        return hsum2.ToScalar();
    }
}
