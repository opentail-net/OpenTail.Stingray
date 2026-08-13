using System.Runtime.InteropServices;
using OpenTail.Stingray.Cpu;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public unsafe class MicroGemmKernelTests
{
    [Fact]
    public void MicroGemm_GatedOffByDefault()
    {
        // Assert that MicroGemmConfig is disabled by default unless explicitly enabled
        MicroGemmConfig.IsEnabled = false;

        float[] a = new float[4 * 8];
        float[] b = new float[8 * 16];
        float[] c = new float[4 * 16];

        fixed (float* pa = a)
        fixed (float* pb = b)
        fixed (float* pc = c)
        {
            bool executed = MicroGemmKernel.TryMatMulF32(pa, pb, pc, 4, 8, 16);
            Assert.False(executed, "MicroGemm must not execute when IsEnabled is false.");
        }
    }

    [Fact]
    public void MicroGemm_ExecutesAndMatchesNaiveMatMul_WhenEnabled()
    {

        MicroGemmConfig.IsEnabled = true;
        try
        {
            int m = 4;
            int k = 16;
            int n = 8;

            float[] a = new float[m * k];
            float[] b = new float[n * k]; // Row-major [N x K] as [K x N]^T
            float[] cActual = new float[m * n];
            float[] cExpected = new float[m * n];

            // Fill input matrices with deterministic non-zero test data
            for (int i = 0; i < a.Length; i++) a[i] = (i + 1) * 0.25f;
            for (int i = 0; i < b.Length; i++) b[i] = (i + 1) * 0.5f;

            // Naive reference matmul: C[i, j] = sum_l (A[i, l] * B[j, l])
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0f;
                    for (int l = 0; l < k; l++)
                    {
                        sum += a[i * k + l] * b[j * k + l];
                    }
                    cExpected[i * n + j] = sum;
                }
            }

            fixed (float* pa = a)
            fixed (float* pb = b)
            fixed (float* pc = cActual)
            {
                bool executed = MicroGemmKernel.TryMatMulF32(pa, pb, pc, m, k, n);
                Assert.True(executed, "MicroGemm should execute when enabled for m <= 64.");
            }

            // Compare result against reference within floating point precision
            for (int idx = 0; idx < cExpected.Length; idx++)
            {
                Assert.Equal(cExpected[idx], cActual[idx], precision: 4);
            }
        }
        finally
        {
            // Restore default off state
            MicroGemmConfig.IsEnabled = false;
        }
    }
}
