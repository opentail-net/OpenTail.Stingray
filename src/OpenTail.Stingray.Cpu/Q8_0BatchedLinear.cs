using System.Runtime.Intrinsics.X86;

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Weight-stationary Q8_0 linear over many activation rows that were already quantized with
/// <see cref="SimdKernels.QuantizeRowToQ8_0"/>: parallel over output rows, each weight row dotted
/// against every activation row (four at a time through the fused 4-input dot) while it sits in
/// L1. The per-row alternative, one full matvec per activation row, streams the whole weight
/// matrix from RAM once per row.
/// <para>Every output is bit-identical to <c>SimdKernels.DotQ8_0_Q8_0(weightRow, xRow, cols)</c>
/// (+ bias), since the 4-input dot is pinned bit-identical to the single one.</para>
/// </summary>
public static unsafe class Q8_0BatchedLinear
{
    private const int RowsPerTask = 8;

    /// <summary>Quantizes <paramref name="count"/> rows of <paramref name="cols"/> floats (row
    /// stride <paramref name="cols"/>) to Q8_0 activation scratch, row stride
    /// <see cref="SimdKernels.Q8_0ScratchBytes"/>(cols).</summary>
    public static void QuantizeRows(float* x, int count, int cols, byte* xq)
    {
        int stride = SimdKernels.Q8_0ScratchBytes(cols);
        Parallel.For(0, count, f => SimdKernels.QuantizeRowToQ8_0(x + (long)f * cols, cols, xq + (long)f * stride));
    }

    /// <summary>
    /// <c>output[f * outStride + r] = dot(weights row r, xq row f) + bias[r]</c> for every output
    /// row <c>r &lt; rows</c> and activation row <c>f &lt; count</c>. <paramref name="weights"/> is
    /// row-major Q8_0 (<c>cols / 32 * 34</c> bytes per row); <paramref name="bias"/> may be null.
    /// </summary>
    public static void MatMul(byte* weights, int rows, int cols, byte* xq, int count,
        float* bias, float* output, int outStride)
    {
        int bpr = (cols / 32) * 34;
        int xs = SimdKernels.Q8_0ScratchBytes(cols);
        int nb = cols / 32;
        bool fused = Avx2.IsSupported && Fma.IsSupported;
        int tasks = (rows + RowsPerTask - 1) / RowsPerTask;
        Parallel.For(0, tasks, task =>
        {
            int r0 = task * RowsPerTask, r1 = Math.Min(rows, r0 + RowsPerTask);
            for (int r = r0; r < r1; r++)
            {
                byte* w = weights + (long)r * bpr;
                float b = bias != null ? bias[r] : 0f;
                int f = 0;
                if (fused)
                {
                    for (; f + 4 <= count; f += 4)
                    {
                        byte* s = xq + (long)f * xs;
                        SimdKernels.DotQ8_0_Q8_0_4In_Avx2(w, s, s + xs, s + 2 * xs, s + 3 * xs, nb,
                            out float o0, out float o1, out float o2, out float o3);
                        output[(long)f * outStride + r] = o0 + b;
                        output[(long)(f + 1) * outStride + r] = o1 + b;
                        output[(long)(f + 2) * outStride + r] = o2 + b;
                        output[(long)(f + 3) * outStride + r] = o3 + b;
                    }
                }
                for (; f < count; f++)
                    output[(long)f * outStride + r] = SimdKernels.DotQ8_0_Q8_0(w, xq + (long)f * xs, cols) + b;
            }
        });
    }
}
