using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Cache-blocked FP32 GEMM for <c>Y[m, n] = X[m, k] · Wᵀ + bias</c> where <c>W</c> is a
/// row-major <c>[n, k]</c> Linear weight, repacked once into 16-column panels
/// (<c>[ceil(n/16)][k][16]</c>, zero-padded) by <see cref="PackWeights"/>.
///
/// <para><b>Why this exists.</b> <see cref="SimdKernels.MatMulBatchedF32"/> computes each output
/// as a full-length dot product (2 tokens × 4 weight rows, horizontal reductions at the end) and
/// re-streams the whole activation matrix once per 4 weight rows. That is the right shape for
/// decode-sized batches but leaves most of the FMA throughput unused once <c>m</c> is in the
/// hundreds/thousands (diffusion DiT token counts). This kernel uses the classic 6×16
/// broadcast-FMA register tile (12 YMM accumulators, the same shape as
/// <see cref="SimdKernels.GemmF32_6x2"/>) over K-blocks small enough that a panel slice stays in
/// L1 and an activation row-block stays in L2.</para>
///
/// <para>Summation order differs from the dot-product kernel (K is accumulated in <see cref="Kc"/>
/// blocks, lane-wise), so results agree to FP32 rounding, not bitwise.</para>
/// </summary>
public static unsafe class PackedSgemmF32
{
    public const int Nr = 16;
    public const int Mr = 6;
    /// <summary>K-block: a 256×16 panel slice is 16 KB (L1-resident across a row block).</summary>
    public const int Kc = 256;
    /// <summary>Rows per work item: 96×256×4 = 96 KB activation slice (L2-resident across panels).</summary>
    public const int Mc = 96;
    /// <summary>Panels per work item (64 output columns).</summary>
    public const int NcPanels = 4;

    public static bool IsSupported => Avx2.IsSupported && Fma.IsSupported;

    public static long PackedFloats(int n, int k) => (long)((n + Nr - 1) / Nr) * k * Nr;

    /// <summary>Allocates (64-byte aligned native memory) and fills the panel layout. Free with
    /// <see cref="NativeMemory.AlignedFree"/>.</summary>
    public static float* PackWeights(float* w, int n, int k)
    {
        long count = PackedFloats(n, k);
        var dst = (float*)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);
        int panels = (n + Nr - 1) / Nr;
        Parallel.For(0, panels, p =>
        {
            float* pd = dst + (long)p * k * Nr;
            int c0 = p * Nr;
            int valid = Math.Min(Nr, n - c0);
            for (int j = 0; j < Nr; j++)
            {
                if (j < valid)
                {
                    float* src = w + (long)(c0 + j) * k;
                    for (int kk = 0; kk < k; kk++) pd[(long)kk * Nr + j] = src[kk];
                }
                else
                {
                    for (int kk = 0; kk < k; kk++) pd[(long)kk * Nr + j] = 0f;
                }
            }
        });
        return dst;
    }

    public static void Gemm(float* output, float* input, float* packedW, float* bias, int m, int n, int k)
    {
        if (!IsSupported) throw new PlatformNotSupportedException("PackedSgemmF32 requires AVX2 and FMA.");
        if (m <= 0 || n <= 0) return;

        int panels = (n + Nr - 1) / Nr;
        int mBlocks = (m + Mc - 1) / Mc;
        int nBlocks = (panels + NcPanels - 1) / NcPanels;
        int tasks = mBlocks * nBlocks;

        nint o = (nint)output, x = (nint)input, w = (nint)packedW, b = (nint)bias;
        if (tasks == 1)
        {
            RunTile(output, input, packedW, bias, m, n, k, 0, 0);
            return;
        }
        Parallel.For(0, tasks, t =>
        {
            // n-block outer so neighbouring tasks share the same activation rows in L3.
            int mb = t % mBlocks, nb = t / mBlocks;
            RunTile((float*)o, (float*)x, (float*)w, (float*)b, m, n, k, mb, nb);
        });
    }

    private static void RunTile(float* output, float* input, float* packedW, float* bias,
        int m, int n, int k, int mb, int nb)
    {
        int r0 = mb * Mc, r1 = Math.Min(m, r0 + Mc);
        int p0 = nb * NcPanels, p1 = Math.Min((n + Nr - 1) / Nr, p0 + NcPanels);
        float* tmp = stackalloc float[Mr * Nr];

        for (int k0 = 0; k0 < k; k0 += Kc)
        {
            int kc = Math.Min(Kc, k - k0);
            bool acc = k0 > 0;
            for (int p = p0; p < p1; p++)
            {
                int c0 = p * Nr;
                int valid = Math.Min(Nr, n - c0);
                float* bp = packedW + (long)p * k * Nr + (long)k0 * Nr;
                int r = r0;
                if (valid == Nr)
                {
                    for (; r + Mr <= r1; r += Mr)
                        Kernel6x16(input + (long)r * k + k0, k, bp, kc, output + (long)r * n + c0, n, acc);
                    for (; r < r1; r++)
                        Kernel1x16(input + (long)r * k + k0, bp, kc, output + (long)r * n + c0, acc);
                }
                else
                {
                    // Ragged last panel: compute a full 16-wide row into scratch, copy the valid part.
                    for (; r < r1; r++)
                    {
                        float* orow = output + (long)r * n + c0;
                        if (acc) { for (int j = 0; j < valid; j++) tmp[j] = orow[j]; }
                        Kernel1x16(input + (long)r * k + k0, bp, kc, tmp, acc);
                        for (int j = 0; j < valid; j++) orow[j] = tmp[j];
                    }
                }
            }
        }

        if (bias != null)
        {
            int c0 = p0 * Nr, c1 = Math.Min(n, p1 * Nr);
            for (int r = r0; r < r1; r++)
            {
                float* orow = output + (long)r * n;
                int c = c0;
                for (; c + 8 <= c1; c += 8)
                    Avx.Store(orow + c, Avx.Add(Avx.LoadVector256(orow + c), Avx.LoadVector256(bias + c)));
                for (; c < c1; c++) orow[c] += bias[c];
            }
        }
    }

    /// <summary>
    /// Same GEMM with <c>W</c> left in its GGUF block-quantized form: each task owns
    /// <see cref="NcPanels"/> panels (64 output rows) for ALL <paramref name="m"/> rows, and for each
    /// K-block dequantizes just those 64 rows × <see cref="Kc"/> into a thread-local panel buffer
    /// that stays in L2 while the 6×16 kernel sweeps every token. Each weight element is
    /// dequantized once per call, and activations stay FP32, unlike the int8-activation
    /// (Q8) paths, which are lossy on wide-range diffusion activations. For weights too large to
    /// keep a dequantized FP32 copy of.
    /// </summary>
    public static bool CanGemmQuant(DType dtype, int k)
    {
        int bs = DTypeInfo.BlockSize(dtype);
        return IsSupported && bs > 1 && Kc % bs == 0 && k % bs == 0;
    }

    public static void GemmQuant(float* output, float* input, byte* weights, DType dtype, int m, int n, int k)
    {
        if (!CanGemmQuant(dtype, k)) throw new NotSupportedException($"GemmQuant: {dtype} with k={k}");
        if (m <= 0 || n <= 0) return;

        int bs = DTypeInfo.BlockSize(dtype);
        long rowBytes = (long)(k / bs) * DTypeInfo.BytesPerBlock(dtype);
        int panels = (n + Nr - 1) / Nr;
        int nBlocks = (panels + NcPanels - 1) / NcPanels;

        nint o = (nint)output, x = (nint)input, w = (nint)weights;
        Parallel.For(0, nBlocks,
            () => (Buf: (nint)NativeMemory.AlignedAlloc((nuint)(NcPanels * Kc * Nr * sizeof(float)), 64),
                   Row: new float[Kc]),
            (nb, _, local) =>
            {
                float* bufBase = (float*)local.Buf;
                int p0 = nb * NcPanels, p1 = Math.Min(panels, p0 + NcPanels);
                int c0 = p0 * Nr, c1 = Math.Min(n, p1 * Nr);
                float* tmp = stackalloc float[Mr * Nr];
                for (int k0 = 0; k0 < k; k0 += Kc)
                {
                    int kc = Math.Min(Kc, k - k0);
                    int srcBytes = kc / bs * DTypeInfo.BytesPerBlock(dtype);
                    // Dequantize rows [c0, c1) x [k0, k0+kc) straight into panel layout.
                    for (int c = c0; c < p1 * Nr; c++)
                    {
                        int p = c / Nr - p0, j = c % Nr;
                        float* pd = bufBase + (long)p * Kc * Nr + j;
                        if (c < c1)
                        {
                            var src = new ReadOnlySpan<byte>((byte*)w + c * rowBytes + (long)k0 / bs * DTypeInfo.BytesPerBlock(dtype), srcBytes);
                            Dequantize.ToFloat32(src, local.Row.AsSpan(0, kc), dtype, kc);
                            for (int kk = 0; kk < kc; kk++) pd[kk * Nr] = local.Row[kk];
                        }
                        else
                        {
                            for (int kk = 0; kk < kc; kk++) pd[kk * Nr] = 0f;
                        }
                    }

                    bool acc = k0 > 0;
                    for (int p = p0; p < p1; p++)
                    {
                        int cc = p * Nr;
                        int valid = Math.Min(Nr, n - cc);
                        float* bp = bufBase + (long)(p - p0) * Kc * Nr;
                        int r = 0;
                        if (valid == Nr)
                        {
                            for (; r + Mr <= m; r += Mr)
                                Kernel6x16((float*)x + (long)r * k + k0, k, bp, kc, (float*)o + (long)r * n + cc, n, acc);
                            for (; r < m; r++)
                                Kernel1x16((float*)x + (long)r * k + k0, bp, kc, (float*)o + (long)r * n + cc, acc);
                        }
                        else
                        {
                            for (; r < m; r++)
                            {
                                float* orow = (float*)o + (long)r * n + cc;
                                if (acc) { for (int j = 0; j < valid; j++) tmp[j] = orow[j]; }
                                Kernel1x16((float*)x + (long)r * k + k0, bp, kc, tmp, acc);
                                for (int j = 0; j < valid; j++) orow[j] = tmp[j];
                            }
                        }
                    }
                }
                return local;
            },
            local => NativeMemory.AlignedFree((void*)local.Buf));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Kernel6x16(float* a, int lda, float* bp, int kc, float* c, int ldc, bool accumulate)
    {
        float* a0 = a, a1 = a + lda, a2 = a + 2 * lda, a3 = a + 3 * lda, a4 = a + 4 * lda, a5 = a + 5 * lda;
        float* c0 = c, c1 = c + ldc, c2 = c + 2 * ldc, c3 = c + 3 * ldc, c4 = c + 4 * ldc, c5 = c + 5 * ldc;

        Vector256<float> x00, x01, x10, x11, x20, x21, x30, x31, x40, x41, x50, x51;
        if (accumulate)
        {
            x00 = Avx.LoadVector256(c0); x01 = Avx.LoadVector256(c0 + 8);
            x10 = Avx.LoadVector256(c1); x11 = Avx.LoadVector256(c1 + 8);
            x20 = Avx.LoadVector256(c2); x21 = Avx.LoadVector256(c2 + 8);
            x30 = Avx.LoadVector256(c3); x31 = Avx.LoadVector256(c3 + 8);
            x40 = Avx.LoadVector256(c4); x41 = Avx.LoadVector256(c4 + 8);
            x50 = Avx.LoadVector256(c5); x51 = Avx.LoadVector256(c5 + 8);
        }
        else
        {
            x00 = x01 = x10 = x11 = x20 = x21 = x30 = x31 = x40 = x41 = x50 = x51 = Vector256<float>.Zero;
        }

        float* bk = bp;
        for (int kk = 0; kk < kc; kk++, bk += Nr)
        {
            var b0 = Avx.LoadAlignedVector256(bk);
            var b1 = Avx.LoadAlignedVector256(bk + 8);
            var q = Avx.BroadcastScalarToVector256(a0 + kk);
            x00 = Fma.MultiplyAdd(q, b0, x00); x01 = Fma.MultiplyAdd(q, b1, x01);
            q = Avx.BroadcastScalarToVector256(a1 + kk);
            x10 = Fma.MultiplyAdd(q, b0, x10); x11 = Fma.MultiplyAdd(q, b1, x11);
            q = Avx.BroadcastScalarToVector256(a2 + kk);
            x20 = Fma.MultiplyAdd(q, b0, x20); x21 = Fma.MultiplyAdd(q, b1, x21);
            q = Avx.BroadcastScalarToVector256(a3 + kk);
            x30 = Fma.MultiplyAdd(q, b0, x30); x31 = Fma.MultiplyAdd(q, b1, x31);
            q = Avx.BroadcastScalarToVector256(a4 + kk);
            x40 = Fma.MultiplyAdd(q, b0, x40); x41 = Fma.MultiplyAdd(q, b1, x41);
            q = Avx.BroadcastScalarToVector256(a5 + kk);
            x50 = Fma.MultiplyAdd(q, b0, x50); x51 = Fma.MultiplyAdd(q, b1, x51);
        }

        Avx.Store(c0, x00); Avx.Store(c0 + 8, x01);
        Avx.Store(c1, x10); Avx.Store(c1 + 8, x11);
        Avx.Store(c2, x20); Avx.Store(c2 + 8, x21);
        Avx.Store(c3, x30); Avx.Store(c3 + 8, x31);
        Avx.Store(c4, x40); Avx.Store(c4 + 8, x41);
        Avx.Store(c5, x50); Avx.Store(c5 + 8, x51);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Kernel1x16(float* a, float* bp, int kc, float* c, bool accumulate)
    {
        var x0 = accumulate ? Avx.LoadVector256(c) : Vector256<float>.Zero;
        var x1 = accumulate ? Avx.LoadVector256(c + 8) : Vector256<float>.Zero;
        float* bk = bp;
        for (int kk = 0; kk < kc; kk++, bk += Nr)
        {
            var q = Avx.BroadcastScalarToVector256(a + kk);
            x0 = Fma.MultiplyAdd(q, Avx.LoadAlignedVector256(bk), x0);
            x1 = Fma.MultiplyAdd(q, Avx.LoadAlignedVector256(bk + 8), x1);
        }
        Avx.Store(c, x0); Avx.Store(c + 8, x1);
    }
}
