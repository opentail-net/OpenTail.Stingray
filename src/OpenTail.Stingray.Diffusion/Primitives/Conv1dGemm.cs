using System.Runtime.InteropServices;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Diffusion.Primitives;

/// <summary>
/// Channel-major ([channels, time]) Conv1d and ConvTranspose1d as tiled GEMM over <see cref="PackedSgemmF32"/>, shared
/// by the audio VAEs / vocoders whose direct forms stream the whole time axis once per (channel, channel, tap): the
/// ACE-Step Oobleck decoder (88 -> 6.5 s per 10 s decode, 2026-09-28) and the MiniMax-Music3 vocoder. Results agree
/// with the direct forms to FP32 rounding (different summation order), not bitwise.
/// </summary>
internal static class Conv1dGemm
{
    /// <summary>True when the GEMM path is available and the problem is big enough to pay for packing.</summary>
    public static bool Applies(int t, int outCh) => PackedSgemmF32.IsSupported && t >= PackedSgemmF32.Mc && outCh >= 16;

    /// <summary>
    /// Stride-1 Conv1d as GEMM: out[oc, t] = Σ_(ic,k) W[oc, ic·K + k] · x[ic, t + k·dilation − padding]. The conv
    /// weight's [outCh][inCh][K] layout is already the GEMM's row-major [n, k] Linear weight, packed once per call;
    /// time is processed in tiles whose im2col panel [rows, inCh·K] stays a few MB, and each tile's time-major result
    /// is transposed back into the channel-major output. Agrees with the direct form to FP32 rounding (different
    /// summation order), not bitwise.
    /// </summary>
    public static unsafe float[] Conv1d(float[] x, int inCh, int outCh, int t, float[] weight, float[]? bias, int kernel, int dilation, int padding)
    {
        int kk = inCh * kernel;
        var output = new float[outCh * t];
        // Rows per tile: ~2M floats of panel, a multiple of the GEMM's row block, at least one block.
        int rows = Math.Max(PackedSgemmF32.Mc, (2 * 1024 * 1024 / kk) / PackedSgemmF32.Mc * PackedSgemmF32.Mc);
        rows = Math.Min(rows, t);
        fixed (float* px = x, pw = weight, pb = bias, po = output)
        {
            float* packed = PackedSgemmF32.PackWeights(pw, outCh, kk);
            float* panel = (float*)NativeMemory.AlignedAlloc((nuint)((long)rows * kk * sizeof(float)), 64);
            float* tileOut = (float*)NativeMemory.AlignedAlloc((nuint)((long)rows * outCh * sizeof(float)), 64);
            try
            {
                float* pxL = px, poL = po;
                for (int t0 = 0; t0 < t; t0 += rows)
                {
                    int tn = Math.Min(rows, t - t0);
                    // im2col: panel[r, ic·K + k] = x[ic, t0 + r + k·dilation − padding], zero outside [0, t).
                    Parallel.For(0, inCh, ic =>
                    {
                        float* xRow = pxL + (long)ic * t;
                        for (int k = 0; k < kernel; k++)
                        {
                            int shift = t0 + k * dilation - padding;
                            float* dst = panel + ic * kernel + k;
                            for (int r = 0; r < tn; r++)
                            {
                                int src = shift + r;
                                dst[(long)r * kk] = (uint)src < (uint)t ? xRow[src] : 0f;
                            }
                        }
                    });
                    PackedSgemmF32.Gemm(tileOut, panel, packed, pb, tn, outCh, kk);
                    Parallel.For(0, outCh, oc =>
                    {
                        float* dst = poL + (long)oc * t + t0;
                        float* src = tileOut + oc;
                        for (int r = 0; r < tn; r++) dst[r] = src[(long)r * outCh];
                    });
                }
            }
            finally
            {
                NativeMemory.AlignedFree(packed);
                NativeMemory.AlignedFree(panel);
                NativeMemory.AlignedFree(tileOut);
            }
        }
        return output;
    }

    /// <summary>
    /// ConvTranspose1d (weight [inCh, outCh, K]) as GEMM + col2im: Z[t, oc·K + k] = Σ_ic x[ic, t] · W[ic, oc, k] through the packed
    /// FP32 GEMM (weight transposed once to [outCh·K, inCh]), then out[oc, t·stride − padding + k] += Z. Time is tiled;
    /// tiles run in order, so the overlapping scatter-adds of neighbouring tiles never race, and within a tile each
    /// output channel is one worker. Agrees with the direct form to FP32 rounding (the direct form spent 7.2 s of an
    /// ACE-Step 10 s decode here, 2026-09-28).
    /// </summary>
    public static unsafe (float[] Data, int T) ConvTranspose1d(float[] x, int inCh, int outCh, int t, float[] weight, float[] bias, int kernel, int stride, int padding)
    {
        int outT = (t - 1) * stride - 2 * padding + kernel;
        int n = outCh * kernel;
        var output = new float[outCh * outT];
        int rows = Math.Max(PackedSgemmF32.Mc, (2 * 1024 * 1024 / Math.Max(inCh, n)) / PackedSgemmF32.Mc * PackedSgemmF32.Mc);
        rows = Math.Min(rows, t);
        fixed (float* px = x, pw = weight, pb = bias, po = output)
        {
            float* wT = (float*)NativeMemory.Alloc((nuint)((long)n * inCh * sizeof(float)));
            float* packed = null;
            float* panel = (float*)NativeMemory.AlignedAlloc((nuint)((long)rows * inCh * sizeof(float)), 64);
            float* z = (float*)NativeMemory.AlignedAlloc((nuint)((long)rows * n * sizeof(float)), 64);
            try
            {
                float* pwL = pw, pxL = px, poL = po, pbL = pb, wTL = wT;
                Parallel.For(0, inCh, ic =>
                {
                    float* src = pwL + (long)ic * n;
                    for (int j = 0; j < n; j++) wTL[(long)j * inCh + ic] = src[j];
                });
                packed = PackedSgemmF32.PackWeights(wT, n, inCh);
                Parallel.For(0, outCh, oc =>
                {
                    float b = pbL[oc];
                    float* dst = poL + (long)oc * outT;
                    for (int o = 0; o < outT; o++) dst[o] = b;
                });

                for (int t0 = 0; t0 < t; t0 += rows)
                {
                    int tn = Math.Min(rows, t - t0);
                    Parallel.For(0, inCh, ic =>
                    {
                        float* src = pxL + (long)ic * t + t0;
                        for (int r = 0; r < tn; r++) panel[(long)r * inCh + ic] = src[r];
                    });
                    PackedSgemmF32.Gemm(z, panel, packed, null, tn, n, inCh);
                    Parallel.For(0, outCh, oc =>
                    {
                        float* dst = poL + (long)oc * outT;
                        for (int r = 0; r < tn; r++)
                        {
                            int o0 = (t0 + r) * stride - padding;
                            float* zr = z + (long)r * n + (long)oc * kernel;
                            int kStart = o0 < 0 ? -o0 : 0;
                            int kEnd = Math.Min(kernel, outT - o0);
                            for (int k = kStart; k < kEnd; k++) dst[o0 + k] += zr[k];
                        }
                    });
                }
            }
            finally
            {
                NativeMemory.Free(wT);
                if (packed != null) NativeMemory.AlignedFree(packed);
                NativeMemory.AlignedFree(panel);
                NativeMemory.AlignedFree(z);
            }
        }
        return (output, outT);
    }
}
