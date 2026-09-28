using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Diffusion.AceStep.Vae;

/// <summary>
/// Shared real Oobleck VAE conv/activation primitives used by BOTH <see cref="AceStepOobleckDecoder"/>
/// and <see cref="AceStepOobleckEncoder"/> -- the residual-unit math (`OobleckResidualUnit`), the
/// two-parameter log-scale Snake activation, and the plain/strided Conv1d helper are byte-identical
/// real formulas on both the encode and decode sides (confirmed from the real `diffusers`
/// `autoencoder_oobleck.py` source), so shared here from the start rather than duplicated then
/// DRY'd later -- these are two real, immediately-existing callers of the same math, not a
/// speculative extraction (see CLAUDE.md rule 7).
/// </summary>
internal static class AceStepOobleckKernels
{
    /// <summary>Real Oobleck `Snake1d`: `x + (1/(exp(beta)+1e-9)) * sin(exp(alpha)*x)^2` -- both `alpha` and `beta` are stored in LOG-SCALE, need `exp()` before use.</summary>
    public static float[] Snake(float[] x, int channels, int t, float[] logAlpha, float[] logBeta)
    {
        var output = new float[x.Length];
        Parallel.For(0, channels, c =>
        {
            float alpha = MathF.Exp(logAlpha[c]);
            float beta = MathF.Exp(logBeta[c]);
            float invBeta = 1f / (beta + 1e-9f);
            int baseIdx = c * t;
            for (int i = 0; i < t; i++)
            {
                float v = x[baseIdx + i];
                float s = MathF.Sin(alpha * v);
                output[baseIdx + i] = v + invBeta * s * s;
            }
        });
        return output;
    }

    public static float[] ResidualUnit(float[] x, int channels, int t, OobleckResidualUnitWeights w, int dilation)
    {
        int pad = (7 - 1) * dilation / 2;
        var y = Snake(x, channels, t, w.Snake1Alpha, w.Snake1Beta);
        y = FullConv1d(y, channels, channels, t, w.Conv1Weight, w.Conv1Bias, kernel: 7, dilation: dilation, padding: pad);
        y = Snake(y, channels, t, w.Snake2Alpha, w.Snake2Beta);
        y = FullConv1d(y, channels, channels, t, w.Conv2Weight, w.Conv2Bias, kernel: 1, dilation: 1, padding: 0);

        var output = new float[y.Length];
        for (int i = 0; i < y.Length; i++) output[i] = x[i] + y[i]; // real OobleckResidualUnit: plain identity shortcut
        return output;
    }

    /// <summary>Time frames per im2col tile in <see cref="FullConv1dGemm"/>: a [tile, inCh*kernel] panel is 7 MB at
    /// 2048 channels x kernel 7 and 1.8 MB at 128 x 7.</summary>
    private const int GemmTimeTile = 128;

    /// <summary>
    /// Real FULL (non-depthwise) Conv1d, stride=1, symmetric ("same"-style) padding.
    /// <para>Large inputs take <see cref="FullConv1dGemm"/> (im2col tiles through the packed FP32 GEMM). The direct form
    /// below streams the whole time axis once per (output channel, input channel, tap), which at the decoder's last
    /// stage (128 channels, 480k samples for 10 s, kernel 7) is hundreds of GB of memory traffic and made VAE decode
    /// 88 % of an ACE-Step generation (2026-09-28).</para>
    /// </summary>
    public static unsafe float[] FullConv1d(float[] x, int inCh, int outCh, int t, float[] weight, float[]? bias, int kernel, int dilation, int padding)
    {
        if (PackedSgemmF32.IsSupported && t >= GemmTimeTile && outCh >= 16)
            return FullConv1dGemm(x, inCh, outCh, t, weight, bias, kernel, dilation, padding);

        var output = new float[outCh * t];
        fixed (float* px = x, pw = weight, pb = bias, po = output)
        {
            float* pxLocal = px;
            float* pwLocal = pw;
            float* pbLocal = pb;
            float* poLocal = po;

            Parallel.For(0, outCh, oc =>
            {
                float b = pbLocal != null ? pbLocal[oc] : 0f;
                float* outRow = poLocal + oc * t;
                if (b != 0f)
                {
                    for (int ti = 0; ti < t; ti++) outRow[ti] = b;
                }

                for (int ic = 0; ic < inCh; ic++)
                {
                    float* xRow = pxLocal + ic * t;
                    float* wRow = pwLocal + (oc * inCh + ic) * kernel;

                    for (int k = 0; k < kernel; k++)
                    {
                        float w = wRow[k];
                        if (w == 0f) continue;

                        int shift = k * dilation - padding;
                        int tStart = Math.Max(0, -shift);
                        int tEnd = Math.Min(t, t - shift);
                        if (tStart >= tEnd) continue;

                        int count = tEnd - tStart;
                        var inSpan = new ReadOnlySpan<float>(xRow + tStart + shift, count);
                        var outSpan = new Span<float>(outRow + tStart, count);
                        TensorPrimitives.MultiplyAdd(inSpan, w, outSpan, outSpan);
                    }
                }
            });
        }
        return output;
    }

    /// <summary>
    /// <see cref="FullConv1d"/> as GEMM: out[oc, t] = Σ_(ic,k) W[oc, ic·K + k] · x[ic, t + k·dilation − padding]. The conv
    /// weight's [outCh][inCh][K] layout is already the GEMM's row-major [n, k] Linear weight, packed once per call;
    /// time is processed in tiles whose im2col panel [rows, inCh·K] stays a few MB, and each tile's time-major result
    /// is transposed back into the channel-major output. Agrees with the direct form to FP32 rounding (different
    /// summation order), not bitwise.
    /// </summary>
    private static unsafe float[] FullConv1dGemm(float[] x, int inCh, int outCh, int t, float[] weight, float[]? bias, int kernel, int dilation, int padding)
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
    /// <see cref="ConvTranspose1d"/> as GEMM + col2im: Z[t, oc·K + k] = Σ_ic x[ic, t] · W[ic, oc, k] through the packed
    /// FP32 GEMM (weight transposed once to [outCh·K, inCh]), then out[oc, t·stride − padding + k] += Z. Time is tiled;
    /// tiles run in order, so the overlapping scatter-adds of neighbouring tiles never race, and within a tile each
    /// output channel is one worker. Agrees with the direct form to FP32 rounding (the direct form spent 7.2 s of an
    /// ACE-Step 10 s decode here, 2026-09-28).
    /// </summary>
    private static unsafe (float[] Data, int T) ConvTranspose1dGemm(float[] x, int inCh, int outCh, int t, float[] weight, float[] bias, int kernel, int stride, int padding)
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

    /// <summary>Real FULL Conv1d with stride &gt; 1 (used by the encoder's per-block downsample conv). Direct implementation without im2col buffer allocations.</summary>
    public static unsafe (float[] Data, int T) StridedConv1d(float[] x, int inCh, int outCh, int t, float[] weight, float[]? bias, int kernel, int stride, int padding)
    {
        int outT = (t + 2 * padding - (kernel - 1) - 1) / stride + 1;
        var output = new float[outCh * outT];
        fixed (float* px = x, pw = weight, pb = bias, po = output)
        {
            float* pxLocal = px;
            float* pwLocal = pw;
            float* pbLocal = pb;
            float* poLocal = po;

            Parallel.For(0, outCh, oc =>
            {
                float b = pbLocal != null ? pbLocal[oc] : 0f;
                float* outBase = poLocal + oc * outT;
                for (int ti = 0; ti < outT; ti++) outBase[ti] = b;

                for (int ic = 0; ic < inCh; ic++)
                {
                    float* xBase = pxLocal + ic * t;
                    float* wRow = pwLocal + (oc * inCh + ic) * kernel;

                    for (int k = 0; k < kernel; k++)
                    {
                        float w = wRow[k];
                        if (w == 0f) continue;

                        int kOffset = k - padding;
                        for (int ti = 0; ti < outT; ti++)
                        {
                            int src = ti * stride + kOffset;
                            if ((uint)src < (uint)t)
                            {
                                outBase[ti] += w * xBase[src];
                            }
                        }
                    }
                }
            });
        }
        return (output, outT);
    }

    /// <summary>Real ConvTranspose1d, weight layout `[inCh, outCh, kernel]` flat row-major. Large inputs take
    /// <see cref="ConvTranspose1dGemm"/>.</summary>
    public static unsafe (float[] Data, int T) ConvTranspose1d(float[] x, int inCh, int outCh, int t, float[] weight, float[] bias, int kernel, int stride, int padding)
    {
        if (PackedSgemmF32.IsSupported && t >= PackedSgemmF32.Mc)
            return ConvTranspose1dGemm(x, inCh, outCh, t, weight, bias, kernel, stride, padding);

        int outT = (t - 1) * stride - 2 * padding + kernel;
        var output = new float[outCh * outT];
        fixed (float* px = x, pw = weight, pb = bias, po = output)
        {
            float* pxLocal = px;
            float* pwLocal = pw;
            float* pbLocal = pb;
            float* poLocal = po;

            Parallel.For(0, outCh, oc =>
            {
                float b = pbLocal[oc];
                float* dstBase = poLocal + oc * outT;
                for (int ti = 0; ti < outT; ti++) dstBase[ti] = b;

                for (int ic = 0; ic < inCh; ic++)
                {
                    float* srcBase = pxLocal + ic * t;
                    float* wBase = pwLocal + (ic * outCh + oc) * kernel;
                    for (int ti = 0; ti < t; ti++)
                    {
                        float v = srcBase[ti];
                        if (v == 0f) continue;

                        int outStart = ti * stride - padding;
                        int kStart = outStart < 0 ? -outStart : 0;
                        int kEnd = outStart + kernel > outT ? outT - outStart : kernel;
                        if (kStart >= kEnd) continue;

                        int count = kEnd - kStart;
                        var wSpan = new ReadOnlySpan<float>(wBase + kStart, count);
                        var dstSpan = new Span<float>(dstBase + outStart + kStart, count);
                        TensorPrimitives.MultiplyAdd(wSpan, v, dstSpan, dstSpan);
                    }
                }
            });
        }
        return (output, outT);
    }
}

