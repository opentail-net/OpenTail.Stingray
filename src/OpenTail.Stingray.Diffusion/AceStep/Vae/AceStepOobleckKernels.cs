using System.Numerics.Tensors;

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

    /// <summary>Real FULL (non-depthwise) Conv1d, stride=1, symmetric ("same"-style) padding. Direct contiguous vectorized implementation without im2col buffer allocations.</summary>
    public static unsafe float[] FullConv1d(float[] x, int inCh, int outCh, int t, float[] weight, float[]? bias, int kernel, int dilation, int padding)
    {
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

    /// <summary>Real ConvTranspose1d, weight layout `[inCh, outCh, kernel]` flat row-major.</summary>
    public static unsafe (float[] Data, int T) ConvTranspose1d(float[] x, int inCh, int outCh, int t, float[] weight, float[] bias, int kernel, int stride, int padding)
    {
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

