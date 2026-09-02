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
        for (int c = 0; c < channels; c++)
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
        }
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

    /// <summary>Real FULL (non-depthwise) Conv1d, stride=1, symmetric ("same"-style) padding.</summary>
    public static unsafe float[] FullConv1d(float[] x, int inCh, int outCh, int t, float[] weight, float[]? bias, int kernel, int dilation, int padding)
    {
        int rowLen = inCh * kernel;
        var col = new float[t * rowLen];
        Parallel.For(0, t, ti =>
        {
            int rowBase = ti * rowLen;
            for (int ic = 0; ic < inCh; ic++)
            {
                int xBase = ic * t;
                int rBase = rowBase + ic * kernel;
                for (int k = 0; k < kernel; k++)
                {
                    int src = ti - padding + k * dilation;
                    col[rBase + k] = (uint)src < (uint)t ? x[xBase + src] : 0f;
                }
            }
        });

        var output = new float[outCh * t];
        fixed (float* colPtr = col, weightPtr = weight, outputPtr = output)
        {
            var colPtrLocal = colPtr;
            var weightPtrLocal = weightPtr;
            var outputPtrLocal = outputPtr;
            Parallel.For(0, outCh, oc =>
            {
                float b = bias?[oc] ?? 0f;
                float* wOc = weightPtrLocal + oc * rowLen;
                float* outBase = outputPtrLocal + oc * t;
                for (int ti = 0; ti < t; ti++)
                    outBase[ti] = b + SimdKernels.DotF32(wOc, colPtrLocal + ti * rowLen, rowLen);
            });
        }
        return output;
    }

    /// <summary>Real FULL Conv1d with stride &gt; 1 (used by the encoder's per-block downsample conv). Standard PyTorch semantics: `outT = (t + 2*padding - dilation*(kernel-1) - 1) / stride + 1`.</summary>
    public static unsafe (float[] Data, int T) StridedConv1d(float[] x, int inCh, int outCh, int t, float[] weight, float[]? bias, int kernel, int stride, int padding)
    {
        int outT = (t + 2 * padding - (kernel - 1) - 1) / stride + 1;
        int rowLen = inCh * kernel;
        var col = new float[outT * rowLen];
        Parallel.For(0, outT, ti =>
        {
            int rowBase = ti * rowLen;
            int inStart = ti * stride - padding;
            for (int ic = 0; ic < inCh; ic++)
            {
                int xBase = ic * t;
                int rBase = rowBase + ic * kernel;
                for (int k = 0; k < kernel; k++)
                {
                    int src = inStart + k;
                    col[rBase + k] = (uint)src < (uint)t ? x[xBase + src] : 0f;
                }
            }
        });

        var output = new float[outCh * outT];
        fixed (float* colPtr = col, weightPtr = weight, outputPtr = output)
        {
            var colPtrLocal = colPtr;
            var weightPtrLocal = weightPtr;
            var outputPtrLocal = outputPtr;
            Parallel.For(0, outCh, oc =>
            {
                float b = bias?[oc] ?? 0f;
                float* wOc = weightPtrLocal + oc * rowLen;
                float* outBase = outputPtrLocal + oc * outT;
                for (int ti = 0; ti < outT; ti++)
                    outBase[ti] = b + SimdKernels.DotF32(wOc, colPtrLocal + ti * rowLen, rowLen);
            });
        }
        return (output, outT);
    }

    /// <summary>Real ConvTranspose1d, weight layout `[inCh, outCh, kernel]` flat row-major.</summary>
    public static (float[] Data, int T) ConvTranspose1d(float[] x, int inCh, int outCh, int t, float[] weight, float[] bias, int kernel, int stride, int padding)
    {
        int outT = (t - 1) * stride - 2 * padding + kernel;
        var output = new float[outCh * outT];
        Parallel.For(0, outCh, oc =>
        {
            float b = bias[oc];
            int dstBase = oc * outT;
            for (int ti = 0; ti < outT; ti++) output[dstBase + ti] = b;

            for (int ic = 0; ic < inCh; ic++)
            {
                int srcBase = ic * t;
                int wBase = (ic * outCh + oc) * kernel;
                for (int ti = 0; ti < t; ti++)
                {
                    float v = x[srcBase + ti];
                    int outStart = ti * stride - padding;

                    int kStart = outStart < 0 ? -outStart : 0;
                    int kEnd = outStart + kernel > outT ? outT - outStart : kernel;
                    if (kStart >= kEnd) continue;

                    var wSpan = weight.AsSpan(wBase + kStart, kEnd - kStart);
                    var dstSpan = output.AsSpan(dstBase + outStart + kStart, kEnd - kStart);
                    TensorPrimitives.MultiplyAdd(wSpan, v, dstSpan, dstSpan);
                }
            }
        });
        return (output, outT);
    }
}
