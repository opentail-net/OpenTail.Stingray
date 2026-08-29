
namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// Shared math kernels for the VITS/VITS2 family's StochasticDurationPredictor (SDP):
/// DDSConv (Dilated and Depth-Separable Convolution), ConvFlow (piecewise rational-quadratic
/// spline normalizing flow, Durkan et al. 2019), ElementwiseAffine, and the channel-flip used
/// between flow blocks. Extracted from the original Piper-only implementation
/// (`Piper.PiperDurationPredictor`) so later VITS-family ports (MeloTTS's `sdp`) reuse the same
/// math instead of a second hand-rolled copy -- see <see cref="VitsAttentionKernels"/> for the
/// equivalent extraction of the TextEncoder's relative-attention math.
/// </summary>
public static class VitsDurationFlowKernels
{
    public const int NumBins = 10;
    private const float TailBound = 5.0f;
    private const float MinBinWidth = 1e-3f;
    private const float MinBinHeight = 1e-3f;
    private const float MinDerivative = 1e-3f;

    /// <summary>torch.flip(x, [1]) for a 2-channel [2, t] tensor: swap the two channels.</summary>
    public static float[] Flip(float[] z, int t)
    {
        var output = new float[2 * t];
        Array.Copy(z, t, output, 0, t);
        Array.Copy(z, 0, output, t, t);
        return output;
    }

    /// <summary>ElementwiseAffine reverse: x = (x - m) * exp(-logs); m/expNegLogs are per-channel (2 channels), broadcast over T.</summary>
    public static float[] ElementwiseAffineReverse(float[] z, int t, float[] m, float[] expNegLogs)
    {
        var output = new float[2 * t];
        for (int c = 0; c < 2; c++)
            for (int ti = 0; ti < t; ti++)
                output[c * t + ti] = (z[c * t + ti] - m[c]) * expNegLogs[c];
        return output;
    }

    /// <summary>
    /// ConvFlow.forward(reverse=True): x0 (channel 0, unchanged) conditions a spline transform
    /// applied to x1 (channel 1). half_channels=1 (in_channels=2 for the duration flow, standard
    /// across the whole VITS family).
    /// </summary>
    public static float[] ConvFlowReverse(float[] z, int t, float[] context, int contextDim, VitsConvFlowWeights fw)
    {
        var x0 = new float[t];
        var x1 = new float[t];
        for (int ti = 0; ti < t; ti++) { x0[ti] = z[ti]; x1[ti] = z[t + ti]; }

        var h = VitsAttentionKernels.Conv1x1(x0, 1, t, fw.PreWeight, fw.PreBias, contextDim);
        h = DDSConv(h, contextDim, t, fw.Convs, context);
        h = VitsAttentionKernels.Conv1x1(h, contextDim, t, fw.ProjWeight, fw.ProjBias, NumBins * 3 - 1); // [29, T] channel-first

        var x1Out = new float[t];
        float invSqrtFilter = 1f / MathF.Sqrt(contextDim);
        for (int ti = 0; ti < t; ti++)
        {
            var widths = new float[NumBins];
            var heights = new float[NumBins];
            var derivatives = new float[NumBins + 1];
            for (int b = 0; b < NumBins; b++)
            {
                widths[b] = h[b * t + ti] * invSqrtFilter;
                heights[b] = h[(NumBins + b) * t + ti] * invSqrtFilter;
            }
            for (int b = 0; b < NumBins - 1; b++)
                derivatives[b + 1] = h[(2 * NumBins + b) * t + ti];
            float boundaryConst = MathF.Log(MathF.Exp(1f - MinDerivative) - 1f);
            derivatives[0] = boundaryConst;
            derivatives[NumBins] = boundaryConst;

            x1Out[ti] = RationalQuadraticSplineInverse(x1[ti], widths, heights, derivatives);
        }

        var output = new float[2 * t];
        for (int ti = 0; ti < t; ti++) { output[ti] = x0[ti]; output[t + ti] = x1Out[ti]; }
        return output;
    }

    /// <summary>
    /// Piecewise rational-quadratic spline, inverse direction, with linear tails (Durkan et al.
    /// 2019, as used by VITS's ConvFlow). Values outside [-tailBound, tailBound] pass through unchanged.
    /// </summary>
    public static float RationalQuadraticSplineInverse(float input, float[] unnormalizedWidths, float[] unnormalizedHeights, float[] unnormalizedDerivatives)
    {
        if (input < -TailBound || input > TailBound) return input;

        int numBins = unnormalizedWidths.Length;

        var widths = Softmax(unnormalizedWidths);
        for (int i = 0; i < numBins; i++) widths[i] = MinBinWidth + (1f - MinBinWidth * numBins) * widths[i];
        var cumwidths = new float[numBins + 1];
        for (int i = 0; i < numBins; i++) cumwidths[i + 1] = cumwidths[i] + widths[i];
        for (int i = 0; i <= numBins; i++) cumwidths[i] = (TailBound - -TailBound) * cumwidths[i] + -TailBound;
        cumwidths[0] = -TailBound;
        cumwidths[numBins] = TailBound;
        for (int i = 0; i < numBins; i++) widths[i] = cumwidths[i + 1] - cumwidths[i];

        var derivatives = new float[numBins + 1];
        for (int i = 0; i <= numBins; i++) derivatives[i] = MinDerivative + Softplus(unnormalizedDerivatives[i]);

        var heights = Softmax(unnormalizedHeights);
        for (int i = 0; i < numBins; i++) heights[i] = MinBinHeight + (1f - MinBinHeight * numBins) * heights[i];
        var cumheights = new float[numBins + 1];
        for (int i = 0; i < numBins; i++) cumheights[i + 1] = cumheights[i] + heights[i];
        for (int i = 0; i <= numBins; i++) cumheights[i] = (TailBound - -TailBound) * cumheights[i] + -TailBound;
        cumheights[0] = -TailBound;
        cumheights[numBins] = TailBound;
        for (int i = 0; i < numBins; i++) heights[i] = cumheights[i + 1] - cumheights[i];

        int binIdx = SearchSorted(cumheights, input);

        float inputCumwidths = cumwidths[binIdx];
        float inputBinWidths = widths[binIdx];
        float inputCumheights = cumheights[binIdx];
        float inputHeights = heights[binIdx];
        float delta = heights[binIdx] / widths[binIdx];
        float inputDelta = delta;
        float inputDerivatives = derivatives[binIdx];
        float inputDerivativesPlusOne = derivatives[binIdx + 1];

        float a = (input - inputCumheights) * (inputDerivatives + inputDerivativesPlusOne - 2f * inputDelta) + inputHeights * (inputDelta - inputDerivatives);
        float b = inputHeights * inputDerivatives - (input - inputCumheights) * (inputDerivatives + inputDerivativesPlusOne - 2f * inputDelta);
        float c = -inputDelta * (input - inputCumheights);

        float discriminant = b * b - 4f * a * c;
        if (discriminant < 0f) discriminant = 0f;
        float root = (2f * c) / (-b - MathF.Sqrt(discriminant));

        return root * inputBinWidths + inputCumwidths;
    }

    private static int SearchSorted(float[] bin, float value)
    {
        int n = bin.Length;
        int count = 0;
        for (int i = 0; i < n; i++)
        {
            float edge = (i == n - 1) ? bin[i] + 1e-6f : bin[i];
            if (value >= edge) count++;
        }
        int idx = count - 1;
        return Math.Clamp(idx, 0, n - 2);
    }

    public static float[] Softmax(float[] x)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < x.Length; i++) if (x[i] > max) max = x[i];
        var output = new float[x.Length];
        float sum = 0f;
        for (int i = 0; i < x.Length; i++)
        {
            float e = MathF.Exp(x[i] - max);
            output[i] = e;
            sum += e;
        }
        float inv = 1f / sum;
        for (int i = 0; i < output.Length; i++) output[i] *= inv;
        return output;
    }

    public static float Softplus(float x) => MathF.Log(1f + MathF.Exp(x));

    /// <summary>
    /// DDSConv (Dilated and Depth-Separable Convolution), 3 layers, dilation = kernel_size^i. If
    /// context is non-null, it's added to x before the layers (matches ConvFlow's `g=context` path).
    /// </summary>
    public static float[] DDSConv(float[] x, int ch, int t, VitsDdsConvWeights dds, float[]? context = null)
    {
        if (context != null)
        {
            var withG = new float[ch * t];
            for (int i = 0; i < withG.Length; i++) withG[i] = x[i] + context[i];
            x = withG;
        }

        for (int layer = 0; layer < dds.NumLayers; layer++)
        {
            int dilation = (int)Math.Pow(3, layer); // kernel_size=3 per this DDSConv's config
            var y = DepthwiseConvDilated(x, ch, t, dds.ConvsSepWeight[layer], dds.ConvsSepBias[layer], dilation);
            y = VitsAttentionKernels.LayerNormChannelFirst(y, ch, t, dds.Norms1Gamma[layer], dds.Norms1Beta[layer]);
            for (int i = 0; i < y.Length; i++) y[i] = Gelu(y[i]);
            y = VitsAttentionKernels.Conv1x1(y, ch, t, dds.Convs1x1Weight[layer], dds.Convs1x1Bias[layer], ch);
            y = VitsAttentionKernels.LayerNormChannelFirst(y, ch, t, dds.Norms2Gamma[layer], dds.Norms2Beta[layer]);
            for (int i = 0; i < y.Length; i++) y[i] = Gelu(y[i]);

            var next = new float[ch * t];
            for (int i = 0; i < next.Length; i++) next[i] = x[i] + y[i];
            x = next;
        }
        return x;
    }

    public static float[] DepthwiseConvDilated(float[] input, int ch, int t, float[] weight, float[] bias, int dilation)
    {
        const int kernel = 3;
        int pad = (kernel * dilation - dilation) / 2;
        var output = new float[ch * t];
        Parallel.For(0, ch, c =>
        {
            float b = bias[c];
            int wBase = c * kernel; // depthwise: weight shape [ch,1,kernel]
            int srcBase = c * t;
            for (int ti = 0; ti < t; ti++)
            {
                float sum = b;
                for (int k = 0; k < kernel; k++)
                {
                    int src = ti - pad + k * dilation;
                    if ((uint)src < (uint)t) sum += weight[wBase + k] * input[srcBase + src];
                }
                output[srcBase + ti] = sum;
            }
        });
        return output;
    }

    public static float Gelu(float x) => 0.5f * x * (1f + Erf(x / 1.4142135f));

    // Abramowitz-Stegun erf approximation (max error ~1.5e-7), sufficient for float32 GELU.
    public static float Erf(float x)
    {
        float sign = x < 0 ? -1f : 1f;
        x = MathF.Abs(x);
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float tt = 1f / (1f + p * x);
        float y = 1f - (((((a5 * tt + a4) * tt) + a3) * tt + a2) * tt + a1) * tt * MathF.Exp(-x * x);
        return sign * y;
    }
}

/// <summary>DDSConv weights: 3 layers, kernel=3, dilation=3^i. Constructed via a caller-supplied
/// tensor getter so both Piper and MeloTTS (different ONNX weight-resolution strategies) can share it.</summary>
public sealed class VitsDdsConvWeights
{
    public int NumLayers { get; } = 3;
    public float[][] ConvsSepWeight { get; } = new float[3][];
    public float[][] ConvsSepBias { get; } = new float[3][];
    public float[][] Convs1x1Weight { get; } = new float[3][];
    public float[][] Convs1x1Bias { get; } = new float[3][];
    public float[][] Norms1Gamma { get; } = new float[3][];
    public float[][] Norms1Beta { get; } = new float[3][];
    public float[][] Norms2Gamma { get; } = new float[3][];
    public float[][] Norms2Beta { get; } = new float[3][];

    public VitsDdsConvWeights(Func<string, float[]> getFloat, string prefix)
    {
        for (int i = 0; i < NumLayers; i++)
        {
            ConvsSepWeight[i] = getFloat($"{prefix}.convs_sep.{i}.weight");
            ConvsSepBias[i] = getFloat($"{prefix}.convs_sep.{i}.bias");
            Convs1x1Weight[i] = getFloat($"{prefix}.convs_1x1.{i}.weight");
            Convs1x1Bias[i] = getFloat($"{prefix}.convs_1x1.{i}.bias");
            Norms1Gamma[i] = getFloat($"{prefix}.norms_1.{i}.gamma");
            Norms1Beta[i] = getFloat($"{prefix}.norms_1.{i}.beta");
            Norms2Gamma[i] = getFloat($"{prefix}.norms_2.{i}.gamma");
            Norms2Beta[i] = getFloat($"{prefix}.norms_2.{i}.beta");
        }
    }
}

/// <summary>ConvFlow: pre (1x1, half_channels->filter_channels) -> DDSConv -> proj (1x1, filter_channels->half_channels*(num_bins*3-1)).</summary>
public sealed class VitsConvFlowWeights
{
    public float[] PreWeight { get; }
    public float[] PreBias { get; }
    public VitsDdsConvWeights Convs { get; }
    public float[] ProjWeight { get; }
    public float[] ProjBias { get; }

    public VitsConvFlowWeights(Func<string, float[]> getFloat, string prefix)
    {
        PreWeight = getFloat($"{prefix}.pre.weight");
        PreBias = getFloat($"{prefix}.pre.bias");
        Convs = new VitsDdsConvWeights(getFloat, $"{prefix}.convs");
        ProjWeight = getFloat($"{prefix}.proj.weight");
        ProjBias = getFloat($"{prefix}.proj.bias");
    }
}
