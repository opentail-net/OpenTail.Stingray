using System;

namespace OpenTail.Stingray.Audio.Piper;

/// <summary>
/// Piper's VITS StochasticDurationPredictor (dp), inference/reverse path only. Ported from the
/// reference VITS `models.py StochasticDurationPredictor.forward(reverse=True)`: a normalizing
/// flow over a 2-channel noise sample, conditioned on the TextEncoder's pre-proj hidden state.
///
/// Confirmed against the real ONNX graph (not just recollection of the reference): the exported
/// inference graph only exercises 8 of the 9 flow-list entries -- ConvFlow at list-index 1 is
/// fully pruned (no nodes/initializers reference `dp.flows.1.*`), matching the reference's
/// `flows = list(reversed(self.flows)); flows = flows[:-2] + [flows[-1]]` "remove a useless
/// vflow" trick. The real execution order, confirmed via each stage's Split node's input chain,
/// is: Flip(idx8) -&gt; ConvFlow(idx7) -&gt; Flip(idx6) -&gt; ConvFlow(idx5) -&gt; Flip(idx4) -&gt;
/// ConvFlow(idx3) -&gt; Flip(idx2) -&gt; ElementwiseAffine(idx0).
/// </summary>
public static class PiperDurationPredictor
{
    private const int NumBins = 10;
    private const float TailBound = 5.0f;
    private const float MinBinWidth = 1e-3f;
    private const float MinBinHeight = 1e-3f;
    private const float MinDerivative = 1e-3f;

    /// <summary>
    /// encoderHidden is the TextEncoder's pre-proj output, channel-first [hidden, T]. Returns
    /// logw (pre-exp predicted log-duration), length T. noise must be a [2*T] array of raw
    /// N(0,1) draws (channel-first [2,T]) -- caller supplies these (real RNG in production,
    /// golden-captured values for verification).
    /// </summary>
    public static float[] Predict(PiperOnnxWeights w, float[] encoderHidden, int t, float[] noise, float noiseScaleW)
    {
        int dim = w.HiddenDim;

        // x = proj(convs(pre(encoderHidden))) -- the shared conditioning context `g` for every flow.
        var x = Conv1x1(encoderHidden, dim, t, w.DpPreWeight, w.DpPreBias, dim);
        x = DDSConv(x, dim, t, w.DpConvs);
        x = Conv1x1(x, dim, t, w.DpProjWeight, w.DpProjBias, dim);

        // z = noise * noise_w, channel-first [2, T].
        var z = new float[2 * t];
        for (int i = 0; i < z.Length; i++) z[i] = noise[i] * noiseScaleW;

        // Flip(8) -> ConvFlow(7) -> Flip(6) -> ConvFlow(5) -> Flip(4) -> ConvFlow(3) -> Flip(2) -> ElementwiseAffine(0)
        z = Flip(z, t);
        z = ConvFlowReverse(z, t, x, dim, w.DpFlow7);
        z = Flip(z, t);
        z = ConvFlowReverse(z, t, x, dim, w.DpFlow5);
        z = Flip(z, t);
        z = ConvFlowReverse(z, t, x, dim, w.DpFlow3);
        z = Flip(z, t);
        z = ElementwiseAffineReverse(z, t, w.DpFlow0M, w.DpFlow0ExpNegLogs);

        // logw = z[0, :] (z0); z[1, :] is discarded (matches torch.split(z, [1,1], 1)).
        var logw = new float[t];
        Array.Copy(z, 0, logw, 0, t);
        return logw;
    }

    private static float[] Flip(float[] z, int t)
    {
        // torch.flip(x, [1]): swap the 2 channels.
        var output = new float[2 * t];
        Array.Copy(z, t, output, 0, t);
        Array.Copy(z, 0, output, t, t);
        return output;
    }

    private static float[] ElementwiseAffineReverse(float[] z, int t, float[] m, float[] expNegLogs)
    {
        // x = (x - m) * exp(-logs); m/expNegLogs are per-channel (2 channels), broadcast over T.
        var output = new float[2 * t];
        for (int c = 0; c < 2; c++)
            for (int ti = 0; ti < t; ti++)
                output[c * t + ti] = (z[c * t + ti] - m[c]) * expNegLogs[c];
        return output;
    }

    /// <summary>
    /// ConvFlow.forward(reverse=True): x0 (channel 0, unchanged) conditions a spline transform
    /// applied to x1 (channel 1). half_channels=1 here (in_channels=2 for the duration flow).
    /// </summary>
    private static float[] ConvFlowReverse(float[] z, int t, float[] context, int contextDim, PiperConvFlowWeights fw)
    {
        var x0 = new float[t];
        var x1 = new float[t];
        for (int ti = 0; ti < t; ti++) { x0[ti] = z[ti]; x1[ti] = z[t + ti]; }

        // h = proj(convs(pre(x0))) -- note: convs (DDSConv) here also adds `g=context` before its layers.
        var h = Conv1x1(x0, 1, t, fw.PreWeight, fw.PreBias, contextDim);
        h = DDSConv(h, contextDim, t, fw.Convs, context);
        h = Conv1x1(h, contextDim, t, fw.ProjWeight, fw.ProjBias, NumBins * 3 - 1); // [29, T] channel-first

        var x1Out = new float[t];
        float invSqrtFilter = 1f / MathF.Sqrt(contextDim);
        for (int ti = 0; ti < t; ti++)
        {
            var widths = new float[NumBins];
            var heights = new float[NumBins];
            var derivatives = new float[NumBins + 1]; // padded, matches F.pad(unnormalized_derivatives, (1,1))
            for (int b = 0; b < NumBins; b++)
            {
                widths[b] = h[b * t + ti] * invSqrtFilter;
                heights[b] = h[(NumBins + b) * t + ti] * invSqrtFilter;
            }
            for (int b = 0; b < NumBins - 1; b++)
                derivatives[b + 1] = h[(2 * NumBins + b) * t + ti];
            // Reference pads with a constant so the boundary derivative equals 1 after softplus:
            // log(exp(1 - min_derivative) - 1).
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
    /// 2019, as used by VITS's ConvFlow). Values outside [-tailBound, tailBound] pass through
    /// unchanged (the reference's `tails='linear'` behavior).
    /// </summary>
    private static float RationalQuadraticSplineInverse(float input, float[] unnormalizedWidths, float[] unnormalizedHeights, float[] unnormalizedDerivatives)
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

        // inverse=True: search in cumheights (searchsorted with the last bin edge nudged by eps).
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
        if (discriminant < 0f) discriminant = 0f; // numerical safety; reference asserts >=0
        float root = (2f * c) / (-b - MathF.Sqrt(discriminant));

        return root * inputBinWidths + inputCumwidths;
    }

    private static int SearchSorted(float[] bin, float value)
    {
        // torch.sum(inputs[...,None] >= bin_locations, -1) - 1, with the last edge nudged by eps
        // so a value exactly at the outer boundary still lands in the last bin.
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

    private static float[] Softmax(float[] x)
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

    private static float Softplus(float x) => MathF.Log(1f + MathF.Exp(x));

    /// <summary>
    /// DDSConv (Dilated and Depth-Separable Convolution), 3 layers, dilation = kernel_size^i.
    /// If context is non-null, it's added to x before the layers (matches ConvFlow's `g=context`
    /// path; the shared dp.convs call has no g).
    /// </summary>
    private static float[] DDSConv(float[] x, int ch, int t, PiperDdsConvWeights dds, float[]? context = null)
    {
        if (context != null)
        {
            var withG = new float[ch * t];
            for (int i = 0; i < withG.Length; i++) withG[i] = x[i] + context[i];
            x = withG;
        }

        for (int layer = 0; layer < dds.NumLayers; layer++)
        {
            int dilation = (int)Math.Pow(3, layer); // kernel_size=3 per Piper's DDSConv config
            var y = DepthwiseConvDilated(x, ch, t, dds.ConvsSepWeight[layer], dds.ConvsSepBias[layer], dilation);
            y = LayerNormChannelFirst(y, ch, t, dds.Norms1Gamma[layer], dds.Norms1Beta[layer]);
            for (int i = 0; i < y.Length; i++) y[i] = Gelu(y[i]);
            y = Conv1x1(y, ch, t, dds.Convs1x1Weight[layer], dds.Convs1x1Bias[layer], ch);
            y = LayerNormChannelFirst(y, ch, t, dds.Norms2Gamma[layer], dds.Norms2Beta[layer]);
            for (int i = 0; i < y.Length; i++) y[i] = Gelu(y[i]);

            var next = new float[ch * t];
            for (int i = 0; i < next.Length; i++) next[i] = x[i] + y[i];
            x = next;
        }
        return x;
    }

    private static float[] DepthwiseConvDilated(float[] input, int ch, int t, float[] weight, float[] bias, int dilation)
    {
        const int kernel = 3;
        int pad = (kernel * dilation - dilation) / 2;
        var output = new float[ch * t];
        System.Threading.Tasks.Parallel.For(0, ch, c =>
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

    private static float Gelu(float x) => 0.5f * x * (1f + Erf(x / 1.4142135f));

    // Abramowitz-Stegun erf approximation (max error ~1.5e-7), sufficient for float32 GELU.
    private static float Erf(float x)
    {
        float sign = x < 0 ? -1f : 1f;
        x = MathF.Abs(x);
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float tt = 1f / (1f + p * x);
        float y = 1f - (((((a5 * tt + a4) * tt) + a3) * tt + a2) * tt + a1) * tt * MathF.Exp(-x * x);
        return sign * y;
    }

    private static float[] Conv1x1(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh)
    {
        var output = new float[outCh * t];
        System.Threading.Tasks.Parallel.For(0, outCh, oc =>
        {
            float b = bias[oc];
            int wBase = oc * inCh;
            for (int ti = 0; ti < t; ti++)
            {
                float sum = b;
                for (int ic = 0; ic < inCh; ic++) sum += weight[wBase + ic] * input[ic * t + ti];
                output[oc * t + ti] = sum;
            }
        });
        return output;
    }

    private static float[] LayerNormChannelFirst(float[] x, int ch, int t, float[] gamma, float[] beta, float eps = 1e-5f)
    {
        var output = new float[ch * t];
        for (int ti = 0; ti < t; ti++)
        {
            double mean = 0;
            for (int c = 0; c < ch; c++) mean += x[c * t + ti];
            mean /= ch;
            double var = 0;
            for (int c = 0; c < ch; c++) { double d = x[c * t + ti] - mean; var += d * d; }
            var /= ch;
            float invStd = (float)(1.0 / Math.Sqrt(var + eps));
            for (int c = 0; c < ch; c++)
                output[c * t + ti] = (float)((x[c * t + ti] - mean) * invStd) * gamma[c] + beta[c];
        }
        return output;
    }
}
