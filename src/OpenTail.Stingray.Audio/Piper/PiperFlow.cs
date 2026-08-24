using System;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Piper;

/// <summary>
/// Piper's VITS ResidualCouplingBlock (flow), inference/reverse path only. Ported from the
/// reference VITS `models.py ResidualCouplingBlock.forward(reverse=True)`: 4 ResidualCouplingLayer
/// flows interleaved with 4 parameter-free Flip flows, each layer's WN (WaveNet-style dilated conv
/// stack) with NO external conditioning (gin_channels=0 for this checkpoint -- confirmed via node
/// inspection, no `g` input feeds `flow.flows.*.enc`).
///
/// Confirmed against the real ONNX graph: n_layers=4, kernel_size=5, dilation=1 for ALL 4 WN
/// layers (constant, not vanilla-VITS's exponential 1,3,9,27 default -- a Piper-specific config),
/// hidden_channels=192, half_channels=96, mean_only=True (post.weight has 96 output channels, not
/// 192 -- logs is always implicitly 0, so the reverse affine transform is just `x1 - m`, no `exp`).
///
/// Reverse execution order (list-reversed, matching `reversed(self.flows)` in the reference):
/// Flip(7) -&gt; Layer(6) -&gt; Flip(5) -&gt; Layer(4) -&gt; Flip(3) -&gt; Layer(2) -&gt; Flip(1) -&gt; Layer(0).
/// </summary>
public static class PiperFlow
{
    /// <summary>
    /// zp is the length-regulator-expanded latent, channel-first [hiddenDim, T] (hiddenDim=192).
    /// Returns the flow's output (z, still channel-first [192, T]) after the reverse transform.
    /// </summary>
    public static float[] Reverse(PiperOnnxWeights w, float[] zp, int t)
    {
        int dim = w.HiddenDim;
        var z = zp;

        z = Flip(z, dim, t);
        z = CouplingLayerReverse(z, t, w.FlowLayer6);
        z = Flip(z, dim, t);
        z = CouplingLayerReverse(z, t, w.FlowLayer4);
        z = Flip(z, dim, t);
        z = CouplingLayerReverse(z, t, w.FlowLayer2);
        z = Flip(z, dim, t);
        z = CouplingLayerReverse(z, t, w.FlowLayer0);

        return z;
    }

    private static float[] Flip(float[] z, int dim, int t)
    {
        // torch.flip(x, [1]): reverse channel order over the full hidden dim (not a 2-channel
        // swap like the SDP's Flip -- ResidualCouplingBlock's Flip operates on all `dim` channels).
        var output = new float[dim * t];
        for (int c = 0; c < dim; c++)
            Array.Copy(z, (dim - 1 - c) * t, output, c * t, t);
        return output;
    }

    private static float[] CouplingLayerReverse(float[] z, int t, PiperCouplingLayerWeights lw)
    {
        int half = PiperOnnxWeights.FlowHalfChannels;

        var x0 = new float[half * t];
        var x1 = new float[half * t];
        Array.Copy(z, 0, x0, 0, half * t);
        Array.Copy(z, half * t, x1, 0, half * t);

        // h = pre(x0), channel-first [half, T] -> [hidden, T].
        var h = Conv1x1(x0, half, t, lw.PreWeight, lw.PreBias, PiperOnnxWeights.FlowHalfChannels * 2);

        h = WnForward(h, t, lw.Enc);

        // m = post(h): mean_only, so post outputs [half, T] directly (no logs channel).
        var m = Conv1x1(h, PiperOnnxWeights.FlowHalfChannels * 2, t, lw.PostWeight, lw.PostBias, half);

        // reverse: x1 = (x1 - m) * exp(-logs), logs implicitly 0 (mean_only) -> x1 - m.
        var x1Out = new float[half * t];
        for (int i = 0; i < x1Out.Length; i++) x1Out[i] = x1[i] - m[i];

        var output = new float[2 * half * t];
        Array.Copy(x0, 0, output, 0, half * t);
        Array.Copy(x1Out, 0, output, half * t, half * t);
        return output;
    }

    /// <summary>
    /// WN (WaveNet-style dilated-conv stack), gin_channels=0 (no conditioning). 4 layers, kernel=5,
    /// dilation=1 constant. Gated activation (tanh*sigmoid) with residual+skip connections; final
    /// layer contributes only to skip (matches reference's `if i &lt; n_layers-1` residual guard).
    /// </summary>
    private static float[] WnForward(float[] x, int t, PiperWnWeights wn)
    {
        int hidden = PiperOnnxWeights.FlowHalfChannels * 2; // 192
        const int kernel = PiperOnnxWeights.FlowWnKernel;
        const int dilation = PiperOnnxWeights.FlowWnDilation;
        int pad = (kernel * dilation - dilation) / 2;

        var output = new float[hidden * t];
        var cur = x;

        for (int layer = 0; layer < PiperOnnxWeights.FlowWnLayers; layer++)
        {
            // in_layer: dilated conv, hidden -> 2*hidden (gate+filter).
            var xIn = DilatedConv(cur, hidden, t, wn.InWeight[layer], wn.InBias[layer], 2 * hidden, kernel, dilation, pad);

            var acts = new float[hidden * t];
            for (int c = 0; c < hidden; c++)
            {
                int filterBase = c * t;
                int gateBase = (hidden + c) * t;
                for (int ti = 0; ti < t; ti++)
                {
                    float filterVal = MathF.Tanh(xIn[filterBase + ti]);
                    float gateVal = 1f / (1f + MathF.Exp(-xIn[gateBase + ti]));
                    acts[filterBase + ti] = filterVal * gateVal;
                }
            }

            // res_skip_layer: 1x1 conv, hidden -> (last layer: hidden [skip only]; else: 2*hidden [res+skip]).
            bool isLast = layer == PiperOnnxWeights.FlowWnLayers - 1;
            int resSkipOutCh = isLast ? hidden : 2 * hidden;
            var resSkip = Conv1x1(acts, hidden, t, wn.ResSkipWeight[layer], wn.ResSkipBias[layer], resSkipOutCh);

            if (!isLast)
            {
                var next = new float[hidden * t];
                for (int i = 0; i < hidden * t; i++) next[i] = cur[i] + resSkip[i];
                for (int i = 0; i < hidden * t; i++) output[i] += resSkip[hidden * t + i];
                cur = next;
            }
            else
            {
                for (int i = 0; i < hidden * t; i++) output[i] += resSkip[i];
            }
        }

        return output;
    }

    /// <summary>im2col + GEMM (see <c>FishSpeechCodec.FullConv1d</c>'s doc comment for the technique): the gather is independent of `oc`, hoisted out, and each output channel reduces to one AVX2/FMA <see cref="SimdKernels.DotF32"/> call per timestep.</summary>
    private static unsafe float[] DilatedConv(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh, int kernel, int dilation, int pad)
    {
        int rowLen = inCh * kernel;
        var col = new float[t * rowLen];
        System.Threading.Tasks.Parallel.For(0, t, ti =>
        {
            int rowBase = ti * rowLen;
            for (int ic = 0; ic < inCh; ic++)
            {
                int xBase = ic * t;
                int rBase = rowBase + ic * kernel;
                for (int k = 0; k < kernel; k++)
                {
                    int src = ti - pad + k * dilation;
                    col[rBase + k] = (uint)src < (uint)t ? input[xBase + src] : 0f;
                }
            }
        });

        var output = new float[outCh * t];
        fixed (float* colPtr = col, weightPtr = weight, outputPtr = output)
        {
            var colLocal = colPtr;
            var weightLocal = weightPtr;
            var outputLocal = outputPtr;
            System.Threading.Tasks.Parallel.For(0, outCh, oc =>
            {
                float b = bias[oc];
                float* wOc = weightLocal + oc * rowLen;
                float* outBase = outputLocal + oc * t;
                for (int ti = 0; ti < t; ti++)
                    outBase[ti] = b + SimdKernels.DotF32(wOc, colLocal + ti * rowLen, rowLen);
            });
        }
        return output;
    }

    /// <summary>Per-timestep 1x1 conv (a Linear applied independently at every timestep), input/output
    /// channel-first -- gathers each timestep's channel column then uses the real SIMD matvec kernel
    /// (same pattern as PiperTextEncoder.ConvOverTime1x1) instead of a scalar strided dot product.</summary>
    private static unsafe float[] Conv1x1(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh)
    {
        var output = new float[outCh * t];
        var col = new float[inCh];
        var outCol = new float[outCh];
        for (int ti = 0; ti < t; ti++)
        {
            for (int c = 0; c < inCh; c++) col[c] = input[c * t + ti];
            fixed (float* w = weight, x = col, y = outCol)
            {
                SimdKernels.MatVecF32(y, w, x, outCh, inCh);
            }
            for (int c = 0; c < outCh; c++) output[c * t + ti] = outCol[c] + bias[c];
        }
        return output;
    }
}
