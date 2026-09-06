
namespace OpenTail.Stingray.Audio.Rvc;

/// <summary>
/// Real HuBERT-base (Wav2Vec2-family) content encoder forward pass for RVC, transcribed directly
/// from `examples/audio.cpp/src/framework/modules/speech_encoders/hubert_encoder.cpp`'s real
/// `build_hubert_graph` (not guessed): 7-stage valid (no-padding) conv1d feature extractor with
/// GroupNorm only after layer 0 (`FirstLayerGroupNorm`) and exact-erf GELU after every layer ->
/// LayerNorm + Linear feature projection (512-&gt;768) -> real weight-normalized grouped
/// positional conv (16 groups, kernel 128, added as a residual, NOT gated) -> encoder input
/// LayerNorm -> 12 standard post-LayerNorm (`attn -&gt; +residual -&gt; self_attn_layer_norm -&gt;
/// ffn -&gt; +residual -&gt; final_layer_norm`) transformer blocks, non-causal (bidirectional)
/// self-attention.
/// </summary>
public static class RvcHubertEncoder
{
    /// <summary>
    /// Runs the encoder on a 16kHz mono waveform. Returns frame-major [numFrames, 768] hidden
    /// states after all 12 layers (the real reference's `output_hidden_layer = 12` case, i.e.
    /// `v1_features=false` -- RVC v2's content features).
    /// </summary>
    public static float[][] Forward(RvcHubertWeights w, ReadOnlySpan<float> waveform16k)
    {
        // 1. Feature extractor: 7 valid (no-padding) conv1d layers, GroupNorm only after layer 0,
        // exact-erf GELU after every layer.
        float[][] x = ToFrames1Channel(waveform16k);
        int channels = 1;
        for (int layerIdx = 0; layerIdx < 7; layerIdx++)
        {
            x = Conv1dValid(x, channels, w.ConvWeights[layerIdx], bias: null, outCh: RvcHubertWeights.ConvDim[layerIdx], kernel: RvcHubertWeights.ConvKernel[layerIdx], stride: RvcHubertWeights.ConvStride[layerIdx]);
            channels = RvcHubertWeights.ConvDim[layerIdx];
            if (layerIdx == 0)
                x = GroupNorm(x, numGroups: channels, w.ConvLayer0GroupNormWeight, w.ConvLayer0GroupNormBias);
            GeluErfInPlaceRows(x);
        }

        // 2. Feature projection: LayerNorm(512) + Linear(512->768).
        int t = x.Length;
        for (int i = 0; i < t; i++)
            x[i] = LayerNorm(x[i], w.LayerNormWeight, w.LayerNormBias);
        var hidden = new float[t][];
        for (int i = 0; i < t; i++)
            hidden[i] = LinearBias(x[i], w.PostExtractProjWeight, w.PostExtractProjBias, inDim: 512, outDim: RvcHubertWeights.HiddenDim);

        // 3. Positional conv: grouped conv1d (16 groups, kernel 128, pad=64 each side, output
        // trimmed by 1 trailing frame since kernel is even), exact-erf GELU, added as a residual.
        var pos = GroupedConv1dSamePad(hidden, RvcHubertWeights.HiddenDim, w.PosConvWeight, w.PosConvBias,
            groups: RvcHubertWeights.ConvPosGroups, kernel: RvcHubertWeights.ConvPosKernel);
        GeluErfInPlaceRows(pos);
        for (int i = 0; i < t; i++)
            for (int d = 0; d < RvcHubertWeights.HiddenDim; d++)
                hidden[i][d] += pos[i][d];

        // 4. Encoder input LayerNorm.
        for (int i = 0; i < t; i++)
            hidden[i] = LayerNorm(hidden[i], w.EncoderLayerNormWeight, w.EncoderLayerNormBias);

        // Real reference (`pad_odd_tokens_with_attention_mask`) pads an odd token count by one
        // zero frame, masked out of every position's attention (score -1e4 added for the padded
        // key) so it can't influence any real position's output, then trims it back off before
        // returning.
        int rawTokens = t;
        bool padded = t % 2 != 0;
        if (padded)
        {
            var withPad = new float[t + 1][];
            Array.Copy(hidden, withPad, t);
            withPad[t] = new float[RvcHubertWeights.HiddenDim];
            hidden = withPad;
            t++;
        }

        // 5. 12 standard post-LayerNorm transformer blocks, non-causal self-attention.
        foreach (var layer in w.Layers)
            hidden = EncoderBlock(hidden, layer, t, RvcHubertWeights.HiddenDim, RvcHubertWeights.NumHeads, maskLastKey: padded);

        if (padded)
        {
            var trimmed = new float[rawTokens][];
            Array.Copy(hidden, trimmed, rawTokens);
            hidden = trimmed;
        }

        return hidden;
    }

    private static float[][] ToFrames1Channel(ReadOnlySpan<float> waveform)
    {
        var frames = new float[waveform.Length][];
        for (int i = 0; i < waveform.Length; i++) frames[i] = [waveform[i]];
        return frames;
    }

    /// <summary>Valid (no padding) 1D conv, input frame-major [T, cin], weight [cout, cin, kernel] (real PyTorch layout), no bias (RVC's HuBERT uses FirstLayerGroupNorm, which omits conv bias per the reference).</summary>
    private static float[][] Conv1dValid(float[][] input, int cin, float[] weight, float[]? bias, int outCh, int kernel, int stride)
    {
        int tIn = input.Length;
        int tOut = (tIn - kernel) / stride + 1;
        var output = new float[tOut][];
        System.Threading.Tasks.Parallel.For(0, tOut, to =>
        {
            var row = new float[outCh];
            int srcBase = to * stride;
            for (int oc = 0; oc < outCh; oc++)
            {
                float sum = bias is null ? 0f : bias[oc];
                int wBase = oc * cin * kernel;
                for (int ic = 0; ic < cin; ic++)
                {
                    int wOff = wBase + ic * kernel;
                    for (int k = 0; k < kernel; k++)
                        sum += weight[wOff + k] * input[srcBase + k][ic];
                }
                row[oc] = sum;
            }
            output[to] = row;
        });
        return output;
    }

    /// <summary>Real PyTorch GroupNorm with numGroups == numChannels (i.e. per-channel InstanceNorm over the time axis, matching HuBERT's real `nn.GroupNorm(dim, dim, affine=True)` feature-extractor norm) -- normalizes EACH channel independently across all T frames, not per-frame.</summary>
    private static float[][] GroupNorm(float[][] x, int numGroups, float[] weight, float[] bias, float eps = 1e-5f)
    {
        int t = x.Length;
        int channels = weight.Length;
        var output = new float[t][];
        for (int i = 0; i < t; i++) output[i] = new float[channels];
        System.Threading.Tasks.Parallel.For(0, channels, c =>
        {
            double sum = 0, sumSq = 0;
            for (int i = 0; i < t; i++) { double v = x[i][c]; sum += v; sumSq += v * v; }
            double mean = sum / t;
            double variance = Math.Max(0, sumSq / t - mean * mean);
            double invStd = 1.0 / Math.Sqrt(variance + eps);
            for (int i = 0; i < t; i++)
                output[i][c] = (float)(((x[i][c] - mean) * invStd) * weight[c] + bias[c]);
        });
        return output;
    }

    private static void GeluErfInPlaceRows(float[][] x)
    {
        foreach (var row in x)
            for (int i = 0; i < row.Length; i++)
                row[i] = GeluErf(row[i]);
    }

    private static float GeluErf(float x) => (float)(0.5 * x * (1.0 + Erf(x / Math.Sqrt(2.0))));

    // Abramowitz-Stegun erf approximation (matches project convention where a real math-library
    // erf isn't already available -- max error ~1.5e-7, ample for this use).
    private static double Erf(double x)
    {
        double sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);
        const double a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
        return sign * y;
    }

    /// <summary>Real weight-normalized grouped positional conv: same padding (pad=kernel/2 each side), then the last output frame is dropped since kernel is even, matching the reference's `SliceModule({2, 0, input_length})` trim exactly.</summary>
    private static float[][] GroupedConv1dSamePad(float[][] input, int channels, float[] weight, float[] bias, int groups, int kernel)
    {
        int t = input.Length;
        int pad = kernel / 2;
        int chPerGroup = channels / groups;
        int tConv = t + 1; // (t + 2*pad - kernel)/1 + 1 == t+1 for even kernel with pad=kernel/2
        var conv = new float[tConv][];
        System.Threading.Tasks.Parallel.For(0, tConv, to =>
        {
            var row = new float[channels];
            for (int g = 0; g < groups; g++)
            {
                int chBase = g * chPerGroup;
                for (int ocLocal = 0; ocLocal < chPerGroup; ocLocal++)
                {
                    int oc = chBase + ocLocal;
                    float sum = bias[oc];
                    int wBase = oc * chPerGroup * kernel;
                    for (int icLocal = 0; icLocal < chPerGroup; icLocal++)
                    {
                        int ic = chBase + icLocal;
                        int wOff = wBase + icLocal * kernel;
                        for (int k = 0; k < kernel; k++)
                        {
                            int srcT = to - pad + k;
                            if ((uint)srcT < (uint)t)
                                sum += weight[wOff + k] * input[srcT][ic];
                        }
                    }
                    row[oc] = sum;
                }
            }
            conv[to] = row;
        });
        // Drop the trailing frame (even-kernel convention).
        var output = new float[t][];
        Array.Copy(conv, output, t);
        return output;
    }

    private static float[][] EncoderBlock(float[][] x, RvcHubertLayerWeights l, int t, int dim, int heads, bool maskLastKey)
    {
        var attnOut = SelfAttention(x, l, t, dim, heads, maskLastKey);
        var afterAttn = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = x[i][d] + attnOut[i][d];
            afterAttn[i] = LayerNorm(row, l.SelfAttnLayerNormWeight, l.SelfAttnLayerNormBias);
        }

        var output = new float[t][];
        System.Threading.Tasks.Parallel.For(0, t, i =>
        {
            var h = LinearBias(afterAttn[i], l.Fc1Weight, l.Fc1Bias, inDim: dim, outDim: l.Fc1Bias.Length);
            for (int d = 0; d < h.Length; d++) h[d] = GeluErf(h[d]);
            var down = LinearBias(h, l.Fc2Weight, l.Fc2Bias, inDim: h.Length, outDim: dim);
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = afterAttn[i][d] + down[d];
            output[i] = LayerNorm(row, l.FinalLayerNormWeight, l.FinalLayerNormBias);
        });
        return output;
    }

    private static float[][] SelfAttention(float[][] x, RvcHubertLayerWeights l, int t, int dim, int heads, bool maskLastKey)
    {
        int headDim = dim / heads;
        float scale = 1f / MathF.Sqrt(headDim);

        var q = new float[t][];
        var k = new float[t][];
        var v = new float[t][];
        System.Threading.Tasks.Parallel.For(0, t, i =>
        {
            q[i] = LinearBias(x[i], l.AttnQWeight, l.AttnQBias, dim, dim);
            k[i] = LinearBias(x[i], l.AttnKWeight, l.AttnKBias, dim, dim);
            v[i] = LinearBias(x[i], l.AttnVWeight, l.AttnVBias, dim, dim);
        });

        var context = new float[t][];
        for (int i = 0; i < t; i++) context[i] = new float[dim];

        for (int h = 0; h < heads; h++)
        {
            int off = h * headDim;
            System.Threading.Tasks.Parallel.For(0, t, i =>
            {
                var scores = new float[t];
                var qi = q[i].AsSpan(off, headDim);
                for (int j = 0; j < t; j++)
                    scores[j] = TensorPrimitives.Dot(qi, k[j].AsSpan(off, headDim)) * scale;
                if (maskLastKey) scores[t - 1] += -1.0e4f;
                OpenTail.Stingray.Audio.Primitives.DenseKernels.SoftmaxInPlace(scores);

                var weighted = context[i].AsSpan(off, headDim);
                for (int j = 0; j < t; j++)
                    TensorPrimitives.MultiplyAdd(v[j].AsSpan(off, headDim), scores[j], weighted, weighted);
            });
        }

        var output = new float[t][];
        for (int i = 0; i < t; i++)
            output[i] = LinearBias(context[i], l.AttnOutWeight, l.AttnOutBias, dim, dim);
        return output;
    }

    private static float[] LinearBias(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = bias[o];
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += input[i] * weight[wBase + i];
            output[o] = sum;
        }
        return output;
    }

    private static float[] LayerNorm(float[] x, float[] weight, float[] bias, float eps = 1e-5f)
    {
        int n = x.Length;
        double sum = 0;
        for (int i = 0; i < n; i++) sum += x[i];
        double mean = sum / n;
        double sumSq = 0;
        for (int i = 0; i < n; i++) { double d = x[i] - mean; sumSq += d * d; }
        double invStd = 1.0 / Math.Sqrt(sumSq / n + eps);
        var output = new float[n];
        for (int i = 0; i < n; i++)
            output[i] = (float)(((x[i] - mean) * invStd) * weight[i] + bias[i]);
        return output;
    }
}
