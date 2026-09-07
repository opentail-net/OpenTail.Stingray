
namespace OpenTail.Stingray.Audio.OmniVoice;

/// <summary>
/// Real HuBERT-base (Wav2Vec2-family) semantic encoder forward pass for OmniVoice --
/// architecturally IDENTICAL to <see cref="Rvc.RvcHubertEncoder"/> (confirmed via a real
/// hyperparameter and tensor-name match, see <see cref="OmniVoiceSemanticWeights"/>'s doc
/// comment), duplicated rather than shared because the two live in different weight-source
/// classes (packed GGUF vs safetensors) -- a candidate for consolidation in a later DRY pass once
/// this port is complete, per this project's standard porting sequence, not before.
/// </summary>
public static class OmniVoiceSemanticEncoder
{
    /// <summary>Runs the encoder on a 16kHz mono waveform. Returns frame-major [numFrames, 768]
    /// hidden states after all 12 layers.</summary>
    public static float[][] Forward(OmniVoiceSemanticWeights w, ReadOnlySpan<float> waveform16k)
    {
        float[][] x = ToFrames1Channel(waveform16k);
        int channels = 1;
        for (int layerIdx = 0; layerIdx < 7; layerIdx++)
        {
            x = Conv1dValid(x, channels, w.ConvWeights[layerIdx], outCh: OmniVoiceSemanticWeights.ConvDim[layerIdx], kernel: OmniVoiceSemanticWeights.ConvKernel[layerIdx], stride: OmniVoiceSemanticWeights.ConvStride[layerIdx]);
            channels = OmniVoiceSemanticWeights.ConvDim[layerIdx];
            if (layerIdx == 0)
                x = GroupNorm(x, w.ConvLayer0GroupNormWeight, w.ConvLayer0GroupNormBias);
            GeluErfInPlaceRows(x);
        }

        int t = x.Length;
        for (int i = 0; i < t; i++)
            x[i] = LayerNorm(x[i], w.FeatureProjLayerNormWeight, w.FeatureProjLayerNormBias);
        var hidden = new float[t][];
        for (int i = 0; i < t; i++)
            hidden[i] = LinearBias(x[i], w.FeatureProjWeight, w.FeatureProjBias, inDim: 512, outDim: OmniVoiceSemanticWeights.HiddenDim);

        var pos = GroupedConv1dSamePad(hidden, OmniVoiceSemanticWeights.HiddenDim, w.PosConvWeight, w.PosConvBias,
            groups: OmniVoiceSemanticWeights.ConvPosGroups, kernel: OmniVoiceSemanticWeights.ConvPosKernel);
        GeluErfInPlaceRows(pos);
        for (int i = 0; i < t; i++)
            for (int d = 0; d < OmniVoiceSemanticWeights.HiddenDim; d++)
                hidden[i][d] += pos[i][d];

        for (int i = 0; i < t; i++)
            hidden[i] = LayerNorm(hidden[i], w.EncoderLayerNormWeight, w.EncoderLayerNormBias);

        int rawTokens = t;
        bool padded = t % 2 != 0;
        if (padded)
        {
            var withPad = new float[t + 1][];
            Array.Copy(hidden, withPad, t);
            withPad[t] = new float[OmniVoiceSemanticWeights.HiddenDim];
            hidden = withPad;
            t++;
        }

        foreach (var layer in w.Layers)
            hidden = EncoderBlock(hidden, layer, t, OmniVoiceSemanticWeights.HiddenDim, OmniVoiceSemanticWeights.NumHeads, maskLastKey: padded);

        if (padded)
        {
            var trimmed = new float[rawTokens][];
            Array.Copy(hidden, trimmed, rawTokens);
            hidden = trimmed;
        }

        return hidden;
    }

    /// <summary>
    /// Real Higgs Audio TTS variant, ported from `codec.cpp`'s `hubert_hidden_state_mean`/
    /// `downsample_time_by_2` (not guessed), added 2026-09-07: instead of returning the FINAL
    /// layer's hidden states (what <see cref="Forward"/> does, correct for OmniVoice's own real
    /// usage), Higgs's real reference-audio encode path AVERAGES the hidden state across all 13
    /// real snapshots (the post-pos-conv/layer-norm state PLUS each of the 12 layers' own
    /// output, divided by 13), then keeps only every OTHER frame up to `targetFrames` (a real 2x
    /// downsample by even-index selection, NOT pair-averaging). `targetFrames` is the caller's
    /// already-computed acoustic-encoder frame count that this semantic stream must align to.
    /// </summary>
    public static float[][] ForwardHiddenStateMean(OmniVoiceSemanticWeights w, ReadOnlySpan<float> waveform16k, int targetFrames)
    {
        float[][] x = ToFrames1Channel(waveform16k);
        int channels = 1;
        for (int layerIdx = 0; layerIdx < 7; layerIdx++)
        {
            x = Conv1dValid(x, channels, w.ConvWeights[layerIdx], outCh: OmniVoiceSemanticWeights.ConvDim[layerIdx], kernel: OmniVoiceSemanticWeights.ConvKernel[layerIdx], stride: OmniVoiceSemanticWeights.ConvStride[layerIdx]);
            channels = OmniVoiceSemanticWeights.ConvDim[layerIdx];
            if (layerIdx == 0)
                x = GroupNorm(x, w.ConvLayer0GroupNormWeight, w.ConvLayer0GroupNormBias);
            GeluErfInPlaceRows(x);
        }

        int t = x.Length;
        for (int i = 0; i < t; i++)
            x[i] = LayerNorm(x[i], w.FeatureProjLayerNormWeight, w.FeatureProjLayerNormBias);
        var hidden = new float[t][];
        for (int i = 0; i < t; i++)
            hidden[i] = LinearBias(x[i], w.FeatureProjWeight, w.FeatureProjBias, inDim: 512, outDim: OmniVoiceSemanticWeights.HiddenDim);

        var pos = GroupedConv1dSamePad(hidden, OmniVoiceSemanticWeights.HiddenDim, w.PosConvWeight, w.PosConvBias,
            groups: OmniVoiceSemanticWeights.ConvPosGroups, kernel: OmniVoiceSemanticWeights.ConvPosKernel);
        GeluErfInPlaceRows(pos);
        for (int i = 0; i < t; i++)
            for (int d = 0; d < OmniVoiceSemanticWeights.HiddenDim; d++)
                hidden[i][d] += pos[i][d];

        for (int i = 0; i < t; i++)
            hidden[i] = LayerNorm(hidden[i], w.EncoderLayerNormWeight, w.EncoderLayerNormBias);

        int rawTokens = t;
        bool padded = t % 2 != 0;
        if (padded)
        {
            var withPad = new float[t + 1][];
            Array.Copy(hidden, withPad, t);
            withPad[t] = new float[OmniVoiceSemanticWeights.HiddenDim];
            hidden = withPad;
            t++;
        }

        // Real running sum, starting from the post-pos-conv/layer-norm state (BEFORE layer 0).
        var sum = new float[t][];
        for (int i = 0; i < t; i++) sum[i] = (float[])hidden[i].Clone();

        foreach (var layer in w.Layers)
        {
            hidden = EncoderBlock(hidden, layer, t, OmniVoiceSemanticWeights.HiddenDim, OmniVoiceSemanticWeights.NumHeads, maskLastKey: padded);
            for (int i = 0; i < t; i++)
                for (int d = 0; d < OmniVoiceSemanticWeights.HiddenDim; d++)
                    sum[i][d] += hidden[i][d];
        }

        if (padded)
        {
            var trimmed = new float[rawTokens][];
            Array.Copy(sum, trimmed, rawTokens);
            sum = trimmed;
            t = rawTokens;
        }

        float invCount = 1f / (OmniVoiceSemanticWeights.NumLayers + 1);
        for (int i = 0; i < t; i++)
            for (int d = 0; d < OmniVoiceSemanticWeights.HiddenDim; d++)
                sum[i][d] *= invCount;

        // Real `downsample_time_by_2`: keep only even-indexed frames, up to targetFrames -- NOT
        // pair-averaging.
        var output = new float[targetFrames][];
        for (int f = 0; f < targetFrames; f++)
            output[f] = sum[f * 2];
        return output;
    }

    private static float[][] ToFrames1Channel(ReadOnlySpan<float> waveform)
    {
        var frames = new float[waveform.Length][];
        for (int i = 0; i < waveform.Length; i++) frames[i] = [waveform[i]];
        return frames;
    }

    private static float[][] Conv1dValid(float[][] input, int cin, float[] weight, int outCh, int kernel, int stride)
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
                float sum = 0f;
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

    private static float[][] GroupNorm(float[][] x, float[] weight, float[] bias, float eps = 1e-5f)
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

    private static double Erf(double x)
    {
        double sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);
        const double a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
        return sign * y;
    }

    private static float[][] GroupedConv1dSamePad(float[][] input, int channels, float[] weight, float[] bias, int groups, int kernel)
    {
        int t = input.Length;
        int pad = kernel / 2;
        int chPerGroup = channels / groups;
        int tConv = t + 1;
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
        var output = new float[t][];
        Array.Copy(conv, output, t);
        return output;
    }

    private static float[][] EncoderBlock(float[][] x, OmniVoiceSemanticLayerWeights l, int t, int dim, int heads, bool maskLastKey)
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

    private static float[][] SelfAttention(float[][] x, OmniVoiceSemanticLayerWeights l, int t, int dim, int heads, bool maskLastKey)
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
