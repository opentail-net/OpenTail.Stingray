
namespace OpenTail.Stingray.Audio.NemotronAsr;

/// <summary>
/// Nemotron 3.5 ASR's `dw_striding` 8x subsampling front-end. Transcribed directly from
/// `examples/audio.cpp/src/models/nemotron_asr/encoder.cpp`'s offline (non-streaming) graph-build
/// path (`causal_conv_output_dim`/`pad_causal_2d`, lines ~35-71 and ~533-604) -- NOT guessed, and
/// notably NOT a "same"-padded front end like <see cref="Parakeet.ParakeetConformerEncoder"/>'s:
/// real kernel=3, stride=2 (3 stages, 8x total), but padding is ASYMMETRIC CAUSAL on BOTH the time
/// AND frequency axes at every stage (`pad_left = kernel-1 = 2`, `pad_right = stride-1 = 1`) --
/// the reference's own `pad_causal_2d` pads freq exactly the same way as time, a real difference
/// from a naive "pad=1 symmetric" assumption. Real stage sequence: full Conv2d(1-&gt;256,k3,s2,bias)
/// -&gt; ReLU -&gt; [DepthwiseConv2d(256,k3,s2,bias) -&gt; pointwise Conv2d(256-&gt;256,k1,bias) -&gt; ReLU] x2
/// -&gt; transpose(0,2,1,3)+reshape to `[1, framesOut, 256*featOut]` -&gt; Linear(256*featOut -&gt; 1024,
/// bias). The reference asserts `featOut == 17` (`256*17 = 4352 == pre_encode.out.weight`'s real
/// input dim) for its real 128-mel input -- this port's own formula must reproduce that.
/// </summary>
public static class NemotronAsrSubsampling
{
    /// <summary>
    /// mel is frame-major [T, NumMels] (row-major, mel[t*NumMels+f]) as produced by
    /// <see cref="NemotronAsrMelExtractor"/>. Returns per-frame subsampled hidden states
    /// [TEnc][HiddenDim] and the resulting frame count.
    /// </summary>
    public static (float[][] X, int TEnc) Forward(NemotronAsrWeights w, float[] mel, int tMel)
    {
        int nMels = w.FeatIn;
        int c = w.SubsampleChannels;
        const int k = 3, s = 2;

        var stage0 = Conv2dFullCausal(mel, cin: 1, hin: tMel, win: nMels, w.PreConv0Weight, w.PreConv0Bias, c, k, s, out int h1, out int w1);
        ReluInPlace(stage0);

        var stage1dw = Conv2dDepthwiseCausal(stage0, c, h1, w1, w.PreConv2Weight, w.PreConv2Bias, k, s, out int h2, out int w2);
        var stage1 = Conv2dPointwise(stage1dw, c, h2, w2, w.PreConv3Weight, w.PreConv3Bias, c);
        ReluInPlace(stage1);

        var stage2dw = Conv2dDepthwiseCausal(stage1, c, h2, w2, w.PreConv5Weight, w.PreConv5Bias, k, s, out int h3, out int w3);
        var stage2 = Conv2dPointwise(stage2dw, c, h3, w3, w.PreConv6Weight, w.PreConv6Bias, c);
        ReluInPlace(stage2);

        // permute(0,2,1,3) + flatten: feature[k] = channel*W3 + freq_w (channel-major), matching
        // Parakeet's own equivalent reshape (same GGUF ne=[KW,KH,Cin,Cout] convention).
        int flatDim = c * w3;
        int expectedFlatDim = w.PreOutWeight.Length / w.HiddenDim;
        if (flatDim != expectedFlatDim)
            throw new InvalidOperationException($"Nemotron ASR subsampling frequency dim mismatch: computed {w3} (flatDim={flatDim}), but pre_encode.out.weight expects flatDim={expectedFlatDim} (reference asserts featOut=17 for 128 mels).");
        var flat = new float[h3][];
        for (int t = 0; t < h3; t++)
        {
            var row = new float[flatDim];
            for (int ch = 0; ch < c; ch++)
                for (int fw = 0; fw < w3; fw++)
                    row[ch * w3 + fw] = stage2[ch * h3 * w3 + t * w3 + fw];
            flat[t] = row;
        }

        var output = new float[h3][];
        for (int t = 0; t < h3; t++)
            output[t] = Linear(flat[t], w.PreOutWeight, w.PreOutBias, flatDim, w.HiddenDim);

        return (output, h3);
    }

    private static int CausalOutDim(int input, int kernel, int stride)
    {
        int left = kernel - 1, right = stride - 1;
        return (input + left + right - kernel) / stride + 1;
    }

    /// <summary>Full (all-input-channel) conv2d with real asymmetric causal padding
    /// (`pad_causal_2d`: left=kernel-1, right=stride-1, on BOTH time(H) and freq(W) axes).</summary>
    private static float[] Conv2dFullCausal(float[] input, int cin, int hin, int win, float[] weight, float[] bias, int cout, int k, int stride, out int hout, out int wout)
    {
        int padTop = k - 1, padLeft = k - 1;
        hout = CausalOutDim(hin, k, stride);
        wout = CausalOutDim(win, k, stride);
        var output = new float[cout * hout * wout];
        for (int co = 0; co < cout; co++)
        {
            for (int ho = 0; ho < hout; ho++)
            {
                for (int wo = 0; wo < wout; wo++)
                {
                    float sum = bias[co];
                    for (int ci = 0; ci < cin; ci++)
                    {
                        for (int kh = 0; kh < k; kh++)
                        {
                            int hi = ho * stride - padTop + kh;
                            if (hi < 0 || hi >= hin) continue;
                            for (int kw = 0; kw < k; kw++)
                            {
                                int wi = wo * stride - padLeft + kw;
                                if (wi < 0 || wi >= win) continue;
                                float wt = weight[kw + kh * k + ci * k * k + co * k * k * cin];
                                sum += wt * input[ci * hin * win + hi * win + wi];
                            }
                        }
                    }
                    output[co * hout * wout + ho * wout + wo] = sum;
                }
            }
        }
        return output;
    }

    private static float[] Conv2dDepthwiseCausal(float[] input, int channels, int hin, int win, float[] weight, float[] bias, int k, int stride, out int hout, out int wout)
    {
        int padTop = k - 1, padLeft = k - 1;
        int houtLocal = CausalOutDim(hin, k, stride);
        int woutLocal = CausalOutDim(win, k, stride);
        hout = houtLocal;
        wout = woutLocal;
        var output = new float[channels * houtLocal * woutLocal];
        Parallel.For(0, channels, c =>
        {
            for (int ho = 0; ho < houtLocal; ho++)
            {
                for (int wo = 0; wo < woutLocal; wo++)
                {
                    float sum = bias[c];
                    for (int kh = 0; kh < k; kh++)
                    {
                        int hi = ho * stride - padTop + kh;
                        if (hi < 0 || hi >= hin) continue;
                        for (int kw = 0; kw < k; kw++)
                        {
                            int wi = wo * stride - padLeft + kw;
                            if (wi < 0 || wi >= win) continue;
                            float wt = weight[kw + kh * k + c * k * k];
                            sum += wt * input[c * hin * win + hi * win + wi];
                        }
                    }
                    output[c * houtLocal * woutLocal + ho * woutLocal + wo] = sum;
                }
            }
        });
        return output;
    }

    /// <summary>Pointwise (1x1, stride 1, no padding) full conv2d, i.e. a per-pixel Linear across channels.</summary>
    private static float[] Conv2dPointwise(float[] input, int cin, int h, int w, float[] weight, float[] bias, int cout)
    {
        var output = new float[cout * h * w];
        Parallel.For(0, cout, co =>
        {
            float b = bias[co];
            int wBase = co * cin;
            for (int hp = 0; hp < h; hp++)
            {
                for (int wp = 0; wp < w; wp++)
                {
                    float sum = b;
                    for (int ci = 0; ci < cin; ci++)
                        sum += weight[wBase + ci] * input[ci * h * w + hp * w + wp];
                    output[co * h * w + hp * w + wp] = sum;
                }
            }
        });
        return output;
    }

    private static void ReluInPlace(float[] x)
    {
        TensorPrimitives.Max((ReadOnlySpan<float>)x, 0f, x);
    }

    private static float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
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
}
