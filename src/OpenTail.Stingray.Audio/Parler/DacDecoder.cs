using System;
using System.Threading.Tasks;

namespace OpenTail.Stingray.Audio.Parler;

/// <summary>
/// Real DAC (Descript Audio Codec) decoder forward pass for Parler-TTS's `audio_encoder`,
/// transcribed directly from the real `descript-audio-codec` Python package (`dac/model/dac.py`,
/// `dac/nn/quantize.py`, fetched via `pip download descript-audio-codec --no-deps`) -- see
/// <see cref="DacWeights"/>'s doc comment for the full derivation.
///
/// <para><b>Two real differences from <see cref="Orpheus.SnacDecoder"/>, do not reuse that
/// class's kernels blindly</b>: (1) DAC's `ResidualUnit` uses FULL (non-depthwise) `Conv1d` --
/// the real `WNConv1d(dim,dim,kernel=7,dilation=dilation,padding=pad)` call has no `groups`
/// parameter at all, unlike SNAC's depthwise convention. (2) DAC's quantizer sums all 9
/// codebooks' contributions at the SAME time resolution (`from_codes`: `sum(out_proj(codebook[i]
/// [codes[i]]))`, no `repeat_interleave`/stride step) -- there is no SNAC-style hierarchical
/// 1/2/4-rate split here, every codebook stream must already be the same length.</para>
///
/// <para>Real per-layer math (same shape as SNAC otherwise): `Decoder.forward`: one plain FULL
/// conv (latent_dim -&gt; decoder_dim, k=7) -&gt; 4x `DecoderBlock` (`Snake1d` -&gt;
/// `ConvTranspose1d` upsample (real rates `[8,8,4,2]`) -&gt; 3x `ResidualUnit` dilations 1/3/9,
/// each: `Snake1d` -&gt; FULL dilated conv (k=7) -&gt; `Snake1d` -&gt; FULL conv (k=1) -&gt;
/// residual) -&gt; final `Snake1d` -&gt; FULL conv (channels-&gt;1, k=7) -&gt; `Tanh`. No
/// `NoiseBlock` anywhere in real DAC (unlike SNAC) -- nothing to no-op here.</para>
/// </summary>
public static class DacDecoder
{
    private static float[] Snake1d(float[] x, int channels, int t, float[] alpha)
    {
        var output = new float[x.Length];
        Parallel.For(0, channels, c =>
        {
            float a = alpha[c];
            float invA = 1f / (a + 1e-9f);
            int baseIdx = c * t;
            for (int i = 0; i < t; i++)
            {
                float v = x[baseIdx + i];
                float s = MathF.Sin(a * v);
                output[baseIdx + i] = v + invA * s * s;
            }
        });
        return output;
    }

    /// <summary>Real FULL (non-depthwise) Conv1d, same-padding, optionally dilated. Weight layout [out, in, kernel] flat row-major.</summary>
    private static float[] FullConv1d(float[] x, int inCh, int outCh, int t, float[] weight, float[] bias, int kernel, int dilation, int padding)
    {
        var output = new float[outCh * t];
        Parallel.For(0, outCh, oc =>
        {
            float b = bias[oc];
            int wOcBase = oc * inCh * kernel;
            int outBase = oc * t;
            for (int ti = 0; ti < t; ti++)
            {
                float sum = b;
                for (int ic = 0; ic < inCh; ic++)
                {
                    int xBase = ic * t;
                    int wBase = wOcBase + ic * kernel;
                    for (int k = 0; k < kernel; k++)
                    {
                        int src = ti - padding + k * dilation;
                        if ((uint)src < (uint)t) sum += x[xBase + src] * weight[wBase + k];
                    }
                }
                output[outBase + ti] = sum;
            }
        });
        return output;
    }

    /// <summary>Real ConvTranspose1d, weight layout [in, out, kernel] flat row-major (same convention as SnacDecoder/HiFTVocoderKernels).</summary>
    private static float[] ConvTranspose1d(float[] x, int inCh, int outCh, int t, float[] weight, float[] bias, int kernel, int stride, int padding)
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
                    for (int k = 0; k < kernel; k++)
                    {
                        int to = outStart + k;
                        if ((uint)to < (uint)outT) output[dstBase + to] += v * weight[wBase + k];
                    }
                }
            }
        });
        return output;
    }

    private static float[] ResidualUnit(float[] x, int channels, int t, DacResidualUnitWeights w, int dilation)
    {
        int pad = (7 - 1) * dilation / 2;
        var y = Snake1d(x, channels, t, w.Alpha0);
        y = FullConv1d(y, channels, channels, t, w.Conv0Weight, w.Conv0Bias, kernel: 7, dilation: dilation, padding: pad);
        y = Snake1d(y, channels, t, w.Alpha1);
        y = FullConv1d(y, channels, channels, t, w.Conv1Weight, w.Conv1Bias, kernel: 1, dilation: 1, padding: 0);

        var output = new float[y.Length];
        for (int i = 0; i < y.Length; i++) output[i] = x[i] + y[i];
        return output;
    }

    private static (float[] Data, int T) DecoderBlock(float[] x, int inCh, int outCh, int t, DacDecoderBlockWeights w, int stride)
    {
        var y = Snake1d(x, inCh, t, w.Alpha);
        int kernel = 2 * stride;
        int padding = (int)MathF.Ceiling(stride / 2f);
        var up = ConvTranspose1d(y, inCh, outCh, t, w.UpWeight, w.UpBias, kernel, stride, padding);
        int outT = (t - 1) * stride - 2 * padding + kernel;

        var cur = up;
        cur = ResidualUnit(cur, outCh, outT, w.Res[0], dilation: 1);
        cur = ResidualUnit(cur, outCh, outT, w.Res[1], dilation: 3);
        cur = ResidualUnit(cur, outCh, outT, w.Res[2], dilation: 9);
        return (cur, outT);
    }

    /// <summary>Real `ResidualVectorQuantize.from_codes`: per-quantizer embedding lookup -> out_proj (pointwise conv) -> sum across all 9 quantizers at the SAME time resolution (no upsampling, unlike SNAC).</summary>
    public static float[] QuantizerFromCodes(DacWeights w, int[][] codes)
    {
        int t = codes[0].Length;
        var zq = new float[DacWeights.LatentDim * t];

        for (int qi = 0; qi < w.Quantizers.Length; qi++)
        {
            var q = w.Quantizers[qi];

            var embed = new float[DacWeights.CodebookDim * t];
            for (int ti = 0; ti < t; ti++)
            {
                int code = codes[qi][ti];
                int cbBase = code * DacWeights.CodebookDim;
                for (int d = 0; d < DacWeights.CodebookDim; d++)
                    embed[d * t + ti] = q.Codebook[cbBase + d];
            }

            var proj = FullConv1d(embed, DacWeights.CodebookDim, DacWeights.LatentDim, t, q.OutProjWeight, q.OutProjBias, kernel: 1, dilation: 1, padding: 0);
            for (int i = 0; i < zq.Length; i++) zq[i] += proj[i];
        }

        return zq;
    }

    /// <summary>Full real decode: 9 codebook streams (same time resolution) -> quantizer.from_codes -> Decoder.forward -> mono float32 PCM at 44.1kHz, range [-1, 1] (post-Tanh).</summary>
    public static float[] Decode(DacWeights w, int[][] codes)
    {
        var zq = QuantizerFromCodes(w, codes);
        int t = codes[0].Length;

        var x = FullConv1d(zq, DacWeights.LatentDim, DacWeights.DecoderDim, t, w.In0Weight, w.In0Bias, kernel: 7, dilation: 1, padding: 3);

        int ch = DacWeights.DecoderDim;
        int curT = t;
        for (int i = 0; i < DacWeights.DecoderRates.Length; i++)
        {
            int outCh = ch / 2;
            (x, curT) = DecoderBlock(x, ch, outCh, curT, w.DecBlocks[i], DacWeights.DecoderRates[i]);
            ch = outCh;
        }

        x = Snake1d(x, ch, curT, w.OutAlpha);
        var pcm = FullConv1d(x, ch, 1, curT, w.OutWeight, w.OutBias, kernel: 7, dilation: 1, padding: 3);

        for (int i = 0; i < pcm.Length; i++) pcm[i] = MathF.Tanh(pcm[i]);
        return pcm;
    }
}
