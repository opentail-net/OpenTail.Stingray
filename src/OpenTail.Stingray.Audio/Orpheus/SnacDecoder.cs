using System;
using System.Threading.Tasks;

namespace OpenTail.Stingray.Audio.Orpheus;

/// <summary>
/// Real SNAC 24kHz decoder forward pass, transcribed directly from the real `snac` Python
/// package (`snac/layers.py`, `snac/vq.py`, `snac/snac.py`, fetched via `pip download snac
/// --no-deps`), config confirmed from the real `hubertsiuzdak/snac_24khz` HF `config.json` -- see
/// <see cref="SnacWeights"/>'s doc comment for the full derivation and real tensor names. Only
/// the DECODE path is ported (`ResidualVectorQuantize.from_codes` -> `Decoder.forward`) since
/// Orpheus only ever calls `SNAC.decode(codes)`, never encodes real audio.
///
/// <para><b>NoiseBlock is made a no-op, matching the real C++ reference's documented choice, not
/// a shortcut of this port's own invention</b>: the real `NoiseBlock.forward` injects
/// `torch.randn(...)` at every call, making the real PyTorch decoder itself non-deterministic.
/// `examples/CrispASR/tools/reference_backends/orpheus_snac.py` (the real oracle used for golden
/// verification here) explicitly monkey-patches `NoiseBlock.forward = lambda self, x: x` for
/// exactly this reason, noting "the noise contribution is ~1e-2 of the signal RMS" -- this port
/// follows the same documented convention, so `noise.weight` is loaded but never used.</para>
///
/// <para><b>Real per-layer math, in order (do not reorder/guess)</b>: `Decoder.forward`:
/// depthwise conv (in0, k=7, 768ch) -&gt; pointwise conv (in1, 768-&gt;1024, k=1) -&gt; 4x
/// `DecoderBlock` (each: `Snake1d` -&gt; `ConvTranspose1d` upsample -&gt; [NoiseBlock, no-op'd] -&gt;
/// 3x `ResidualUnit` with dilations 1/3/9) -&gt; final `Snake1d` -&gt; conv (64-&gt;1, k=7) -&gt;
/// `Tanh`. Each `ResidualUnit`: `Snake1d` -&gt; depthwise dilated conv (k=7, `pad=(kernel-1)*
/// dilation/2`) -&gt; `Snake1d` -&gt; pointwise conv (k=1) -&gt; residual add (real code
/// center-crops the residual to match the conv output length via `pad=(x.T - y.T)/2`, but with
/// same-padding depthwise convs T never shrinks here, so the crop is always a no-op -- kept as
/// an explicit no-op-safe add, not silently assumed away).</para>
/// </summary>
public static class SnacDecoder
{
    /// <summary>Real `snake(x, alpha)`: `x + (1/(alpha+1e-9)) * sin(alpha*x)^2`, per-channel alpha.</summary>
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

    /// <summary>Real depthwise (groups=channels) dilated Conv1d, same-padding (`pad=(kernel-1)*dilation/2`), weight layout [out=channels, inPerGroup=1, kernel] flat row-major (confirmed matches this GGUF's real flat byte layout, see SnacWeights doc comment).</summary>
    private static float[] DepthwiseConv1d(float[] x, int channels, int t, float[] weight, float[] bias, int kernel, int dilation)
    {
        int pad = (kernel - 1) * dilation / 2;
        var output = new float[x.Length];
        Parallel.For(0, channels, c =>
        {
            int xBase = c * t;
            int wBase = c * kernel;
            float b = bias[c];
            for (int ti = 0; ti < t; ti++)
            {
                float sum = b;
                for (int k = 0; k < kernel; k++)
                {
                    int src = ti - pad + k * dilation;
                    if ((uint)src < (uint)t) sum += x[xBase + src] * weight[wBase + k];
                }
                output[xBase + ti] = sum;
            }
        });
        return output;
    }

    /// <summary>Real pointwise (kernel=1) Conv1d: weight layout [out, in, 1] flat row-major -> effectively a per-position Linear across channels.</summary>
    private static float[] PointwiseConv1d(float[] x, int inCh, int outCh, int t, float[] weight, float[] bias)
    {
        var output = new float[outCh * t];
        Parallel.For(0, outCh, oc =>
        {
            float b = bias[oc];
            int wBase = oc * inCh;
            int outBase = oc * t;
            for (int ti = 0; ti < t; ti++)
            {
                float sum = b;
                for (int ic = 0; ic < inCh; ic++)
                    sum += x[ic * t + ti] * weight[wBase + ic];
                output[outBase + ti] = sum;
            }
        });
        return output;
    }

    /// <summary>Real ConvTranspose1d, weight layout [in, out, kernel] flat row-major (matches HiFTVocoderKernels' established convention). `output_padding` is always 0 for this checkpoint's strides (2/4/8 all even -> `stride % 2 == 0`), so it's omitted rather than modeled unused.</summary>
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

    private static float[] ResidualUnit(float[] x, int channels, int t, SnacResidualUnitWeights w, int dilation)
    {
        var y = Snake1d(x, channels, t, w.Alpha0);
        y = DepthwiseConv1d(y, channels, t, w.Conv0Weight, w.Conv0Bias, kernel: 7, dilation: dilation);
        y = Snake1d(y, channels, t, w.Alpha1);
        y = PointwiseConv1d(y, channels, channels, t, w.Conv1Weight, w.Conv1Bias);

        // Real code center-crops the residual if the conv output is shorter; same-padding convs
        // above never shrink T here, so this is always a no-op in practice -- kept explicit, not
        // silently assumed, matching the real ResidualUnit.forward's own defensive crop.
        var output = new float[y.Length];
        for (int i = 0; i < y.Length; i++) output[i] = x[i] + y[i];
        return output;
    }

    private static (float[] Data, int T) DecoderBlock(float[] x, int inCh, int outCh, int t, SnacDecoderBlockWeights w, int stride)
    {
        var y = Snake1d(x, inCh, t, w.Alpha);
        int kernel = 2 * stride;
        int padding = (int)MathF.Ceiling(stride / 2f);
        var up = ConvTranspose1d(y, inCh, outCh, t, w.UpWeight, w.UpBias, kernel, stride, padding);
        int outT = (t - 1) * stride - 2 * padding + kernel;

        // NoiseBlock is a documented no-op here, see class doc comment -- w.Res is applied
        // directly to `up`, matching the real graph with NoiseBlock.forward patched to identity.
        var cur = up;
        cur = ResidualUnit(cur, outCh, outT, w.Res[0], dilation: 1);
        cur = ResidualUnit(cur, outCh, outT, w.Res[1], dilation: 3);
        cur = ResidualUnit(cur, outCh, outT, w.Res[2], dilation: 9);
        return (cur, outT);
    }

    /// <summary>Real `ResidualVectorQuantize.from_codes`: per-quantizer embedding lookup -> out_proj (pointwise conv) -> nearest-neighbor time-upsample by that quantizer's own stride -> sum across quantizers. `codes[i]` are the real de-interleaved SNAC codebook indices (see FunASR-analogous doc comment in the pipeline class for the token de-interleaving formula), each in `[0, CodebookSize)`.</summary>
    public static float[] QuantizerFromCodes(SnacWeights w, int[][] codes)
    {
        int tOut = codes[^1].Length; // codes[2] (stride 1) defines the decoder-input rate
        var zq = new float[SnacWeights.LatentDim * tOut];

        for (int qi = 0; qi < w.Quantizers.Length; qi++)
        {
            var q = w.Quantizers[qi];
            int tIn = codes[qi].Length;

            // decode_code: embedding lookup [T, CodebookDim] -> effectively transpose to [CodebookDim, T].
            var embed = new float[SnacWeights.CodebookDim * tIn];
            for (int ti = 0; ti < tIn; ti++)
            {
                int code = codes[qi][ti];
                int cbBase = code * SnacWeights.CodebookDim;
                for (int d = 0; d < SnacWeights.CodebookDim; d++)
                    embed[d * tIn + ti] = q.Codebook[cbBase + d];
            }

            var proj = PointwiseConv1d(embed, SnacWeights.CodebookDim, SnacWeights.LatentDim, tIn, q.OutProjWeight, q.OutProjBias);

            // repeat_interleave(stride, dim=-1): nearest-neighbor upsample along time, then sum in.
            int stride = q.Stride;
            Parallel.For(0, SnacWeights.LatentDim, d =>
            {
                int srcBase = d * tIn;
                int dstBase = d * tOut;
                for (int ti = 0; ti < tIn; ti++)
                {
                    float v = proj[srcBase + ti];
                    int dstStart = ti * stride;
                    for (int s = 0; s < stride; s++)
                        zq[dstBase + dstStart + s] += v;
                }
            });
        }

        return zq;
    }

    /// <summary>Full real decode: 3 codebook streams -> quantizer.from_codes -> Decoder.forward -> mono float32 PCM at 24kHz, range [-1, 1] (post-Tanh).</summary>
    public static float[] Decode(SnacWeights w, int[][] codes)
    {
        var zq = QuantizerFromCodes(w, codes);
        int t = codes[^1].Length;

        var x = DepthwiseConv1d(zq, SnacWeights.LatentDim, t, w.In0Weight, w.In0Bias, kernel: 7, dilation: 1);
        x = PointwiseConv1d(x, SnacWeights.LatentDim, SnacWeights.DecoderDim, t, w.In1Weight, w.In1Bias);

        int ch = SnacWeights.DecoderDim;
        int curT = t;
        for (int i = 0; i < SnacWeights.DecoderRates.Length; i++)
        {
            int outCh = ch / 2;
            (x, curT) = DecoderBlock(x, ch, outCh, curT, w.DecBlocks[i], SnacWeights.DecoderRates[i]);
            ch = outCh;
        }

        x = Snake1d(x, ch, curT, w.OutAlpha);
        var pcm = FullConv1dToMono(x, ch, curT, w.OutWeight, w.OutBias);

        for (int i = 0; i < pcm.Length; i++) pcm[i] = MathF.Tanh(pcm[i]);
        return pcm;
    }

    /// <summary>Real final conv: FULL (non-grouped, `groups=1`, unlike the depthwise `ResidualUnit`/`in0` convs above) Conv1d, `channels -> 1`, kernel=7, same-padding. Weight layout [out=1, in=channels, kernel] flat row-major.</summary>
    private static float[] FullConv1dToMono(float[] x, int channels, int t, float[] weight, float[] bias)
    {
        const int kernel = 7;
        const int pad = 3;
        var output = new float[t];
        float b = bias[0];
        Parallel.For(0, t, ti =>
        {
            float sum = b;
            for (int c = 0; c < channels; c++)
            {
                int wBase = c * kernel;
                int xBase = c * t;
                for (int k = 0; k < kernel; k++)
                {
                    int src = ti - pad + k;
                    if ((uint)src < (uint)t) sum += x[xBase + src] * weight[wBase + k];
                }
            }
            output[ti] = sum;
        });
        return output;
    }
}
