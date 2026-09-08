using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.PersonaPlex;

/// <summary>
/// Real forward pass for the Mimi neural codec's DECODE path (RVQ codes -> waveform), ported
/// from `mimi_codec_runtime.cpp`'s `quantizer_decode`/`MimiDecoderPreTransformerGraph`/
/// `MimiTransformerRuntime`/`MimiDecoderPostTransformerGraph` (not guessed). A ONE-SHOT
/// (non-streaming) port: every real "stateful" conv in the reference degenerates to a plain
/// causal (left-zero-pad) Conv1d or plain unpadded ConvTranspose1d for a single full-sequence
/// call with no prior state -- confirmed via the reference's own `history=nullopt`/
/// `partial=nullopt` first-call behavior (see this port's own progress-doc scoping entry).
///
/// <para>Real pipeline: `quantizer_decode` (codebook 0 = semantic, summed codebooks 1-7 =
/// acoustic, each stream separately projected then ADDED) -&gt; a real per-channel DEPTHWISE
/// `ConvTranspose1d(kernel=4,stride=2)` doubling the frame rate (real, kernel-reversed-at-load
/// weight) -&gt; 8-layer real pre-norm transformer (bias-affine LayerNorm, packed-QKV causal MHA,
/// LayerScale, GELU-erf FFN) -&gt; `Conv1d(hidden-&gt;1024,k=7)` -&gt; 4x
/// [ELU -&gt; `ConvTranspose1d`(real per-stage ratio) -&gt; ONE SEANet residual block] -&gt; ELU
/// -&gt; `Conv1d(64-&gt;channels,k=3)` -&gt; clamp to [-1,1].</para>
/// </summary>
public static class MimiCodecDecoder
{
    /// <summary>Decodes `codes[frames][8]` (real active-codebook ids, codebook 0 semantic,
    /// 1-7 acoustic, each in `[0, codebookSize)`) into a mono waveform.</summary>
    public static float[] Decode(MimiCodecDecoderWeights w, int[][] codes)
    {
        int frames = codes.Length;
        var quantized = QuantizerDecode(w, codes); // [hidden][frames]

        var upsampled = DepthwiseConvTranspose1d(quantized, MimiCodecDecoderWeights.HiddenDim, w.FrameUpsampleWeight, w.FrameUpsampleBias, kernel: 4, stride: 2);

        var transformed = RunTransformer(w, upsampled);

        var hidden = CausalConv1d(transformed, MimiCodecDecoderWeights.HiddenDim, 1024, w.DecoderInputProjWeight, w.DecoderInputProjBias, kernel: 7, dilation: 1);

        for (int stage = 0; stage < 4; stage++)
        {
            int inCh = MimiCodecDecoderWeights.StageInChannels[stage];
            int outCh = MimiCodecDecoderWeights.StageOutChannels[stage];
            int hiddenCh = MimiCodecDecoderWeights.StageHiddenChannels[stage];
            int kernel = MimiCodecDecoderWeights.StageKernelSizes[stage];
            int stride = MimiCodecDecoderWeights.StageStrides[stage];
            var sw = w.Stages[stage];

            Elu(hidden);
            hidden = ConvTranspose1d(hidden, inCh, outCh, sw.UpsampleWeight, sw.UpsampleBias, kernel, stride);
            hidden = SeanetResidualUnit(hidden, outCh, hiddenCh, sw.ResidualBlock);
        }

        Elu(hidden);
        var output = CausalConv1d(hidden, 64, MimiCodecDecoderWeights.Channels, w.DecoderOutputProjWeight, w.DecoderOutputProjBias, kernel: 3, dilation: 1);

        var waveform = output[0];
        for (int i = 0; i < waveform.Length; i++) waveform[i] = Math.Clamp(waveform[i], -1f, 1f);
        return waveform;
    }

    private static float[][] QuantizerDecode(MimiCodecDecoderWeights w, int[][] codes)
    {
        int frames = codes.Length;
        int hidden = MimiCodecDecoderWeights.HiddenDim;
        int latent = MimiCodecDecoderWeights.LatentSize;

        // Semantic stream: codebook 0 lookup -> project.
        var semanticLatent = new float[frames][];
        for (int t = 0; t < frames; t++)
        {
            var embed = new float[latent];
            Array.Copy(w.SemanticCodebook.Embedding, (long)codes[t][0] * latent, embed, 0, latent);
            semanticLatent[t] = embed;
        }
        var semanticProjected = ProjectPerFrame(semanticLatent, w.SemanticOutputProjWeight, latent, hidden);

        // Acoustic stream: sum codebooks 1..7 embeddings, then ONE shared projection.
        var acousticLatent = new float[frames][];
        for (int t = 0; t < frames; t++) acousticLatent[t] = new float[latent];
        for (int cb = 1; cb < MimiCodecDecoderWeights.ActiveCodebooks; cb++)
        {
            var table = w.AcousticCodebooks[cb - 1].Embedding;
            for (int t = 0; t < frames; t++)
            {
                long baseIdx = (long)codes[t][cb] * latent;
                for (int d = 0; d < latent; d++) acousticLatent[t][d] += table[baseIdx + d];
            }
        }
        var acousticProjected = ProjectPerFrame(acousticLatent, w.AcousticOutputProjWeight, latent, hidden);

        // Combine into channel-major [hidden][frames], real semantic+acoustic ADD.
        var output = new float[hidden][];
        for (int c = 0; c < hidden; c++)
        {
            output[c] = new float[frames];
            for (int t = 0; t < frames; t++) output[c][t] = semanticProjected[t][c] + acousticProjected[t][c];
        }
        return output;
    }

    private static float[][] ProjectPerFrame(float[][] frameMajorLatent, float[] weight, int inDim, int outDim)
    {
        int frames = frameMajorLatent.Length;
        var output = new float[frames][];
        for (int t = 0; t < frames; t++) output[t] = DenseKernels.LinearNoBias(frameMajorLatent[t], weight, inDim, outDim);
        return output;
    }

    private static float[][] RunTransformer(MimiCodecDecoderWeights w, float[][] channelMajor) =>
        RunTransformer(w.TransformerLayers, channelMajor);

    /// <summary>Real 8-layer Mimi transformer (pre-norm, packed-QKV causal MHA, LayerScale,
    /// GELU-erf FFN) shared by BOTH the decoder and encoder directions -- same real per-layer
    /// structure, different learned weights (`decoder_transformer.*` vs `encoder_transformer.*`
    /// in the checkpoint).</summary>
    internal static float[][] RunTransformer(MimiCodecTransformerLayerWeights[] layers, float[][] channelMajor)
    {
        int hidden = MimiCodecDecoderWeights.HiddenDim;
        int frames = channelMajor[0].Length;
        int numHeads = MimiCodecDecoderWeights.NumHeads;
        int headDim = MimiCodecDecoderWeights.HeadDim;
        float scale = 1f / MathF.Sqrt(headDim);

        // Transpose to frame-major [frames][hidden] for per-frame LayerNorm/Linear ops.
        var x = new float[frames][];
        for (int t = 0; t < frames; t++)
        {
            x[t] = new float[hidden];
            for (int c = 0; c < hidden; c++) x[t][c] = channelMajor[c][t];
        }

        foreach (var layer in layers)
        {
            var normed = new float[frames][];
            for (int t = 0; t < frames; t++) normed[t] = LayerNorm(x[t], layer.Norm1Weight, layer.Norm1Bias, MimiCodecDecoderWeights.NormEps);

            var q = new float[frames][];
            var k = new float[frames][];
            var v = new float[frames][];
            for (int t = 0; t < frames; t++)
            {
                q[t] = DenseKernels.LinearNoBias(normed[t], layer.QWeight, hidden, hidden);
                k[t] = DenseKernels.LinearNoBias(normed[t], layer.KWeight, hidden, hidden);
                v[t] = DenseKernels.LinearNoBias(normed[t], layer.VWeight, hidden, hidden);
            }

            var attnOut = new float[frames][];
            for (int t = 0; t < frames; t++) attnOut[t] = new float[hidden];
            for (int h = 0; h < numHeads; h++)
            {
                int off = h * headDim;
                for (int tq = 0; tq <= frames - 1; tq++)
                {
                    var scores = new float[tq + 1]; // causal: only attend to <= tq
                    for (int tk = 0; tk <= tq; tk++)
                    {
                        float dot = 0f;
                        for (int d = 0; d < headDim; d++) dot += q[tq][off + d] * k[tk][off + d];
                        scores[tk] = dot * scale;
                    }
                    DenseKernels.SoftmaxInPlace(scores);
                    for (int tk = 0; tk <= tq; tk++)
                    {
                        float p = scores[tk];
                        for (int d = 0; d < headDim; d++) attnOut[tq][off + d] += p * v[tk][off + d];
                    }
                }
            }

            var projected = new float[frames][];
            for (int t = 0; t < frames; t++)
            {
                var o = DenseKernels.LinearNoBias(attnOut[t], layer.OutWeight, hidden, hidden);
                for (int c = 0; c < hidden; c++) o[c] *= layer.LayerScale1[c];
                projected[t] = o;
            }
            for (int t = 0; t < frames; t++)
                for (int c = 0; c < hidden; c++) x[t][c] += projected[t][c];

            var ffnNormed = new float[frames][];
            for (int t = 0; t < frames; t++) ffnNormed[t] = LayerNorm(x[t], layer.Norm2Weight, layer.Norm2Bias, MimiCodecDecoderWeights.NormEps);
            int intermediate = MimiCodecDecoderWeights.IntermediateSize;
            for (int t = 0; t < frames; t++)
            {
                var h1 = DenseKernels.LinearNoBias(ffnNormed[t], layer.Linear1Weight, hidden, intermediate);
                GeluErfInPlace(h1);
                var h2 = DenseKernels.LinearNoBias(h1, layer.Linear2Weight, intermediate, hidden);
                for (int c = 0; c < hidden; c++) x[t][c] += h2[c] * layer.LayerScale2[c];
            }
        }

        // Back to channel-major.
        var output = new float[hidden][];
        for (int c = 0; c < hidden; c++) output[c] = new float[frames];
        for (int t = 0; t < frames; t++)
            for (int c = 0; c < hidden; c++) output[c][t] = x[t][c];
        return output;
    }

    internal static float[][] SeanetResidualUnit(float[][] x, int channels, int hiddenChannels, MimiCodecResidualUnitWeights w)
    {
        var h = (float[][])x.Clone();
        for (int c = 0; c < channels; c++) h[c] = (float[])x[c].Clone();
        Elu(h);
        h = CausalConv1d(h, channels, hiddenChannels, w.Conv1Weight, w.Conv1Bias, kernel: 3, dilation: 1);
        Elu(h);
        h = CausalConv1d(h, hiddenChannels, channels, w.Conv2Weight, w.Conv2Bias, kernel: 1, dilation: 1);

        var output = new float[channels][];
        for (int c = 0; c < channels; c++)
        {
            output[c] = new float[x[c].Length];
            for (int t = 0; t < x[c].Length; t++) output[c][t] = x[c][t] + h[c][t];
        }
        return output;
    }

    /// <summary>Real strided causal Conv1d (ONE-SHOT/non-streaming): left-zero-pads by the real
    /// `history_frames = max(0, (kernel-1)*dilation+1-stride)` amount -- the same real quantity
    /// the reference's real streaming `StreamingConv1dState` lazily zero-initializes on its FIRST
    /// call, so a one-shot full-sequence call with this same left-pad is mathematically identical
    /// to the streaming path's first call (not a guess -- this is the exact convention
    /// <see cref="CausalConv1d"/>'s own doc history already established for the stride=1 case,
    /// generalized here for the encoder's real stride&gt;1 downsample convs).</summary>
    internal static float[][] CausalConv1dStrided(float[][] input, int inChannels, int outChannels, float[] weight, float[] bias, int kernel, int stride, int dilation = 1)
    {
        int frames = input[0].Length;
        int padLeft = Math.Max(0, (kernel - 1) * dilation + 1 - stride);
        var padded = new float[inChannels][];
        for (int c = 0; c < inChannels; c++)
        {
            padded[c] = new float[frames + padLeft];
            Array.Copy(input[c], 0, padded[c], padLeft, frames);
        }
        int paddedFrames = frames + padLeft;
        int outFrames = (paddedFrames - (kernel - 1) * dilation - 1) / stride + 1;

        var output = new float[outChannels][];
        for (int oc = 0; oc < outChannels; oc++)
        {
            output[oc] = new float[outFrames];
            int wBaseOc = oc * inChannels * kernel;
            for (int t = 0; t < outFrames; t++)
            {
                float sum = bias[oc];
                int start = t * stride;
                for (int ic = 0; ic < inChannels; ic++)
                {
                    int wBase = wBaseOc + ic * kernel;
                    var row = padded[ic];
                    for (int kk = 0; kk < kernel; kk++) sum += weight[wBase + kk] * row[start + kk * dilation];
                }
                output[oc][t] = sum;
            }
        }
        return output;
    }

    internal static float[][] CausalConv1d(float[][] input, int inChannels, int outChannels, float[] weight, float[] bias, int kernel, int dilation)
    {
        int frames = input[0].Length;
        int padLeft = (kernel - 1) * dilation;
        var padded = new float[inChannels][];
        for (int c = 0; c < inChannels; c++)
        {
            padded[c] = new float[frames + padLeft];
            Array.Copy(input[c], 0, padded[c], padLeft, frames);
        }

        var output = new float[outChannels][];
        for (int oc = 0; oc < outChannels; oc++)
        {
            output[oc] = new float[frames];
            int wBaseOc = oc * inChannels * kernel;
            for (int t = 0; t < frames; t++)
            {
                float sum = bias[oc];
                for (int ic = 0; ic < inChannels; ic++)
                {
                    int wBase = wBaseOc + ic * kernel;
                    var row = padded[ic];
                    for (int kk = 0; kk < kernel; kk++) sum += weight[wBase + kk] * row[t + kk * dilation];
                }
                output[oc][t] = sum;
            }
        }
        return output;
    }

    private static float[][] ConvTranspose1d(float[][] input, int inChannels, int outChannels, float[] weight, float[] bias, int kernel, int stride)
    {
        int inLen = input[0].Length;
        int outLen = (inLen - 1) * stride + kernel;
        var output = new float[outChannels][];
        for (int oc = 0; oc < outChannels; oc++)
        {
            output[oc] = new float[outLen];
            for (int o = 0; o < outLen; o++) output[oc][o] = bias[oc];
        }
        for (int ic = 0; ic < inChannels; ic++)
        {
            var inRow = input[ic];
            int wBaseIc = ic * outChannels * kernel;
            for (int i = 0; i < inLen; i++)
            {
                float v = inRow[i];
                if (v == 0f) continue;
                int baseOut = i * stride;
                for (int k = 0; k < kernel; k++)
                {
                    int o = baseOut + k;
                    for (int oc = 0; oc < outChannels; oc++)
                        output[oc][o] += weight[wBaseIc + oc * kernel + k] * v;
                }
            }
        }
        return output;
    }

    /// <summary>Depthwise ConvTranspose1d: one filter per channel, weight `[channels][kernel]`.</summary>
    private static float[][] DepthwiseConvTranspose1d(float[][] input, int channels, float[] weight, float[] bias, int kernel, int stride)
    {
        int inLen = input[0].Length;
        int outLen = (inLen - 1) * stride + kernel;
        var output = new float[channels][];
        for (int c = 0; c < channels; c++)
        {
            output[c] = new float[outLen];
            float b = bias[c];
            for (int o = 0; o < outLen; o++) output[c][o] = b;
            var inRow = input[c];
            int wBase = c * kernel;
            for (int i = 0; i < inLen; i++)
            {
                float v = inRow[i];
                if (v == 0f) continue;
                int baseOut = i * stride;
                for (int k = 0; k < kernel; k++) output[c][baseOut + k] += weight[wBase + k] * v;
            }
        }
        return output;
    }

    internal static float[] LayerNorm(float[] x, float[] weight, float[] bias, float eps)
    {
        double mean = 0;
        for (int i = 0; i < x.Length; i++) mean += x[i];
        mean /= x.Length;
        double variance = 0;
        for (int i = 0; i < x.Length; i++) { double d = x[i] - mean; variance += d * d; }
        variance /= x.Length;
        float invStd = (float)(1.0 / Math.Sqrt(variance + eps));
        var output = new float[x.Length];
        for (int i = 0; i < x.Length; i++) output[i] = (float)((x[i] - mean) * invStd) * weight[i] + bias[i];
        return output;
    }

    internal static void Elu(float[][] channelMajor)
    {
        for (int c = 0; c < channelMajor.Length; c++)
        {
            var row = channelMajor[c];
            for (int t = 0; t < row.Length; t++)
            {
                float v = row[t];
                row[t] = v > 0f ? v : MathF.Exp(v) - 1f;
            }
        }
    }

    internal static void GeluErfInPlace(float[] x)
    {
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            x[i] = 0.5f * v * (1f + Erf(v * 0.70710678f));
        }
    }

    private static float Erf(float x)
    {
        // Abramowitz-Stegun approximation, same formula used elsewhere in this codebase for exact-erf GELU.
        float sign = x < 0 ? -1f : 1f;
        x = MathF.Abs(x);
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float t = 1f / (1f + p * x);
        float y = 1f - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);
        return sign * y;
    }
}
