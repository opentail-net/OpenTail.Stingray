using System;
using System.Threading.Tasks;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Real batch (whole-utterance) forward pass for the Qwen3-TTS 12Hz codec decoder's own 8-layer
/// transformer, transcribed directly from the real, local
/// `examples/qwentts.cpp/src/tokenizer-transformer.h`'s `tok_trans_forward`/
/// `tok_trans_layer_forward` -- see <see cref="QwenTtsCodecTransformerWeights"/>'s doc comment
/// for the full architecture derivation.
///
/// <para>Real per-layer math: pre-RMSNorm -&gt; full MHA (NOT GQA, no QK-norm) with NEOX RoPE
/// (theta=10000) -&gt; causal SLIDING-WINDOW attention (window=72: query q attends to keys in
/// <c>[max(0,q-71), q]</c>, not the whole causal prefix) -&gt; o_proj -&gt; per-channel LayerScale
/// multiply -&gt; residual add -&gt; pre-RMSNorm -&gt; SwiGLU FFN -&gt; per-channel LayerScale
/// multiply -&gt; residual add. The whole utterance (all T frames) is processed in one batched
/// pass -- there is no autoregressive dependency here (unlike the Talker/Code Predictor), since
/// the codec DECODES a complete, already-known code sequence.</para>
/// </summary>
public static class QwenTtsCodecTransformer
{
    /// <summary>Real forward: latent[T][1024] (post RVQ decode + pre-conv) -&gt; latent[T][1024] (pre-ConvNeXt-upsample), via `input_proj` -&gt; 8 layers -&gt; final RMSNorm -&gt; `output_proj`.</summary>
    public static float[][] Forward(QwenTtsCodecTransformerWeights w, float[][] latentIn)
    {
        int t = latentIn.Length;
        var h = new float[t][];
        for (int i = 0; i < t; i++) h[i] = LinearWithBias(latentIn[i], w.InputProjWeight, w.InputProjBias, w.LatentDim, w.HiddenSize);

        foreach (var layer in w.Layers)
            h = Layer(h, layer, w);

        var normed = new float[t][];
        Parallel.For(0, t, i => normed[i] = RmsNorm(h[i], w.NormWeight, w.RmsNormEps));

        var output = new float[t][];
        Parallel.For(0, t, i => output[i] = LinearWithBias(normed[i], w.OutputProjWeight, w.OutputProjBias, w.HiddenSize, w.LatentDim));
        return output;
    }

    private static float[][] Layer(float[][] x, QwenTtsCodecTransformerLayerWeights lw, QwenTtsCodecTransformerWeights w)
    {
        int t = x.Length;
        int dim = w.HiddenSize;
        int nHeads = w.NumHeads;
        int headDim = w.HeadDim;
        int qkvDim = nHeads * headDim; // full MHA: q/k/v all this width

        var normed1 = new float[t][];
        Parallel.For(0, t, i => normed1[i] = RmsNorm(x[i], lw.AttnNormWeight, w.RmsNormEps));

        var q = new float[t][];
        var k = new float[t][];
        var v = new float[t][];
        Parallel.For(0, t, i =>
        {
            q[i] = LinearNoBias(normed1[i], lw.QWeight, dim, qkvDim);
            k[i] = LinearNoBias(normed1[i], lw.KWeight, dim, qkvDim);
            v[i] = LinearNoBias(normed1[i], lw.VWeight, dim, qkvDim);
        });

        for (int i = 0; i < t; i++)
        {
            ApplyRopeNeox(q[i], nHeads, headDim, i, w.RopeTheta);
            ApplyRopeNeox(k[i], nHeads, headDim, i, w.RopeTheta);
        }

        var context = new float[t][];
        for (int i = 0; i < t; i++) context[i] = new float[qkvDim];

        float scale = 1f / MathF.Sqrt(headDim);
        int window = w.SlidingWindow;
        Parallel.For(0, nHeads, h =>
        {
            int off = h * headDim;
            var scores = new float[t];
            for (int i = 0; i < t; i++)
            {
                int kMin = Math.Max(0, i - window + 1);
                for (int j = kMin; j <= i; j++)
                    scores[j] = Dot(q[i], k[j], off, headDim) * scale;
                SoftmaxRange(scores, kMin, i);

                var ctxSpan = context[i].AsSpan(off, headDim);
                for (int j = kMin; j <= i; j++)
                    for (int d = 0; d < headDim; d++) ctxSpan[d] += scores[j] * v[j][off + d];
            }
        });

        var attnOut = new float[t][];
        Parallel.For(0, t, i => attnOut[i] = LinearNoBias(context[i], lw.OWeight, qkvDim, dim));

        var afterAttn = new float[t][];
        Parallel.For(0, t, i =>
        {
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = x[i][d] + attnOut[i][d] * lw.AttnScale[d];
            afterAttn[i] = row;
        });

        var normed2 = new float[t][];
        Parallel.For(0, t, i => normed2[i] = RmsNorm(afterAttn[i], lw.FfnNormWeight, w.RmsNormEps));

        var output = new float[t][];
        Parallel.For(0, t, i =>
        {
            var gate = LinearNoBias(normed2[i], lw.GateWeight, dim, w.IntermediateSize);
            var up = LinearNoBias(normed2[i], lw.UpWeight, dim, w.IntermediateSize);
            for (int d = 0; d < w.IntermediateSize; d++) gate[d] = Silu(gate[d]) * up[d];
            var ffnOut = LinearNoBias(gate, lw.DownWeight, w.IntermediateSize, dim);

            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = afterAttn[i][d] + ffnOut[d] * lw.FfnScale[d];
            output[i] = row;
        });
        return output;
    }

    /// <summary>Real NEOX RoPE (half-split rotation), matching the real `ggml_rope_ext(..., GGML_ROPE_TYPE_NEOX, ...)` call.</summary>
    private static void ApplyRopeNeox(float[] vec, int nHeads, int headDim, int position, float freqBase)
    {
        int half = headDim / 2;
        for (int h = 0; h < nHeads; h++)
        {
            int off = h * headDim;
            for (int i = 0; i < half; i++)
            {
                float freq = 1f / MathF.Pow(freqBase, 2f * i / headDim);
                float angle = position * freq;
                float cos = MathF.Cos(angle), sin = MathF.Sin(angle);
                int idx0 = off + i;
                int idx1 = off + i + half;
                float a = vec[idx0], b = vec[idx1];
                vec[idx0] = a * cos - b * sin;
                vec[idx1] = a * sin + b * cos;
            }
        }
    }

    private static float Silu(float x) => x / (1f + MathF.Exp(-x));

    private static float Dot(float[] a, float[] b, int offset, int len)
    {
        float sum = 0f;
        for (int i = 0; i < len; i++) sum += a[offset + i] * b[offset + i];
        return sum;
    }

    private static unsafe float[] LinearNoBias(float[] input, float[] weight, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* wp = weight, xp = input, op = output)
        {
            SimdKernels.MatVecF32(op, wp, xp, outDim, inDim);
        }
        return output;
    }

    private static float[] LinearWithBias(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = LinearNoBias(input, weight, inDim, outDim);
        for (int d = 0; d < outDim; d++) output[d] += bias[d];
        return output;
    }

    private static float[] RmsNorm(float[] x, float[] weight, float eps)
    {
        int n = x.Length;
        float sumSq = 0f;
        for (int i = 0; i < n; i++) sumSq += x[i] * x[i];
        float invRms = 1f / MathF.Sqrt(sumSq / n + eps);
        var output = new float[n];
        for (int i = 0; i < n; i++) output[i] = x[i] * invRms * weight[i];
        return output;
    }

    private static void SoftmaxRange(float[] scores, int start, int end)
    {
        float max = float.NegativeInfinity;
        for (int i = start; i <= end; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = start; i <= end; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = start; i <= end; i++) scores[i] *= invSum;
    }
}
