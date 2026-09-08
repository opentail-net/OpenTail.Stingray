using System.Numerics.Tensors;
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.OmniVoice;

/// <summary>
/// Real, self-contained (non-`IForwardPass`) forward pass for OmniVoice's own MaskGIT transformer,
/// ported from `generator.cpp`'s `decoder_layer` (not guessed): standard Qwen3-style GQA + RoPE-
/// NEOX + per-head q/k RMSNorm + bias-free SwiGLU MLP, but FULLY NON-CAUSAL -- every position
/// attends to every other position with no masking at all (confirmed real: the reference's own
/// `attention_mask` zeroes the WHOLE valid `[0,len)x[0,len)` block, no triangular restriction),
/// so this implementation simply omits any masking rather than replicating the reference's
/// padding-capacity scheme (this port has no fixed-capacity graph to pad).
/// </summary>
public static class OmniVoiceMaskGitForward
{
    /// <summary>Runs the full `NumLayers`-deep transformer over one already-built
    /// `[seqLen][HiddenDim]` embedding sequence (frame-major), returning the FINAL RMSNorm'd
    /// hidden states. Positions are real sequential `0..seqLen-1` (matches the reference's own
    /// `position_host[i]=i`).</summary>
    public static float[][] Forward(OmniVoiceMaskGitWeights w, float[][] embeddings)
    {
        int seqLen = embeddings.Length;
        var hidden = embeddings;
        for (int l = 0; l < OmniVoiceMaskGitWeights.NumLayers; l++)
            hidden = Layer(w.Layers[l], hidden, seqLen);

        var output = new float[seqLen][];
        Parallel.For(0, seqLen, t => output[t] = RmsNorm(hidden[t], w.FinalNorm));
        return output;
    }

    private static float[][] Layer(OmniVoiceMaskGitLayerWeights w, float[][] input, int seqLen)
    {
        const int hidden = OmniVoiceMaskGitWeights.HiddenDim;
        const int numHeads = OmniVoiceMaskGitWeights.NumHeads;
        const int numKvHeads = OmniVoiceMaskGitWeights.NumKvHeads;
        const int headDim = OmniVoiceMaskGitWeights.HeadDim;
        int kvRepeats = numHeads / numKvHeads;

        var xNorm = new float[seqLen][];
        Parallel.For(0, seqLen, t => xNorm[t] = RmsNorm(input[t], w.InputNorm));

        // Real per-head q/k RMSNorm + RoPE-NEOX, applied per position/head in parallel across tokens.
        var q = new float[seqLen][][]; // [t][head][headDim]
        var k = new float[seqLen][][];
        var v = new float[seqLen][][];
        Parallel.For(0, seqLen, t =>
        {
            var qFlat = DenseKernels.LinearNoBias(xNorm[t], w.QProj, hidden, numHeads * headDim);
            var kFlat = DenseKernels.LinearNoBias(xNorm[t], w.KProj, hidden, numKvHeads * headDim);
            var vFlat = DenseKernels.LinearNoBias(xNorm[t], w.VProj, hidden, numKvHeads * headDim);
            var qHeads = new float[numHeads][];
            for (int h = 0; h < numHeads; h++)
            {
                var head = new float[headDim];
                Array.Copy(qFlat, h * headDim, head, 0, headDim);
                head = RmsNorm(head, w.QNorm);
                RopeNeoxInPlace(head, t, headDim);
                qHeads[h] = head;
            }
            q[t] = qHeads;

            var kHeads = new float[numKvHeads][];
            var vHeads = new float[numKvHeads][];
            for (int h = 0; h < numKvHeads; h++)
            {
                var head = new float[headDim];
                Array.Copy(kFlat, h * headDim, head, 0, headDim);
                head = RmsNorm(head, w.KNorm);
                RopeNeoxInPlace(head, t, headDim);
                kHeads[h] = head;
                var vHead = new float[headDim];
                Array.Copy(vFlat, h * headDim, vHead, 0, headDim);
                vHeads[h] = vHead;
            }
            k[t] = kHeads;
            v[t] = vHeads;
        });

        // Real full (non-causal) scaled-dot-product attention per head, GQA-repeated kv in parallel across heads.
        float scale = 1f / MathF.Sqrt(headDim);
        var context = new float[seqLen][]; // [t][numHeads*headDim]
        for (int t = 0; t < seqLen; t++) context[t] = new float[numHeads * headDim];

        Parallel.For(0, numHeads, h =>
        {
            int kvHead = h / kvRepeats;
            int baseOff = h * headDim;
            var scores = new float[seqLen];
            for (int tq = 0; tq < seqLen; tq++)
            {
                var qv = (ReadOnlySpan<float>)q[tq][h];
                float maxScore = float.NegativeInfinity;
                for (int tk = 0; tk < seqLen; tk++)
                {
                    float dot = TensorPrimitives.Dot(qv, k[tk][kvHead]) * scale;
                    scores[tk] = dot;
                    if (dot > maxScore) maxScore = dot;
                }
                double sum = 0;
                for (int tk = 0; tk < seqLen; tk++)
                {
                    float exp = MathF.Exp(scores[tk] - maxScore);
                    scores[tk] = exp;
                    sum += exp;
                }
                var outHead = context[tq].AsSpan(baseOff, headDim);
                for (int tk = 0; tk < seqLen; tk++)
                {
                    float p = (float)(scores[tk] / sum);
                    if (p == 0f) continue;
                    TensorPrimitives.MultiplyAdd((ReadOnlySpan<float>)v[tk][kvHead], p, outHead, outHead);
                }
            }
        });

        var afterAttn = new float[seqLen][];
        Parallel.For(0, seqLen, t =>
        {
            var o = DenseKernels.LinearNoBias(context[t], w.OProj, numHeads * headDim, hidden);
            var row = new float[hidden];
            TensorPrimitives.Add((ReadOnlySpan<float>)input[t], o, row);
            afterAttn[t] = row;
        });

        var output = new float[seqLen][];
        Parallel.For(0, seqLen, t =>
        {
            var ffNorm = RmsNorm(afterAttn[t], w.PostNorm);
            var gate = DenseKernels.LinearNoBias(ffNorm, w.GateProj, hidden, OmniVoiceMaskGitWeights.FfDim);
            var up = DenseKernels.LinearNoBias(ffNorm, w.UpProj, hidden, OmniVoiceMaskGitWeights.FfDim);
            DenseKernels.SiluInPlace(gate);
            TensorPrimitives.Multiply((ReadOnlySpan<float>)gate, up, gate);
            var down = DenseKernels.LinearNoBias(gate, w.DownProj, OmniVoiceMaskGitWeights.FfDim, hidden);
            var row = new float[hidden];
            TensorPrimitives.Add((ReadOnlySpan<float>)afterAttn[t], down, row);
            output[t] = row;
        });
        return output;
    }

    /// <summary>Real RoPE-NEOX (rotate-half convention, matching `GGML_ROPE_TYPE_NEOX`): pairs
    /// `(i, i+headDim/2)` for `i` in `[0, headDim/2)`, `theta_i = ropeTheta^(-2i/headDim)`.</summary>
    private static void RopeNeoxInPlace(float[] head, int position, int headDim)
    {
        int half = headDim / 2;
        for (int i = 0; i < half; i++)
        {
            double theta = Math.Pow(OmniVoiceMaskGitWeights.RopeTheta, -2.0 * i / headDim);
            double angle = position * theta;
            float cos = (float)Math.Cos(angle), sin = (float)Math.Sin(angle);
            float a = head[i], b = head[i + half];
            head[i] = a * cos - b * sin;
            head[i + half] = a * sin + b * cos;
        }
    }

    private static float[] RmsNorm(float[] x, float[] weight)
    {
        double sumSq = TensorPrimitives.SumOfSquares((ReadOnlySpan<float>)x);
        float invRms = (float)(1.0 / Math.Sqrt(sumSq / x.Length + OmniVoiceMaskGitWeights.RmsNormEps));
        var output = new float[x.Length];
        TensorPrimitives.Multiply((ReadOnlySpan<float>)x, invRms, output);
        TensorPrimitives.Multiply((ReadOnlySpan<float>)output, weight, output);
        return output;
    }

}
