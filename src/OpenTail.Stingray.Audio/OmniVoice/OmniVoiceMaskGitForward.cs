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
        for (int t = 0; t < seqLen; t++) output[t] = RmsNorm(hidden[t], w.FinalNorm);
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
        for (int t = 0; t < seqLen; t++) xNorm[t] = RmsNorm(input[t], w.InputNorm);

        // Real per-head q/k RMSNorm + RoPE-NEOX, applied per position/head.
        var q = new float[seqLen][][]; // [t][head][headDim]
        var k = new float[seqLen][][];
        var v = new float[seqLen][][];
        for (int t = 0; t < seqLen; t++)
        {
            var qFlat = DenseKernels.LinearNoBias(xNorm[t], w.QProj, hidden, numHeads * headDim);
            var kFlat = DenseKernels.LinearNoBias(xNorm[t], w.KProj, hidden, numKvHeads * headDim);
            var vFlat = DenseKernels.LinearNoBias(xNorm[t], w.VProj, hidden, numKvHeads * headDim);
            q[t] = new float[numHeads][];
            for (int h = 0; h < numHeads; h++)
            {
                var head = new float[headDim];
                Array.Copy(qFlat, h * headDim, head, 0, headDim);
                head = RmsNorm(head, w.QNorm);
                RopeNeoxInPlace(head, t, headDim);
                q[t][h] = head;
            }
            k[t] = new float[numKvHeads][];
            v[t] = new float[numKvHeads][];
            for (int h = 0; h < numKvHeads; h++)
            {
                var head = new float[headDim];
                Array.Copy(kFlat, h * headDim, head, 0, headDim);
                head = RmsNorm(head, w.KNorm);
                RopeNeoxInPlace(head, t, headDim);
                k[t][h] = head;
                var vHead = new float[headDim];
                Array.Copy(vFlat, h * headDim, vHead, 0, headDim);
                v[t][h] = vHead;
            }
        }

        // Real full (non-causal) scaled-dot-product attention per head, GQA-repeated kv.
        float scale = 1f / MathF.Sqrt(headDim);
        var context = new float[seqLen][]; // [t][numHeads*headDim]
        for (int t = 0; t < seqLen; t++) context[t] = new float[numHeads * headDim];

        var scores = new float[seqLen];
        for (int h = 0; h < numHeads; h++)
        {
            int kvHead = h / kvRepeats;
            for (int tq = 0; tq < seqLen; tq++)
            {
                float maxScore = float.NegativeInfinity;
                for (int tk = 0; tk < seqLen; tk++)
                {
                    float dot = 0f;
                    var qv = q[tq][h]; var kv = k[tk][kvHead];
                    for (int d = 0; d < headDim; d++) dot += qv[d] * kv[d];
                    dot *= scale;
                    scores[tk] = dot;
                    if (dot > maxScore) maxScore = dot;
                }
                double sum = 0;
                for (int tk = 0; tk < seqLen; tk++) { scores[tk] = MathF.Exp(scores[tk] - maxScore); sum += scores[tk]; }
                var outHead = context[tq];
                int baseOff = h * headDim;
                for (int tk = 0; tk < seqLen; tk++)
                {
                    float p = (float)(scores[tk] / sum);
                    if (p == 0f) continue;
                    var vv = v[tk][kvHead];
                    for (int d = 0; d < headDim; d++) outHead[baseOff + d] += p * vv[d];
                }
            }
        }

        var afterAttn = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            var o = DenseKernels.LinearNoBias(context[t], w.OProj, numHeads * headDim, hidden);
            var row = new float[hidden];
            for (int d = 0; d < hidden; d++) row[d] = input[t][d] + o[d];
            afterAttn[t] = row;
        }

        var output = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            var ffNorm = RmsNorm(afterAttn[t], w.PostNorm);
            var gate = DenseKernels.LinearNoBias(ffNorm, w.GateProj, hidden, OmniVoiceMaskGitWeights.FfDim);
            var up = DenseKernels.LinearNoBias(ffNorm, w.UpProj, hidden, OmniVoiceMaskGitWeights.FfDim);
            DenseKernels.SiluInPlace(gate);
            for (int d = 0; d < gate.Length; d++) gate[d] *= up[d];
            var down = DenseKernels.LinearNoBias(gate, w.DownProj, OmniVoiceMaskGitWeights.FfDim, hidden);
            var row = new float[hidden];
            for (int d = 0; d < hidden; d++) row[d] = afterAttn[t][d] + down[d];
            output[t] = row;
        }
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
        double sumSq = 0;
        for (int i = 0; i < x.Length; i++) sumSq += (double)x[i] * x[i];
        float invRms = (float)(1.0 / Math.Sqrt(sumSq / x.Length + OmniVoiceMaskGitWeights.RmsNormEps));
        var output = new float[x.Length];
        for (int i = 0; i < x.Length; i++) output[i] = x[i] * invRms * weight[i];
        return output;
    }

}
