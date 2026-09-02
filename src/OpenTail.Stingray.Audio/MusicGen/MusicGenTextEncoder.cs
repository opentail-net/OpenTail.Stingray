
namespace OpenTail.Stingray.Audio.MusicGen;

/// <summary>
/// Real stock-`t5-base` encoder forward pass for MusicGen's text conditioning, transcribed from
/// the real `transformers` `modeling_t5.py` -- same real T5-specific quirks as
/// <see cref="Parler.T5Encoder"/> (no attention 1/sqrt(headDim) scaling, RMSNorm-only
/// `T5LayerNorm` with no bias/mean-subtraction, relative position bias computed once from block
/// 0 and shared across all layers), but a DIFFERENT, non-gated FFN: real `T5DenseActDense` is
/// `wo(relu(wi(x)))` -- one `wi` matrix, plain ReLU, no gating multiply. See
/// <see cref="MusicGenTextEncoderWeights"/>'s doc comment for why this must load the real stock
/// `t5-base` checkpoint rather than reusing Parler's gated-GELU flan-t5.
/// </summary>
public static class MusicGenTextEncoder
{
    /// <summary>Runs the full T5 encoder. `tokenIds` -&gt; embed -&gt; 12x T5 layer (self-attn + plain ReLU FFN) -&gt; final RMSNorm. Returns [t][DModel].</summary>
    public static float[][] Forward(MusicGenTextEncoderWeights w, int[] tokenIds)
    {
        int t = tokenIds.Length;
        int dim = MusicGenConfig.TextDModel;
        var x = new float[t * dim];
        for (int i = 0; i < t; i++)
            Array.Copy(w.SharedEmbedding, (long)tokenIds[i] * dim, x, (long)i * dim, dim);

        var positionBias = ComputeRelativePositionBias(w, t);

        foreach (var layer in w.Layers)
            x = Layer(x, layer, t, positionBias);

        var flatOut = new float[t * dim];
        Parallel.For(0, t, i => T5LayerNorm(x.AsSpan(i * dim, dim), w.FinalLayerNormWeight, flatOut.AsSpan(i * dim, dim)));

        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            output[i] = new float[dim];
            Array.Copy(flatOut, i * dim, output[i], 0, dim);
        }
        return output;
    }

    private static float[] Layer(float[] x, MusicGenTextLayerWeights lw, int t, float[][,] positionBias)
    {
        int dim = MusicGenConfig.TextDModel;
        var normed1 = new float[t * dim];
        Parallel.For(0, t, i => T5LayerNorm(x.AsSpan(i * dim, dim), lw.SelfAttnLayerNormWeight, normed1.AsSpan(i * dim, dim)));

        var attnOut = SelfAttention(normed1, lw, t, positionBias);

        var afterAttn = new float[t * dim];
        TensorPrimitives.Add(x, attnOut, afterAttn);

        var normed2 = new float[t * dim];
        Parallel.For(0, t, i => T5LayerNorm(afterAttn.AsSpan(i * dim, dim), lw.FfnLayerNormWeight, normed2.AsSpan(i * dim, dim)));

        var ffnOut = Ffn(normed2, lw, t);

        var output = new float[t * dim];
        TensorPrimitives.Add(afterAttn, ffnOut, output);
        return output;
    }

    private static unsafe float[] SelfAttention(float[] x, MusicGenTextLayerWeights lw, int t, float[][,] positionBias)
    {
        int dim = MusicGenConfig.TextDModel;
        int nHeads = MusicGenConfig.TextNumHeads;
        int dKv = MusicGenConfig.TextDKv;
        int qkvDim = nHeads * dKv;

        var q = new float[t * qkvDim];
        var k = new float[t * qkvDim];
        var v = new float[t * qkvDim];
        fixed (float* xp = x, qp = q, kp = k, vp = v)
        {
            lw.SelfAttnQWeight.MatMul(xp, t, qp);
            lw.SelfAttnKWeight.MatMul(xp, t, kp);
            lw.SelfAttnVWeight.MatMul(xp, t, vp);
        }

        var context = new float[t * qkvDim];
        Parallel.For(0, nHeads, h =>
        {
            int off = h * dKv;
            var scores = new float[t];
            for (int i = 0; i < t; i++)
            {
                for (int j = 0; j < t; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < dKv; d++) dot += q[i * qkvDim + off + d] * k[j * qkvDim + off + d];
                    scores[j] = dot + positionBias[h][i, j]; // NO scaling -- real T5 quirk
                }
                SoftmaxInPlace(scores);

                var ctxSpan = context.AsSpan(i * qkvDim + off, dKv);
                for (int j = 0; j < t; j++)
                {
                    float s = scores[j];
                    var vSpan = v.AsSpan(j * qkvDim + off, dKv);
                    for (int d = 0; d < dKv; d++) ctxSpan[d] += s * vSpan[d];
                }
            }
        });

        var output = new float[t * dim];
        fixed (float* cp = context, op = output)
            lw.SelfAttnOWeight.MatMul(cp, t, op);
        return output;
    }

    /// <summary>Real T5DenseActDense: `wo(relu(wi(x)))`, no biases anywhere. Plain, non-gated -- do not add a second wi matrix here.</summary>
    private static unsafe float[] Ffn(float[] x, MusicGenTextLayerWeights lw, int t)
    {
        int dim = MusicGenConfig.TextDModel;
        int ff = MusicGenConfig.TextDFf;
        var hidden = new float[t * ff];
        fixed (float* xp = x, hp = hidden)
            lw.FfnWiWeight.MatMul(xp, t, hp);

        for (int i = 0; i < hidden.Length; i++)
            hidden[i] = MathF.Max(0f, hidden[i]);

        var output = new float[t * dim];
        fixed (float* hp = hidden, op = output)
            lw.FfnWoWeight.MatMul(hp, t, op);
        return output;
    }

    private static float[][,] ComputeRelativePositionBias(MusicGenTextEncoderWeights w, int t)
    {
        var bias = new float[MusicGenConfig.TextNumHeads][,];
        for (int h = 0; h < MusicGenConfig.TextNumHeads; h++) bias[h] = new float[t, t];

        for (int qi = 0; qi < t; qi++)
        {
            for (int kj = 0; kj < t; kj++)
            {
                int relPos = kj - qi;
                int bucket = RelativePositionBucket(relPos, bidirectional: true,
                    MusicGenConfig.TextRelativeAttentionNumBuckets, MusicGenConfig.TextRelativeAttentionMaxDistance);
                for (int h = 0; h < MusicGenConfig.TextNumHeads; h++)
                    bias[h][qi, kj] = w.RelativeAttentionBias[bucket * MusicGenConfig.TextNumHeads + h];
            }
        }
        return bias;
    }

    /// <summary>Real `_relative_position_bucket`, transcribed exactly from `transformers/models/t5/modeling_t5.py`.</summary>
    private static int RelativePositionBucket(int relativePosition, bool bidirectional, int numBuckets, int maxDistance)
    {
        int relativeBuckets = 0;
        if (bidirectional)
        {
            numBuckets /= 2;
            relativeBuckets += relativePosition > 0 ? numBuckets : 0;
            relativePosition = Math.Abs(relativePosition);
        }
        else
        {
            relativePosition = -Math.Min(relativePosition, 0);
        }

        int maxExact = numBuckets / 2;
        bool isSmall = relativePosition < maxExact;

        int relativePositionIfLarge = maxExact + (int)(
            MathF.Log(relativePosition / (float)maxExact)
            / MathF.Log(maxDistance / (float)maxExact)
            * (numBuckets - maxExact));
        relativePositionIfLarge = Math.Min(relativePositionIfLarge, numBuckets - 1);

        relativeBuckets += isSmall ? relativePosition : relativePositionIfLarge;
        return relativeBuckets;
    }

    private static void T5LayerNorm(ReadOnlySpan<float> x, float[] weight, Span<float> output, float eps = 1e-6f)
    {
        int n = x.Length;
        float sumSq = 0f;
        for (int i = 0; i < n; i++) sumSq += x[i] * x[i];
        float invRms = 1f / MathF.Sqrt(sumSq / n + eps);
        for (int i = 0; i < n; i++) output[i] = x[i] * invRms * weight[i];
    }

    private static void SoftmaxInPlace(float[] scores)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = 0; i < scores.Length; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = 0; i < scores.Length; i++) scores[i] *= invSum;
    }
}
