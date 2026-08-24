using System;
using System.Threading.Tasks;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Parler;

/// <summary>
/// Real T5 encoder forward pass, transcribed directly from the real `transformers` Python
/// package's `modeling_t5.py` (`T5Attention`/`T5LayerNorm`/`T5DenseGatedActDense`), fetched from
/// the locally-installed `transformers` package, not re-derived from memory -- see
/// <see cref="T5EncoderWeights"/>'s doc comment and docs/audio-review-progress.md's Parler-TTS
/// section for the full derivation.
///
/// <para><b>Three real, easy-to-get-wrong T5-specific quirks, confirmed from source, do not
/// "fix" any of these back to a standard-transformer assumption</b>:
/// (1) Attention scores are NOT scaled by <c>1/sqrt(head_dim)</c> -- T5 omits this scaling
/// entirely (confirmed: `scores = torch.matmul(query_states, key_states.transpose(3, 2))`, no
/// division anywhere in the real source).
/// (2) `T5LayerNorm` is a pure RMSNorm variant: <c>x * rsqrt(mean(x^2) + eps) * weight</c> -- NO
/// bias, NO mean-subtraction (confirmed from the real class's own doc comment: "No bias and no
/// subtraction of mean").
/// (3) The relative position bias is computed ONCE, using ONLY block 0's
/// <c>relative_attention_bias</c> table, and the SAME bias tensor is reused/added into every
/// subsequent layer's attention scores -- it is NOT recomputed per layer (confirmed: only block
/// 0 has a real `relative_attention_bias.weight` tensor in the checkpoint; `compute_bias` is
/// called once and the result threaded through as `position_bias` in the real source).</para>
/// </summary>
public static class T5Encoder
{
    /// <summary>Runs the full T5 encoder. `tokenIds` -&gt; embed -&gt; 24x T5 layer (self-attn + gated-GELU FFN) -&gt; final RMSNorm. Returns [T, DModel].</summary>
    public static float[][] Forward(T5EncoderWeights w, int[] tokenIds)
    {
        int t = tokenIds.Length;
        var x = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[T5EncoderWeights.DModel];
            Array.Copy(w.SharedEmbedding, (long)tokenIds[i] * T5EncoderWeights.DModel, row, 0, T5EncoderWeights.DModel);
            x[i] = row;
        }

        var positionBias = ComputeRelativePositionBias(w, t);

        foreach (var layer in w.Layers)
            x = T5Layer(x, layer, positionBias);

        var output = new float[t][];
        Parallel.For(0, t, i => output[i] = T5LayerNorm(x[i], w.FinalLayerNormWeight));
        return output;
    }

    private static float[][] T5Layer(float[][] x, T5LayerWeights lw, float[][,] positionBias)
    {
        int t = x.Length;
        var normed1 = new float[t][];
        Parallel.For(0, t, i => normed1[i] = T5LayerNorm(x[i], lw.SelfAttnLayerNormWeight));

        var attnOut = SelfAttention(normed1, lw, positionBias);

        var afterAttn = new float[t][];
        Parallel.For(0, t, i =>
        {
            var row = new float[T5EncoderWeights.DModel];
            for (int d = 0; d < T5EncoderWeights.DModel; d++) row[d] = x[i][d] + attnOut[i][d];
            afterAttn[i] = row;
        });

        var normed2 = new float[t][];
        Parallel.For(0, t, i => normed2[i] = T5LayerNorm(afterAttn[i], lw.FfnLayerNormWeight));

        var ffnOut = GatedFfn(normed2, lw);

        var output = new float[t][];
        Parallel.For(0, t, i =>
        {
            var row = new float[T5EncoderWeights.DModel];
            for (int d = 0; d < T5EncoderWeights.DModel; d++) row[d] = afterAttn[i][d] + ffnOut[i][d];
            output[i] = row;
        });
        return output;
    }

    /// <summary>Real T5 self-attention: NO 1/sqrt(headDim) scaling, plus the shared relative position bias added to raw scores before softmax.</summary>
    private static float[][] SelfAttention(float[][] x, T5LayerWeights lw, float[][,] positionBias)
    {
        int t = x.Length;
        int nHeads = T5EncoderWeights.NumHeads;
        int dKv = T5EncoderWeights.DKv;
        int qkvDim = nHeads * dKv; // 1024, equals DModel for this config

        var q = new float[t][];
        var k = new float[t][];
        var v = new float[t][];
        Parallel.For(0, t, i =>
        {
            q[i] = lw.SelfAttnQWeight.MatVec(x[i]);
            k[i] = lw.SelfAttnKWeight.MatVec(x[i]);
            v[i] = lw.SelfAttnVWeight.MatVec(x[i]);
        });

        var context = new float[t][];
        for (int i = 0; i < t; i++) context[i] = new float[qkvDim];

        Parallel.For(0, nHeads, h =>
        {
            int off = h * dKv;
            var scores = new float[t];
            for (int i = 0; i < t; i++)
            {
                for (int j = 0; j < t; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < dKv; d++) dot += q[i][off + d] * k[j][off + d];
                    scores[j] = dot + positionBias[h][i, j]; // NO scaling -- real T5 quirk
                }
                SoftmaxInPlace(scores);

                var ctxSpan = context[i].AsSpan(off, dKv);
                for (int j = 0; j < t; j++)
                    for (int d = 0; d < dKv; d++) ctxSpan[d] += scores[j] * v[j][off + d];
            }
        });

        var output = new float[t][];
        Parallel.For(0, t, i => output[i] = lw.SelfAttnOWeight.MatVec(context[i]));
        return output;
    }

    /// <summary>Real T5DenseGatedActDense: `wo(gelu_new(wi_0(x)) * wi_1(x))`, no biases anywhere.</summary>
    private static float[][] GatedFfn(float[][] x, T5LayerWeights lw)
    {
        int t = x.Length;
        var output = new float[t][];
        Parallel.For(0, t, i =>
        {
            var gate = lw.FfnWi0Weight.MatVec(x[i]);
            var up = lw.FfnWi1Weight.MatVec(x[i]);
            for (int d = 0; d < T5EncoderWeights.DFf; d++) gate[d] = GeluNew(gate[d]) * up[d];
            output[i] = lw.FfnWoWeight.MatVec(gate);
        });
        return output;
    }

    /// <summary>Real "gelu_new" (tanh approximation), matching Parler's real `dense_act_fn=gelu_new` config.</summary>
    private static float GeluNew(float x) =>
        0.5f * x * (1f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));

    /// <summary>Real T5 relative position bucketing + bias lookup, computed once for the whole sequence and shared across all layers. Bidirectional (encoder, not decoder) -- confirmed from the real `compute_bias`/`_relative_position_bucket` source.</summary>
    private static float[][,] ComputeRelativePositionBias(T5EncoderWeights w, int t)
    {
        var bias = new float[T5EncoderWeights.NumHeads][,];
        for (int h = 0; h < T5EncoderWeights.NumHeads; h++) bias[h] = new float[t, t];

        for (int qi = 0; qi < t; qi++)
        {
            for (int kj = 0; kj < t; kj++)
            {
                int relPos = kj - qi;
                int bucket = RelativePositionBucket(relPos, bidirectional: true,
                    T5EncoderWeights.RelativeAttentionNumBuckets, T5EncoderWeights.RelativeAttentionMaxDistance);
                for (int h = 0; h < T5EncoderWeights.NumHeads; h++)
                    bias[h][qi, kj] = w.RelativeAttentionBias[bucket * T5EncoderWeights.NumHeads + h];
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

    /// <summary>Real T5LayerNorm: pure RMSNorm, NO bias, NO mean-subtraction.</summary>
    private static float[] T5LayerNorm(float[] x, float[] weight, float eps = 1e-6f)
    {
        int n = x.Length;
        float sumSq = 0f;
        for (int i = 0; i < n; i++) sumSq += x[i] * x[i];
        float invRms = 1f / MathF.Sqrt(sumSq / n + eps);

        var output = new float[n];
        for (int i = 0; i < n; i++) output[i] = x[i] * invRms * weight[i];
        return output;
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
