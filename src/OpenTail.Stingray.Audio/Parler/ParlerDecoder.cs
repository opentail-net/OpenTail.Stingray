using System;
using System.Threading.Tasks;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Parler;

/// <summary>
/// Real Parler-TTS decoder forward pass (MusicGen-style causal decoder), transcribed directly
/// from the real `parler_tts` Python package's `ParlerTTSDecoderLayer`/`ParlerTTSDecoder`
/// (`modeling_parler_tts.py`, fetched via `pip download parler-tts --no-deps`) -- see
/// <see cref="ParlerDecoderWeights"/>'s doc comment for the full config derivation.
///
/// <para>Standard pre-LN transformer decoder, genuinely simpler than every other pipeline
/// finished this session: plain LayerNorm (mean-subtract + bias, NOT RMSNorm), full MHA (no
/// GQA), NO RoPE (real precomputed sinusoidal positions, loaded as a table, not a formula), plain
/// GELU FFN (not gated). Per layer: `self_attn_layer_norm(x)` -&gt; causal self-attention -&gt;
/// residual -&gt; `encoder_attn_layer_norm(x)` -&gt; cross-attention to the T5 encoder's output
/// -&gt; residual -&gt; `final_layer_norm(x)` -&gt; `fc2(gelu(fc1(x)))` -&gt; residual.</para>
///
/// <para>Embedding: `sum(embed_tokens[cb][input_ids[cb]] for cb in 0..8)` (9 real codebook
/// streams summed, no scale) + the real precomputed sinusoidal position embedding for that
/// timestep. Output: 9 SEPARATE `lm_heads` projections (not tied), all predicted in parallel per
/// timestep -- unlike Fish Speech's sequential fast-AR codebook expansion, Parler's decoder
/// emits all 9 codebooks' logits directly from the same 24-layer trunk.</para>
/// </summary>
public static class ParlerDecoder
{
    private const int HeadDim = ParlerDecoderWeights.HiddenDim / ParlerDecoderWeights.NumHeads; // 64

    /// <summary>Composes the real input embedding for one timestep: sum of 9 codebook-token lookups + the real sinusoidal position embedding.</summary>
    public static float[] EmbedStep(ParlerDecoderWeights w, int[] codebookTokenIds, int position)
    {
        var emb = new float[ParlerDecoderWeights.HiddenDim];
        for (int cb = 0; cb < ParlerDecoderWeights.NumCodebooks; cb++)
        {
            long row = (long)codebookTokenIds[cb] * ParlerDecoderWeights.HiddenDim;
            var table = w.EmbedTokens[cb];
            for (int d = 0; d < ParlerDecoderWeights.HiddenDim; d++) emb[d] += table[row + d];
        }
        long posRow = (long)position * ParlerDecoderWeights.HiddenDim;
        for (int d = 0; d < ParlerDecoderWeights.HiddenDim; d++) emb[d] += w.EmbedPositions[posRow + d];
        return emb;
    }

    /// <summary>Runs the full decoder trunk over a sequence of already-composed input embeddings, cross-attending to the real T5 encoder output. Returns per-position hidden states [T, HiddenDim] (post final LayerNorm).</summary>
    public static float[][] Forward(ParlerDecoderWeights w, float[][] inputEmbeds, float[][] encoderHidden)
    {
        var x = inputEmbeds;
        foreach (var layer in w.Layers)
            x = DecoderLayer(x, encoderHidden, layer);

        int t = x.Length;
        var output = new float[t][];
        Parallel.For(0, t, i => output[i] = LayerNorm(x[i], w.FinalLayerNormWeight, w.FinalLayerNormBias));
        return output;
    }

    /// <summary>Projects the final hidden states through all 9 real, separate lm_heads. Returns [T][9][OutputVocabSize].</summary>
    public static float[][][] ComputeLogits(ParlerDecoderWeights w, float[][] hidden)
    {
        int t = hidden.Length;
        var result = new float[t][][];
        Parallel.For(0, t, i =>
        {
            result[i] = new float[ParlerDecoderWeights.NumCodebooks][];
            for (int cb = 0; cb < ParlerDecoderWeights.NumCodebooks; cb++)
                result[i][cb] = LinearNoBias(hidden[i], w.LmHeads[cb], ParlerDecoderWeights.HiddenDim, ParlerDecoderWeights.OutputVocabSize);
        });
        return result;
    }

    private static float[][] DecoderLayer(float[][] x, float[][] encoderHidden, ParlerDecoderLayerWeights lw)
    {
        int t = x.Length;

        var normed1 = new float[t][];
        Parallel.For(0, t, i => normed1[i] = LayerNorm(x[i], lw.SelfAttnLayerNormWeight, lw.SelfAttnLayerNormBias));
        var selfAttnOut = SelfAttentionCausal(normed1, lw);

        var afterSelf = new float[t][];
        Parallel.For(0, t, i =>
        {
            var row = new float[ParlerDecoderWeights.HiddenDim];
            for (int d = 0; d < ParlerDecoderWeights.HiddenDim; d++) row[d] = x[i][d] + selfAttnOut[i][d];
            afterSelf[i] = row;
        });

        var normed2 = new float[t][];
        Parallel.For(0, t, i => normed2[i] = LayerNorm(afterSelf[i], lw.CrossAttnLayerNormWeight, lw.CrossAttnLayerNormBias));
        var crossAttnOut = CrossAttention(normed2, encoderHidden, lw);

        var afterCross = new float[t][];
        Parallel.For(0, t, i =>
        {
            var row = new float[ParlerDecoderWeights.HiddenDim];
            for (int d = 0; d < ParlerDecoderWeights.HiddenDim; d++) row[d] = afterSelf[i][d] + crossAttnOut[i][d];
            afterCross[i] = row;
        });

        var normed3 = new float[t][];
        Parallel.For(0, t, i => normed3[i] = LayerNorm(afterCross[i], lw.FinalLayerNormWeight, lw.FinalLayerNormBias));
        var ffnOut = Ffn(normed3, lw);

        var output = new float[t][];
        Parallel.For(0, t, i =>
        {
            var row = new float[ParlerDecoderWeights.HiddenDim];
            for (int d = 0; d < ParlerDecoderWeights.HiddenDim; d++) row[d] = afterCross[i][d] + ffnOut[i][d];
            output[i] = row;
        });
        return output;
    }

    /// <summary>Real causal self-attention: full MHA (no GQA), standard `1/sqrt(headDim)` scaling, no RoPE.</summary>
    private static float[][] SelfAttentionCausal(float[][] x, ParlerDecoderLayerWeights lw)
    {
        int t = x.Length;
        int dim = ParlerDecoderWeights.HiddenDim;
        int heads = ParlerDecoderWeights.NumHeads;

        var q = new float[t][];
        var k = new float[t][];
        var v = new float[t][];
        Parallel.For(0, t, i =>
        {
            q[i] = LinearNoBias(x[i], lw.SelfAttnQWeight, dim, dim);
            k[i] = LinearNoBias(x[i], lw.SelfAttnKWeight, dim, dim);
            v[i] = LinearNoBias(x[i], lw.SelfAttnVWeight, dim, dim);
        });

        var context = new float[t][];
        for (int i = 0; i < t; i++) context[i] = new float[dim];

        float scale = 1f / MathF.Sqrt(HeadDim);
        Parallel.For(0, heads, h =>
        {
            int off = h * HeadDim;
            var scores = new float[t];
            for (int i = 0; i < t; i++)
            {
                for (int j = 0; j <= i; j++) // causal
                    scores[j] = Dot(q[i], k[j], off, HeadDim) * scale;
                SoftmaxInPlace(scores, i + 1);

                var ctxSpan = context[i].AsSpan(off, HeadDim);
                for (int j = 0; j <= i; j++)
                    for (int d = 0; d < HeadDim; d++) ctxSpan[d] += scores[j] * v[j][off + d];
            }
        });

        var output = new float[t][];
        Parallel.For(0, t, i => output[i] = LinearNoBias(context[i], lw.SelfAttnOutWeight, dim, dim));
        return output;
    }

    /// <summary>Real cross-attention to the T5 encoder's output: full MHA, standard scaling, non-causal (attends to all encoder positions).</summary>
    private static float[][] CrossAttention(float[][] x, float[][] encoderHidden, ParlerDecoderLayerWeights lw)
    {
        int tq = x.Length;
        int tk = encoderHidden.Length;
        int dim = ParlerDecoderWeights.HiddenDim;
        int heads = ParlerDecoderWeights.NumHeads;

        var q = new float[tq][];
        Parallel.For(0, tq, i => q[i] = LinearNoBias(x[i], lw.CrossAttnQWeight, dim, dim));
        var k = new float[tk][];
        var v = new float[tk][];
        Parallel.For(0, tk, j =>
        {
            k[j] = LinearNoBias(encoderHidden[j], lw.CrossAttnKWeight, dim, dim);
            v[j] = LinearNoBias(encoderHidden[j], lw.CrossAttnVWeight, dim, dim);
        });

        var context = new float[tq][];
        for (int i = 0; i < tq; i++) context[i] = new float[dim];

        float scale = 1f / MathF.Sqrt(HeadDim);
        Parallel.For(0, heads, h =>
        {
            int off = h * HeadDim;
            var scores = new float[tk];
            for (int i = 0; i < tq; i++)
            {
                for (int j = 0; j < tk; j++)
                    scores[j] = Dot(q[i], k[j], off, HeadDim) * scale;
                SoftmaxInPlace(scores, tk);

                var ctxSpan = context[i].AsSpan(off, HeadDim);
                for (int j = 0; j < tk; j++)
                    for (int d = 0; d < HeadDim; d++) ctxSpan[d] += scores[j] * v[j][off + d];
            }
        });

        var output = new float[tq][];
        Parallel.For(0, tq, i => output[i] = LinearNoBias(context[i], lw.CrossAttnOutWeight, dim, dim));
        return output;
    }

    /// <summary>Real plain (non-gated) FFN: `fc2(gelu(fc1(x)))`, no bias.</summary>
    private static float[][] Ffn(float[][] x, ParlerDecoderLayerWeights lw)
    {
        int t = x.Length;
        var output = new float[t][];
        Parallel.For(0, t, i =>
        {
            var h = LinearNoBias(x[i], lw.Fc1Weight, ParlerDecoderWeights.HiddenDim, ParlerDecoderWeights.FfnDim);
            for (int d = 0; d < h.Length; d++) h[d] = Gelu(h[d]);
            output[i] = LinearNoBias(h, lw.Fc2Weight, ParlerDecoderWeights.FfnDim, ParlerDecoderWeights.HiddenDim);
        });
        return output;
    }

    /// <summary>Real (exact, erf-based) GELU -- HF's default "gelu" activation, NOT the tanh approximation ("gelu_new") T5 uses.</summary>
    private static float Gelu(float x) => 0.5f * x * (1f + Erf(x / 1.4142135f));

    private static float Erf(float x)
    {
        // Abramowitz-Stegun 7.1.26 approximation, max error ~1.5e-7 -- sufficient for F32 activations.
        float sign = MathF.Sign(x);
        x = MathF.Abs(x);
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float tt = 1f / (1f + p * x);
        float y = 1f - (((((a5 * tt + a4) * tt) + a3) * tt + a2) * tt + a1) * tt * MathF.Exp(-x * x);
        return sign * y;
    }

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

    /// <summary>Real `nn.LayerNorm`: mean-subtract, variance-normalize, scale + bias (NOT RMSNorm).</summary>
    private static float[] LayerNorm(float[] x, float[] weight, float[] bias, float eps = ParlerDecoderWeights.LayerNormEps)
    {
        int n = x.Length;
        float mean = 0f;
        for (int i = 0; i < n; i++) mean += x[i];
        mean /= n;
        float variance = 0f;
        for (int i = 0; i < n; i++) { float d = x[i] - mean; variance += d * d; }
        variance /= n;
        float invStd = 1f / MathF.Sqrt(variance + eps);

        var output = new float[n];
        for (int i = 0; i < n; i++) output[i] = (x[i] - mean) * invStd * weight[i] + bias[i];
        return output;
    }

    private static void SoftmaxInPlace(float[] scores, int count)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < count; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = 0; i < count; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = 0; i < count; i++) scores[i] *= invSum;
    }
}
