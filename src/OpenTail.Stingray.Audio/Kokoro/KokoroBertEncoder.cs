using System;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Kokoro;

/// <summary>
/// PLBERT text encoder: HF ALBERT-style transformer (examples/kokoro-py/modules.py CustomAlbert,
/// wrapping transformers.AlbertModel). Parameter-shared: the SAME attn/ffn weight set is applied
/// <see cref="KokoroWeights.BertNumHiddenLayers"/> times in a loop (that's why the GGUF file has
/// only one un-indexed bert.attn_*/bert.ffn_* tensor set, not one per layer).
///
/// Verified against examples/kokoro-py/model.py's KModel.forward (the exact real algorithm --
/// see docs/audio-review-progress.md "Kokoro exact algorithm") and against HF transformers'
/// AlbertModel/AlbertLayer/AlbertAttention source (post-LN block: residual+LN after attention,
/// residual+LN after FFN; layer_norm_eps=1e-12, hidden_act="gelu_new" tanh-approximation --
/// ALBERT's own defaults, not Whisper/GPT2's). Assumes a single utterance with no padding
/// (text_mask all-false), matching how this pipeline is actually driven.
/// </summary>
public static class KokoroBertEncoder
{
    private const float LayerNormEps = 1e-12f;

    /// <summary>Runs the full embeddings + 12x shared-layer ALBERT stack. Returns last_hidden_state, [T, 768] row-major.</summary>
    public static float[] Forward(KokoroWeights weights, ReadOnlySpan<int> inputIds)
    {
        var bert = weights.Bert;
        int t = inputIds.Length;
        int embSize = weights.BertEmbeddingSize;   // 128
        int hidden = weights.BertHiddenSize;        // 768
        int heads = weights.BertNumHeads;           // 12
        int headDim = hidden / heads;                // 64
        int ffDim = weights.BertIntermediateSize;    // 2048

        // 1. Embeddings: word + position + token_type(=0), then LayerNorm, in embSize (128) dims.
        var emb = new float[t * embSize];
        for (int i = 0; i < t; i++)
        {
            int tok = inputIds[i];
            for (int d = 0; d < embSize; d++)
            {
                float word = bert.EmbdTokWeight[tok * embSize + d];
                float pos = bert.EmbdPosWeight[i * embSize + d];
                float tt = bert.EmbdTtWeight[0 * embSize + d]; // token_type_id always 0 here
                emb[i * embSize + d] = word + pos + tt;
            }
        }
        var embNormed = new float[t * embSize];
        unsafe
        {
            fixed (float* src = emb, dst = embNormed, w = bert.EmbdLnWeight, b = bert.EmbdLnBias)
            {
                for (int i = 0; i < t; i++)
                    SimdKernels.LayerNorm(dst + i * embSize, src + i * embSize, w, b, embSize, LayerNormEps);
            }
        }

        // 2. Project embSize (128) -> hidden (768).
        var hiddenStates = new float[t * hidden];
        unsafe
        {
            fixed (float* outp = hiddenStates, wgt = bert.EmbdProjWeight, inp = embNormed)
            {
                SimdKernels.MatMulBatchedF32(outp, wgt, inp, t, hidden, embSize);
            }
        }
        AddBiasInPlace(hiddenStates, bert.EmbdProjBias, t, hidden);

        // 3. Shared transformer layer, applied BertNumHiddenLayers times.
        var q = new float[t * hidden];
        var k = new float[t * hidden];
        var v = new float[t * hidden];
        var attnCtx = new float[t * hidden];
        var attnProj = new float[t * hidden];
        var ffnMid = new float[t * ffDim];
        var ffnOut = new float[t * hidden];
        float scale = 1.0f / MathF.Sqrt(headDim);

        for (int layer = 0; layer < weights.BertNumHiddenLayers; layer++)
        {
            unsafe
            {
                fixed (float* qp = q, kp = k, vp = v, hp = hiddenStates,
                       qw = bert.AttnQWeight, kw = bert.AttnKWeight, vw = bert.AttnVWeight)
                {
                    SimdKernels.MatMulBatchedF32(qp, qw, hp, t, hidden, hidden);
                    SimdKernels.MatMulBatchedF32(kp, kw, hp, t, hidden, hidden);
                    SimdKernels.MatMulBatchedF32(vp, vw, hp, t, hidden, hidden);
                }
            }
            AddBiasInPlace(q, bert.AttnQBias, t, hidden);
            AddBiasInPlace(k, bert.AttnKBias, t, hidden);
            AddBiasInPlace(v, bert.AttnVBias, t, hidden);

            MultiHeadAttention(q, k, v, attnCtx, t, heads, headDim, scale);

            unsafe
            {
                fixed (float* outp = attnProj, wgt = bert.AttnOWeight, inp = attnCtx)
                {
                    SimdKernels.MatMulBatchedF32(outp, wgt, inp, t, hidden, hidden);
                }
            }
            AddBiasInPlace(attnProj, bert.AttnOBias, t, hidden);

            // residual + LN
            for (int i = 0; i < t * hidden; i++) attnProj[i] += hiddenStates[i];
            unsafe
            {
                fixed (float* src = attnProj, dst = hiddenStates, w = bert.AttnLnWeight, b = bert.AttnLnBias)
                {
                    for (int i = 0; i < t; i++)
                        SimdKernels.LayerNorm(dst + i * hidden, src + i * hidden, w, b, hidden, LayerNormEps);
                }
            }

            // FFN: hidden -> ffDim -> hidden, GELU (tanh approx, ALBERT's "gelu_new") in between.
            unsafe
            {
                fixed (float* outp = ffnMid, wgt = bert.FfnUpWeight, inp = hiddenStates)
                {
                    SimdKernels.MatMulBatchedF32(outp, wgt, inp, t, ffDim, hidden);
                }
            }
            AddBiasInPlace(ffnMid, bert.FfnUpBias, t, ffDim);
            for (int i = 0; i < ffnMid.Length; i++) ffnMid[i] = GeluNew(ffnMid[i]);

            unsafe
            {
                fixed (float* outp = ffnOut, wgt = bert.FfnDownWeight, inp = ffnMid)
                {
                    SimdKernels.MatMulBatchedF32(outp, wgt, inp, t, hidden, ffDim);
                }
            }
            AddBiasInPlace(ffnOut, bert.FfnDownBias, t, hidden);

            // residual + LN
            for (int i = 0; i < t * hidden; i++) ffnOut[i] += hiddenStates[i];
            unsafe
            {
                fixed (float* src = ffnOut, dst = hiddenStates, w = bert.FfnLnWeight, b = bert.FfnLnBias)
                {
                    for (int i = 0; i < t; i++)
                        SimdKernels.LayerNorm(dst + i * hidden, src + i * hidden, w, b, hidden, LayerNormEps);
                }
            }
        }

        return hiddenStates;
    }

    /// <summary>Projects last_hidden_state [T,768] to [T,512] via bert_proj (KModel.bert_encoder), channel-first output [512,T].</summary>
    public static float[] ProjectToWorkingDim(KokoroWeights weights, float[] lastHiddenState, int t)
    {
        int hidden = weights.BertHiddenSize;
        int outDim = weights.HiddenDim; // 512
        var projected = new float[t * outDim];
        unsafe
        {
            fixed (float* outp = projected, wgt = weights.BertProjWeight, inp = lastHiddenState)
            {
                SimdKernels.MatMulBatchedF32(outp, wgt, inp, t, outDim, hidden);
            }
        }
        AddBiasInPlace(projected, weights.BertProjBias, t, outDim);

        // transpose [T, outDim] -> [outDim, T] (channel-first, matches d_en in model.py)
        var channelFirst = new float[outDim * t];
        for (int i = 0; i < t; i++)
            for (int d = 0; d < outDim; d++)
                channelFirst[d * t + i] = projected[i * outDim + d];
        return channelFirst;
    }

    private static void MultiHeadAttention(float[] q, float[] k, float[] v, float[] output, int t, int heads, int headDim, float scale)
    {
        int hidden = heads * headDim;
        Array.Clear(output);
        var scores = new float[t];

        for (int h = 0; h < heads; h++)
        {
            int off = h * headDim;
            for (int i = 0; i < t; i++)
            {
                float maxScore = float.NegativeInfinity;
                for (int j = 0; j < t; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                        dot += q[i * hidden + off + d] * k[j * hidden + off + d];
                    dot *= scale;
                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }
                float sum = 0f;
                for (int j = 0; j < t; j++)
                {
                    scores[j] = MathF.Exp(scores[j] - maxScore);
                    sum += scores[j];
                }
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0f;
                    for (int j = 0; j < t; j++)
                        acc += scores[j] * v[j * hidden + off + d];
                    output[i * hidden + off + d] = acc / sum;
                }
            }
        }
    }

    private static void AddBiasInPlace(float[] data, float[] bias, int rows, int cols)
    {
        for (int i = 0; i < rows; i++)
            for (int d = 0; d < cols; d++)
                data[i * cols + d] += bias[d];
    }

    /// <summary>ALBERT's default hidden_act "gelu_new": tanh-approximation GELU (matches GPT-2/BERT-new, not the erf-exact variant).</summary>
    private static float GeluNew(float x) =>
        0.5f * x * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (x + 0.044715f * x * x * x)));
}
