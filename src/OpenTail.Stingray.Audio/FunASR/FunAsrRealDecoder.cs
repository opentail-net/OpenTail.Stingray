using System;
using System.Numerics.Tensors;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.FunASR;

/// <summary>
/// Real Paraformer decoder forward pass, transcribed from the real `funasr` Python package
/// (`funasr/models/sanm/decoder.py`, `funasr/models/sanm/attention.py`'s
/// `MultiHeadedAttentionSANMDecoder`/`MultiHeadedAttentionCrossAtt`), see docs/audio-review-
/// progress.md's FunASR section for the full derivation. Named <c>FunAsrRealDecoder</c> (not
/// <c>FunAsrDecoder</c>) to avoid colliding with the pre-existing fake/procedural
/// <see cref="FunAsrDecoder"/> class in <c>FunAsrPipeline.cs</c> until that class is rewired to
/// call this one.
///
/// <para><b>Genuinely surprising layer order, confirmed from real source, do not assume the
/// "obvious" self-attn-then-FFN transformer order</b>: each decoder layer runs its FFN FIRST
/// (on `norm1(input)`), THEN FSMN-only self-attention (on `norm2(ffn_output)`) with a residual
/// back to the ORIGINAL layer input (not the FFN output), THEN cross-attention (on
/// `norm3(self_attn_output)`) with its own residual. The final `decoders3.0` layer has NEITHER
/// self-attn NOR cross-attn, so it reduces to just `feed_forward(norm1(x))` with NO residual at
/// all.</para>
/// </summary>
public static class FunAsrRealDecoder
{
    /// <summary>Runs the full decoder: acousticEmbeds (decoder.embed is Identity for this checkpoint, so this IS the decoder's input) -> 16x main decoders.N (self-attn + cross-attn) -> decoders3.0 (FFN-only, no residual) -> after_norm -> output_layer. Returns per-position vocab logits [numTokens, VocabSize].</summary>
    public static float[][] Forward(FunAsrWeights w, float[][] acousticEmbeds, float[][] encoderOutput)
    {
        var x = acousticEmbeds;
        foreach (var layer in w.DecoderLayerWeights)
            x = DecoderLayer(x, encoderOutput, layer, w.DecoderHeads, hasSelfAttn: true, hasSrcAttn: true);

        x = DecoderFfnOnlyLayer(x, w.Decoders3Layer);

        int t = x.Length;
        var normed = new float[t][];
        for (int i = 0; i < t; i++) normed[i] = LayerNorm(x[i], w.DecoderAfterNormWeight, w.DecoderAfterNormBias);

        var logits = new float[t][];
        for (int i = 0; i < t; i++)
            logits[i] = Linear(normed[i], w.DecoderOutputWeight, w.DecoderOutputBias, w.EncoderDim, w.VocabSize);
        return logits;
    }

    private static float[][] DecoderLayer(float[][] tgt, float[][] memory, FunAsrDecoderLayerWeights lw, int heads, bool hasSelfAttn, bool hasSrcAttn)
    {
        int t = tgt.Length;
        int size = lw.Norm1Weight.Length;

        var normed1 = new float[t][];
        for (int i = 0; i < t; i++) normed1[i] = LayerNorm(tgt[i], lw.Norm1Weight, lw.Norm1Bias);

        var ffnOut = FfnDecoderSanm(normed1, lw);
        var x = ffnOut;

        if (hasSelfAttn)
        {
            var normed2 = new float[t][];
            for (int i = 0; i < t; i++) normed2[i] = LayerNorm(ffnOut[i], lw.Norm2Weight, lw.Norm2Bias);
            var selfAttnOut = DecoderFsmnSelfAttn(normed2, lw.SelfAttnFsmnWeight, kernel: 11);

            var afterSelf = new float[t][];
            for (int i = 0; i < t; i++)
            {
                var row = new float[size];
                for (int d = 0; d < size; d++) row[d] = tgt[i][d] + selfAttnOut[i][d]; // residual = ORIGINAL tgt, not ffnOut
                afterSelf[i] = row;
            }
            x = afterSelf;
        }

        if (hasSrcAttn)
        {
            var normed3 = new float[t][];
            for (int i = 0; i < t; i++) normed3[i] = LayerNorm(x[i], lw.Norm3Weight, lw.Norm3Bias);
            var crossOut = CrossAttention(normed3, memory, lw, heads);

            var afterCross = new float[t][];
            for (int i = 0; i < t; i++)
            {
                var row = new float[size];
                for (int d = 0; d < size; d++) row[d] = x[i][d] + crossOut[i][d];
                afterCross[i] = row;
            }
            x = afterCross;
        }

        return x;
    }

    /// <summary>`decoders3.0`: self_attn=None, src_attn=None -- reduces to feed_forward(norm1(x)) with NO residual add at all.</summary>
    private static float[][] DecoderFfnOnlyLayer(float[][] tgt, FunAsrDecoderFfnLayerWeights lw)
    {
        int t = tgt.Length;
        var normed1 = new float[t][];
        for (int i = 0; i < t; i++) normed1[i] = LayerNorm(tgt[i], lw.Norm1Weight, lw.Norm1Bias);

        int size = lw.Norm1Weight.Length;
        int ffnDim = lw.FfnW1Bias.Length;
        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var h = Linear(normed1[i], lw.FfnW1Weight, lw.FfnW1Bias, size, ffnDim);
            for (int d = 0; d < ffnDim; d++) h[d] = MathF.Max(0f, h[d]);
            h = LayerNorm(h, lw.FfnNormWeight, lw.FfnNormBias);
            output[i] = LinearNoBias(h, lw.FfnW2Weight, ffnDim, size);
        }
        return output;
    }

    /// <summary>`PositionwiseFeedForwardDecoderSANM`: w_2(norm(ReLU(w_1(x)))), w_2 has NO bias.</summary>
    private static float[][] FfnDecoderSanm(float[][] x, FunAsrDecoderLayerWeights lw)
    {
        int t = x.Length;
        int size = lw.Norm1Weight.Length;
        int ffnDim = lw.FfnW1Bias.Length;
        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var h = Linear(x[i], lw.FfnW1Weight, lw.FfnW1Bias, size, ffnDim);
            for (int d = 0; d < ffnDim; d++) h[d] = MathF.Max(0f, h[d]);
            h = LayerNorm(h, lw.FfnNormWeight, lw.FfnNormBias);
            output[i] = LinearNoBias(h, lw.FfnW2Weight, ffnDim, size);
        }
        return output;
    }

    /// <summary>Decoder self-attn is FSMN-ONLY (no Q/K/V), applied directly to the layer input -- same depthwise-conv pattern as the encoder's FSMN branch.</summary>
    private static float[][] DecoderFsmnSelfAttn(float[][] x, float[] fsmnWeight, int kernel)
    {
        int t = x.Length;
        int c = x[0].Length;
        int left = (kernel - 1) / 2;
        int right = kernel - 1 - left;

        var output = new float[t][];
        for (int ti = 0; ti < t; ti++)
        {
            var row = new float[c];
            for (int ch = 0; ch < c; ch++)
            {
                float sum = 0f;
                int wBase = ch * kernel;
                for (int kk = 0; kk < kernel; kk++)
                {
                    int srcT = ti - left + kk;
                    if ((uint)srcT < (uint)t) sum += x[srcT][ch] * fsmnWeight[wBase + kk];
                }
                row[ch] = sum + x[ti][ch];
            }
            output[ti] = row;
        }
        return output;
    }

    private static float[][] CrossAttention(float[][] x, float[][] memory, FunAsrDecoderLayerWeights lw, int heads)
    {
        int tq = x.Length;
        int tk = memory.Length;
        int nFeat = lw.SrcAttnOutBias.Length;
        int dK = nFeat / heads;

        var q = new float[tq][];
        for (int i = 0; i < tq; i++) q[i] = Linear(x[i], lw.SrcAttnQWeight, lw.SrcAttnQBias, nFeat, nFeat);

        var k = new float[tk][];
        var v = new float[tk][];
        for (int j = 0; j < tk; j++)
        {
            var kv = Linear(memory[j], lw.SrcAttnKvWeight, lw.SrcAttnKvBias, nFeat, nFeat * 2);
            k[j] = kv.AsSpan(0, nFeat).ToArray();
            v[j] = kv.AsSpan(nFeat, nFeat).ToArray();
        }

        float scale = MathF.Pow(dK, -0.5f);
        var context = new float[tq][];
        for (int i = 0; i < tq; i++) context[i] = new float[nFeat];

        for (int h = 0; h < heads; h++)
        {
            int off = h * dK;
            for (int i = 0; i < tq; i++)
            {
                var scores = new float[tk];
                for (int j = 0; j < tk; j++)
                    scores[j] = TensorPrimitives.Dot(q[i].AsSpan(off, dK), k[j].AsSpan(off, dK)) * scale;
                SoftmaxInPlace(scores);

                var ctxSpan = context[i].AsSpan(off, dK);
                for (int j = 0; j < tk; j++)
                    TensorPrimitives.MultiplyAdd(v[j].AsSpan(off, dK), scores[j], ctxSpan, ctxSpan);
            }
        }

        var output = new float[tq][];
        for (int i = 0; i < tq; i++)
            output[i] = Linear(context[i], lw.SrcAttnOutWeight, lw.SrcAttnOutBias, nFeat, nFeat);
        return output;
    }

    private static unsafe float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* wp = weight, xp = input)
        {
            for (int o = 0; o < outDim; o++)
                output[o] = bias[o] + SimdKernels.DotF32(wp + (long)o * inDim, xp, inDim);
        }
        return output;
    }

    private static unsafe float[] LinearNoBias(float[] input, float[] weight, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* wp = weight, xp = input)
        {
            for (int o = 0; o < outDim; o++)
                output[o] = SimdKernels.DotF32(wp + (long)o * inDim, xp, inDim);
        }
        return output;
    }

    private static float[] LayerNorm(float[] x, float[] weight, float[] bias, float eps = 1e-12f)
    {
        int n = x.Length;
        float mean = TensorPrimitives.Sum((ReadOnlySpan<float>)x) / n;
        float variance = 0f;
        for (int i = 0; i < n; i++) { float d = x[i] - mean; variance += d * d; }
        variance /= n;
        float invStd = 1f / MathF.Sqrt(variance + eps);

        var output = new float[n];
        for (int i = 0; i < n; i++)
            output[i] = (x[i] - mean) * invStd * weight[i] + bias[i];
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
