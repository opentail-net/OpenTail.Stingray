using System;
using System.Threading.Tasks;
using OpenTail.Stingray.Audio.Primitives;

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
///
/// <para>Shared <c>Linear</c>/<c>LayerNorm</c>/<c>SoftmaxInPlace</c>/FSMN-conv/multi-head-
/// attention math lives in <see cref="FunAsrKernels"/> (extracted alongside
/// <c>FunAsrEncoder.cs</c> this fire, since both had copy-pasted the identical helpers -- see
/// that class's doc comment for the parallelization/performance rationale: routing `Linear`
/// through <see cref="SimdKernels.MatVecF32"/> instead of a scalar per-output-channel loop
/// picks up its internal per-row parallelization for free at the output dims this pipeline
/// actually uses (QKV/FFN/vocab projections all &gt;= 64), and multi-head attention is now
/// parallelized over heads).</para>
/// </summary>
public static class FunAsrRealDecoder
{
    /// <summary>Runs the full decoder: acousticEmbeds (decoder.embed is Identity for this checkpoint, so this IS the decoder's input) -> 16x main decoders.N (self-attn + cross-attn) -> decoders3.0 (FFN-only, no residual) -> after_norm -> output_layer. Returns per-position vocab logits [numTokens, VocabSize].</summary>
    public static float[][] Forward(FunAsrWeights w, float[][] acousticEmbeds, float[][] encoderOutput)
    {
        var x = acousticEmbeds;
        foreach (var layer in w.DecoderLayerWeights)
            x = DecoderLayer(x, encoderOutput, layer, w.DecoderHeads);

        x = DecoderFfnOnlyLayer(x, w.Decoders3Layer);

        int t = x.Length;
        var normed = new float[t][];
        Parallel.For(0, t, i => normed[i] = FunAsrKernels.LayerNorm(x[i], w.DecoderAfterNormWeight, w.DecoderAfterNormBias));

        // Kept as Parallel.For -- measured faster than a serial loop despite the nested-
        // parallelism theory, see FunAsrEncoder.SelfAttentionSanm's comment for the A/B.
        var logits = new float[t][];
        Parallel.For(0, t, i => logits[i] = FunAsrKernels.Linear(normed[i], w.DecoderOutputWeight, w.DecoderOutputBias, w.EncoderDim, w.VocabSize));
        return logits;
    }

    private static float[][] DecoderLayer(float[][] tgt, float[][] memory, FunAsrDecoderLayerWeights lw, int heads)
    {
        int t = tgt.Length;
        int size = lw.Norm1Weight.Length;

        var normed1 = new float[t][];
        Parallel.For(0, t, i => normed1[i] = FunAsrKernels.LayerNorm(tgt[i], lw.Norm1Weight, lw.Norm1Bias));

        var ffnOut = FfnDecoderSanm(normed1, lw);

        var normed2 = new float[t][];
        Parallel.For(0, t, i => normed2[i] = FunAsrKernels.LayerNorm(ffnOut[i], lw.Norm2Weight, lw.Norm2Bias));
        var selfAttnOut = FunAsrKernels.FsmnDepthwiseConv(normed2, lw.SelfAttnFsmnWeight, kernel: 11);

        var afterSelf = new float[t][];
        Parallel.For(0, t, i =>
        {
            var row = new float[size];
            for (int d = 0; d < size; d++) row[d] = tgt[i][d] + selfAttnOut[i][d]; // residual = ORIGINAL tgt, not ffnOut
            afterSelf[i] = row;
        });

        var normed3 = new float[t][];
        Parallel.For(0, t, i => normed3[i] = FunAsrKernels.LayerNorm(afterSelf[i], lw.Norm3Weight, lw.Norm3Bias));
        var crossOut = CrossAttention(normed3, memory, lw, heads);

        var afterCross = new float[t][];
        Parallel.For(0, t, i =>
        {
            var row = new float[size];
            for (int d = 0; d < size; d++) row[d] = afterSelf[i][d] + crossOut[i][d];
            afterCross[i] = row;
        });
        return afterCross;
    }

    /// <summary>`decoders3.0`: self_attn=None, src_attn=None -- reduces to feed_forward(norm1(x)) with NO residual add at all.</summary>
    private static float[][] DecoderFfnOnlyLayer(float[][] tgt, FunAsrDecoderFfnLayerWeights lw)
    {
        int t = tgt.Length;
        var normed1 = new float[t][];
        Parallel.For(0, t, i => normed1[i] = FunAsrKernels.LayerNorm(tgt[i], lw.Norm1Weight, lw.Norm1Bias));

        int size = lw.Norm1Weight.Length;
        int ffnDim = lw.FfnW1Bias.Length;
        var output = new float[t][];
        Parallel.For(0, t, i =>
        {
            var h = FunAsrKernels.Linear(normed1[i], lw.FfnW1Weight, lw.FfnW1Bias, size, ffnDim);
            for (int d = 0; d < ffnDim; d++) h[d] = MathF.Max(0f, h[d]);
            h = FunAsrKernels.LayerNorm(h, lw.FfnNormWeight, lw.FfnNormBias);
            output[i] = FunAsrKernels.LinearNoBias(h, lw.FfnW2Weight, ffnDim, size);
        });
        return output;
    }

    /// <summary>`PositionwiseFeedForwardDecoderSANM`: w_2(norm(ReLU(w_1(x)))), w_2 has NO bias.</summary>
    private static float[][] FfnDecoderSanm(float[][] x, FunAsrDecoderLayerWeights lw)
    {
        int t = x.Length;
        int size = lw.Norm1Weight.Length;
        int ffnDim = lw.FfnW1Bias.Length;
        var output = new float[t][];
        Parallel.For(0, t, i =>
        {
            var h = FunAsrKernels.Linear(x[i], lw.FfnW1Weight, lw.FfnW1Bias, size, ffnDim);
            for (int d = 0; d < ffnDim; d++) h[d] = MathF.Max(0f, h[d]);
            h = FunAsrKernels.LayerNorm(h, lw.FfnNormWeight, lw.FfnNormBias);
            output[i] = FunAsrKernels.LinearNoBias(h, lw.FfnW2Weight, ffnDim, size);
        });
        return output;
    }

    private static float[][] CrossAttention(float[][] x, float[][] memory, FunAsrDecoderLayerWeights lw, int heads)
    {
        int tq = x.Length;
        int tk = memory.Length;
        int nFeat = lw.SrcAttnOutBias.Length;

        var q = new float[tq][];
        Parallel.For(0, tq, i => q[i] = FunAsrKernels.Linear(x[i], lw.SrcAttnQWeight, lw.SrcAttnQBias, nFeat, nFeat));

        var k = new float[tk][];
        var v = new float[tk][];
        Parallel.For(0, tk, j =>
        {
            var kv = FunAsrKernels.Linear(memory[j], lw.SrcAttnKvWeight, lw.SrcAttnKvBias, nFeat, nFeat * 2);
            k[j] = kv.AsSpan(0, nFeat).ToArray();
            v[j] = kv.AsSpan(nFeat, nFeat).ToArray();
        });

        var context = FunAsrKernels.MultiHeadAttention(q, k, v, heads);

        var output = new float[tq][];
        Parallel.For(0, tq, i => output[i] = FunAsrKernels.Linear(context[i], lw.SrcAttnOutWeight, lw.SrcAttnOutBias, nFeat, nFeat));
        return output;
    }
}
