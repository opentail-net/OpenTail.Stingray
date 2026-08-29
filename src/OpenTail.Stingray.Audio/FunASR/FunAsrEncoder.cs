
namespace OpenTail.Stingray.Audio.FunASR;

/// <summary>
/// Real SAN-M encoder forward pass for Paraformer, transcribed from the real `funasr` Python
/// package (`funasr/models/sanm/{attention,encoder,positionwise_feed_forward}.py`, see
/// docs/audio-review-progress.md's FunASR section for the full derivation -- do not re-derive
/// or guess any of this, especially the two non-obvious details below).
///
/// <para><b>Two critical, easy-to-miss details, both confirmed from real source, not
/// assumed</b>: (1) `encoders0.0` (560-dim input) has NO residual connection around its
/// self-attention, since `in_size (560) != size (512)` -- every other layer (512-&gt;512) DOES
/// get the residual. (2) The FSMN memory branch is added to standard self-attention's output
/// (`att_outs + fsmn_memory`) -- `examples/paraformer.cpp` has this exact add commented out/
/// disabled, confirmed a known-broken reference on this detail, do not port its encoder as-is.
/// </para>
///
/// <para>Shared <c>Linear</c>/<c>LayerNorm</c>/<c>SoftmaxInPlace</c>/FSMN-conv/multi-head-
/// attention math lives in <see cref="FunAsrKernels"/> (extracted alongside
/// <c>FunAsrRealDecoder.cs</c> in this fire's DRY + performance pass -- see that class's doc
/// comment for the parallelization rationale). Per-position LayerNorm/residual loops here are
/// parallelized with <c>Parallel.For</c>, matching the convention already used by
/// <c>WhisperEncoder.cs</c>.</para>
/// </summary>
public static class FunAsrEncoder
{
    /// <summary>Runs the full encoder: encoders0.0 (560-dim, no residual) -> 49x main encoders.N (512-dim, with residual) -> after_norm. Input is frame-major [T, 560] (already CMVN-normalized + mel-splice, see FunAsrMelExtractor). Returns [T, 512].</summary>
    public static float[][] Forward(FunAsrWeights w, float[][] input)
    {
        int t = input.Length;
        var x = EncoderLayer(input, w.Encoders0Layer, w.EncoderHeads, inSize: 560, size: w.EncoderDim);

        foreach (var layer in w.EncoderLayerWeights)
            x = EncoderLayer(x, layer, w.EncoderHeads, inSize: w.EncoderDim, size: w.EncoderDim);

        var output = new float[t][];
        Parallel.For(0, t, i => output[i] = FunAsrKernels.LayerNorm(x[i], w.EncoderAfterNormWeight, w.EncoderAfterNormBias));
        return output;
    }

    private static float[][] EncoderLayer(float[][] x, FunAsrEncoderLayerWeights lw, int heads, int inSize, int size)
    {
        int t = x.Length;
        var normed1 = new float[t][];
        Parallel.For(0, t, i => normed1[i] = FunAsrKernels.LayerNorm(x[i], lw.Norm1Weight, lw.Norm1Bias));

        var attnOut = SelfAttentionSanm(normed1, lw, heads);

        var afterAttn = new float[t][];
        if (inSize == size)
        {
            Parallel.For(0, t, i =>
            {
                var row = new float[size];
                for (int d = 0; d < size; d++) row[d] = x[i][d] + attnOut[i][d];
                afterAttn[i] = row;
            });
        }
        else
        {
            // encoders0.0 only: in_size (560) != size (512), so the residual is skipped entirely
            // (confirmed from EncoderLayerSANM.forward's `if self.in_size == self.size` branch).
            afterAttn = attnOut;
        }

        var normed2 = new float[t][];
        Parallel.For(0, t, i => normed2[i] = FunAsrKernels.LayerNorm(afterAttn[i], lw.Norm2Weight, lw.Norm2Bias));

        var ffnOut = FfnPlain(normed2, lw);

        var output = new float[t][];
        Parallel.For(0, t, i =>
        {
            var row = new float[size];
            for (int d = 0; d < size; d++) row[d] = afterAttn[i][d] + ffnOut[i][d];
            output[i] = row;
        });
        return output;
    }

    /// <summary>Real `MultiHeadedAttentionSANM.forward`: standard scaled-dot-product attention PLUS the FSMN memory branch (both summed) -- see class doc comment.</summary>
    private static float[][] SelfAttentionSanm(float[][] x, FunAsrEncoderLayerWeights lw, int heads)
    {
        int t = x.Length;
        int nFeat = lw.AttnOutBias.Length; // 512

        // NOTE: measured this fire -- despite FunAsrKernels.Linear already parallelizing
        // internally over output channels via SimdKernels.MatVecF32 (outDim >= 64 threshold),
        // wrapping these per-position calls in an outer Parallel.For(0, t, ...) measured FASTER
        // than a plain serial loop on a 12s-audio benchmark (median ~3.6s vs ~3.8s, 8 samples
        // each, consistent direction across all samples -- not noise). The theoretical nested-
        // parallelism-oversubscription concern doesn't hold up in practice here, likely because
        // t (encoder frame count) is small enough per layer that the outer Parallel.For's task
        // overhead is cheap relative to the work each task does. Do not "fix" this back to serial
        // without re-measuring -- see docs/audio-review-progress.md's FunASR performance-pass
        // entry for the full A/B.
        var q = new float[t][];
        var k = new float[t][];
        var v = new float[t][];
        Parallel.For(0, t, i =>
        {
            var qkv = FunAsrKernels.Linear(x[i], lw.AttnQkvWeight, lw.AttnQkvBias, lw.InputDim, nFeat * 3);
            q[i] = qkv.AsSpan(0, nFeat).ToArray();
            k[i] = qkv.AsSpan(nFeat, nFeat).ToArray();
            v[i] = qkv.AsSpan(2 * nFeat, nFeat).ToArray();
        });

        var fsmnMemory = FunAsrKernels.FsmnDepthwiseConv(v, lw.AttnFsmnWeight, kernel: 11);
        var context = FunAsrKernels.MultiHeadAttention(q, k, v, heads);

        var attOuts = new float[t][];
        Parallel.For(0, t, i => attOuts[i] = FunAsrKernels.Linear(context[i], lw.AttnOutWeight, lw.AttnOutBias, nFeat, nFeat));

        var result = new float[t][];
        Parallel.For(0, t, i =>
        {
            var row = new float[nFeat];
            for (int d = 0; d < nFeat; d++) row[d] = attOuts[i][d] + fsmnMemory[i][d];
            result[i] = row;
        });
        return result;
    }

    /// <summary>Real plain `PositionwiseFeedForward`: w_2(ReLU(w_1(x))), no internal norm.</summary>
    private static float[][] FfnPlain(float[][] x, FunAsrEncoderLayerWeights lw)
    {
        int t = x.Length;
        int size = lw.FfnW2Bias.Length;
        int ffnDim = lw.FfnW1Bias.Length;
        var output = new float[t][];
        Parallel.For(0, t, i =>
        {
            var h = FunAsrKernels.Linear(x[i], lw.FfnW1Weight, lw.FfnW1Bias, size, ffnDim);
            for (int d = 0; d < ffnDim; d++) h[d] = MathF.Max(0f, h[d]);
            output[i] = FunAsrKernels.Linear(h, lw.FfnW2Weight, lw.FfnW2Bias, ffnDim, size);
        });
        return output;
    }
}
