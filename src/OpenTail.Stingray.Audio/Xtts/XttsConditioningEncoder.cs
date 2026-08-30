
namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 conditioning path: mel-spectrogram (reference audio) -&gt;
/// `ConditioningEncoder` (conv1x1 init + 6x self-attention `AttentionBlock`) -&gt;
/// `PerceiverResampler` (32 learned latents cross-attending to the encoder output) -&gt; a
/// `[32, 1024]` conditioning sequence prefixed onto the GPT2's real input. See
/// <see cref="XttsConditioningWeights"/>'s doc comment for the real source this was confirmed
/// against.
/// </summary>
public static class XttsConditioningEncoder
{
    /// <summary>mel is channel-first [80, T] (reference audio's mel-spectrogram). Returns the real Perceiver-resampled conditioning latents, channel-first [1024, 32].</summary>
    public static float[] Encode(XttsConditioningWeights w, float[] mel, int t)
    {
        var h = VitsAttentionKernels.Conv1x1(mel, XttsConditioningWeights.MelDim, t, w.EncoderInitWeight, w.EncoderInitBias, XttsConditioningWeights.ModelDim);
        foreach (var block in w.EncoderBlocks)
            h = AttentionBlockForward(h, t, block);

        // h is channel-first [1024, T] -- PerceiverResampler's real input convention is [B, T, dim]
        // (token-major); transpose once here to token-major [T, 1024] for the perceiver's own math
        // (cross-attention is naturally expressed per-token), then transpose the final [32,1024]
        // output BACK to channel-first [1024, 32] to match this codebase's usual convention
        // (matches the real reference's own `conds.permute(0, 2, 1)).transpose(1, 2)` round trip).
        var hTokenMajor = new float[t * XttsConditioningWeights.ModelDim];
        for (int c = 0; c < XttsConditioningWeights.ModelDim; c++)
            for (int ti = 0; ti < t; ti++)
                hTokenMajor[ti * XttsConditioningWeights.ModelDim + c] = h[c * t + ti];

        var latents = PerceiverResamplerForward(w, hTokenMajor, t);

        int numLatents = XttsConditioningWeights.PerceiverNumLatents;
        var latentsChannelFirst = new float[XttsConditioningWeights.ModelDim * numLatents];
        for (int li = 0; li < numLatents; li++)
            for (int c = 0; c < XttsConditioningWeights.ModelDim; c++)
                latentsChannelFirst[c * numLatents + li] = latents[li * XttsConditioningWeights.ModelDim + c];
        return latentsChannelFirst;
    }

    /// <summary>Real `AttentionBlock`: GroupNorm(32) -> qkv conv1x1 -> self-attn (16 heads, no mask) -> proj_out conv1x1, residual added to the NORMALIZED input (`tortoise_norm=False`).</summary>
    private static float[] AttentionBlockForward(float[] x, int t, XttsAttentionBlockWeights bw)
    {
        int ch = XttsConditioningWeights.ModelDim;
        var xNorm = GroupNorm32(x, ch, t, bw.NormWeight, bw.NormBias);

        var qkv = VitsAttentionKernels.Conv1x1(xNorm, ch, t, bw.QkvWeight, bw.QkvBias, 3 * ch);
        var attnOut = QkvSelfAttention(qkv, t, ch, XttsConditioningWeights.EncoderHeads);
        var projOut = VitsAttentionKernels.Conv1x1(attnOut, ch, t, bw.ProjOutWeight, bw.ProjOutBias, ch);

        var output = new float[ch * t];
        for (int i = 0; i < output.Length; i++) output[i] = xNorm[i] + projOut[i];
        return output;
    }

    /// <summary>Real `QKVAttentionLegacy`: qkv is channel-first [3*heads*headDim, T] (q/k/v concatenated along the channel dim, THEN split per-head); scale = 1/sqrt(sqrt(headDim)) applied to both q and k (equivalent to the usual 1/sqrt(headDim) total). No causal mask.</summary>
    private static float[] QkvSelfAttention(float[] qkv, int t, int ch, int heads)
    {
        int headDim = ch / heads;
        float scale = 1f / MathF.Sqrt(MathF.Sqrt(headDim));
        var context = new float[ch * t];

        // Real reshape: qkv.reshape(B*heads, 3*headDim, T).split(headDim, dim=1) -- i.e. for a
        // given head h, its q/k/v rows are qkv[h*3*headDim : h*3*headDim+headDim] (q),
        // [+headDim:+2*headDim] (k), [+2*headDim:+3*headDim] (v) -- NOT q-block/k-block/v-block
        // each spanning all heads (that would be the OTHER common QKV layout, this one interleaves
        // per-head instead -- confirmed from the real `.reshape(bs*heads, ch*3, length)` shape math).
        for (int h = 0; h < heads; h++)
        {
            int qOff = h * 3 * headDim;
            int kOff = qOff + headDim;
            int vOff = qOff + 2 * headDim;

            var scores = new float[t];
            for (int i = 0; i < t; i++)
            {
                for (int j = 0; j < t; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                        dot += (qkv[(qOff + d) * t + i] * scale) * (qkv[(kOff + d) * t + j] * scale);
                    scores[j] = dot;
                }
                VitsAttentionKernels.SoftmaxInPlace(scores);

                for (int j = 0; j < t; j++)
                {
                    float p = scores[j];
                    for (int d = 0; d < headDim; d++)
                        context[(h * headDim + d) * t + i] += p * qkv[(vOff + d) * t + j];
                }
            }
        }

        return context;
    }

    /// <summary>Real GroupNorm(32 groups, channel-first [C,T]): normalize each group (C/32 consecutive channels) jointly across (channels-in-group x T), then per-channel affine.</summary>
    private static float[] GroupNorm32(float[] x, int ch, int t, float[] gamma, float[] beta, float eps = 1e-5f)
    {
        const int groups = 32;
        int chPerGroup = ch / groups;
        var output = new float[ch * t];

        for (int g = 0; g < groups; g++)
        {
            int chStart = g * chPerGroup;
            double mean = 0, sumSq = 0;
            int n = chPerGroup * t;
            for (int c = chStart; c < chStart + chPerGroup; c++)
                for (int ti = 0; ti < t; ti++)
                    mean += x[c * t + ti];
            mean /= n;
            for (int c = chStart; c < chStart + chPerGroup; c++)
                for (int ti = 0; ti < t; ti++)
                {
                    double d = x[c * t + ti] - mean;
                    sumSq += d * d;
                }
            float invStd = (float)(1.0 / Math.Sqrt(sumSq / n + eps));
            for (int c = chStart; c < chStart + chPerGroup; c++)
            {
                float g0 = gamma[c], b0 = beta[c];
                for (int ti = 0; ti < t; ti++)
                    output[c * t + ti] = (float)((x[c * t + ti] - mean) * invStd) * g0 + b0;
            }
        }
        return output;
    }

    /// <summary>Real `PerceiverResampler.forward`: latents (broadcast to batch) cross-attend to [latents ++ context] for `depth` layers (each: cross-attn+residual, GEGLU FFN+residual), final RMSNorm. `x`/context here is token-major [T, ModelDim] (real reference's own convention for this stage).</summary>
    private static float[] PerceiverResamplerForward(XttsConditioningWeights w, float[] context, int contextT)
    {
        int dim = XttsConditioningWeights.ModelDim;
        int numLatents = XttsConditioningWeights.PerceiverNumLatents;
        var latents = (float[])w.PerceiverLatents.Clone(); // [numLatents, dim] token-major

        foreach (var layer in w.PerceiverLayers)
        {
            var attnOut = CrossAttention(latents, numLatents, context, contextT, dim, layer);
            for (int i = 0; i < latents.Length; i++) latents[i] += attnOut[i];

            var ffnOut = Geglu(latents, numLatents, layer);
            for (int i = 0; i < latents.Length; i++) latents[i] += ffnOut[i];
        }

        return RmsNormTokenMajor(latents, numLatents, dim, w.PerceiverNormGamma);
    }

    /// <summary>Real `Attention.forward` with `cross_attn_include_queries=True`: context for K/V is [latents ++ input] (concatenated along sequence), Q comes from latents only. No bias on any of to_q/to_kv/to_out. Token-major [T,dim] in/out.</summary>
    private static float[] CrossAttention(float[] latents, int numLatents, float[] context, int contextT, int dim, XttsPerceiverLayerWeights lw)
    {
        int heads = XttsConditioningWeights.PerceiverHeads;
        int headDim = XttsConditioningWeights.PerceiverDimHead;
        int dimInner = heads * headDim; // 512
        float scale = 1f / MathF.Sqrt(headDim);

        int kvLen = numLatents + contextT;
        var kvInput = new float[kvLen * dim];
        Array.Copy(latents, 0, kvInput, 0, latents.Length);
        Array.Copy(context, 0, kvInput, latents.Length, context.Length);

        var q = LinearTokenMajor(latents, numLatents, dim, lw.ToQWeight, null, dimInner);
        var kv = LinearTokenMajor(kvInput, kvLen, dim, lw.ToKvWeight, null, 2 * dimInner);

        var output = new float[numLatents * dimInner];
        var scores = new float[kvLen];
        for (int h = 0; h < heads; h++)
        {
            int hOff = h * headDim;
            for (int i = 0; i < numLatents; i++)
            {
                for (int j = 0; j < kvLen; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                        dot += q[i * dimInner + hOff + d] * kv[j * 2 * dimInner + hOff + d];
                    scores[j] = dot * scale;
                }
                VitsAttentionKernels.SoftmaxInPlace(scores);

                for (int j = 0; j < kvLen; j++)
                {
                    float p = scores[j];
                    int vBase = j * 2 * dimInner + dimInner + hOff;
                    for (int d = 0; d < headDim; d++)
                        output[i * dimInner + hOff + d] += p * kv[vBase + d];
                }
            }
        }

        return LinearTokenMajor(output, numLatents, dimInner, lw.ToOutWeight, null, dim);
    }

    /// <summary>Real GEGLU FeedForward: Linear(dim->2*inner) -> split -> x*GELU(gates) -> Linear(inner->dim). GELU here is the real erf-based GELU (PyTorch's `F.gelu` default, NOT the tanh approximation the GPT2 trunk's MLP uses).</summary>
    private static float[] Geglu(float[] x, int numTokens, XttsPerceiverLayerWeights lw)
    {
        int dim = XttsConditioningWeights.ModelDim;
        int inner = XttsConditioningWeights.PerceiverFfnInner;

        var h = LinearTokenMajor(x, numTokens, dim, lw.Ffn0Weight, lw.Ffn0Bias, 2 * inner);
        var gated = new float[numTokens * inner];
        for (int ti = 0; ti < numTokens; ti++)
        {
            int rowBase = ti * 2 * inner;
            int outBase = ti * inner;
            for (int d = 0; d < inner; d++)
                gated[outBase + d] = h[rowBase + d] * VitsDurationFlowKernels.Gelu(h[rowBase + inner + d]);
        }
        return LinearTokenMajor(gated, numTokens, inner, lw.Ffn2Weight, lw.Ffn2Bias, dim);
    }

    /// <summary>Token-major Linear: input [T,inDim] row-major, weight [outDim,inDim] row-major (this codebase's usual convention), optional bias.</summary>
    private static float[] LinearTokenMajor(float[] input, int numTokens, int inDim, float[] weight, float[]? bias, int outDim)
    {
        var output = new float[numTokens * outDim];
        for (int ti = 0; ti < numTokens; ti++)
        {
            int inBase = ti * inDim;
            int outBase = ti * outDim;
            for (int o = 0; o < outDim; o++)
            {
                float sum = bias?[o] ?? 0f;
                int wBase = o * inDim;
                for (int i = 0; i < inDim; i++)
                    sum += weight[wBase + i] * input[inBase + i];
                output[outBase + o] = sum;
            }
        }
        return output;
    }

    private static float[] RmsNormTokenMajor(float[] x, int numTokens, int dim, float[] gamma)
    {
        float scale = MathF.Sqrt(dim);
        var output = new float[x.Length];
        for (int ti = 0; ti < numTokens; ti++)
        {
            int baseIdx = ti * dim;
            double sumSq = 0;
            for (int d = 0; d < dim; d++) sumSq += (double)x[baseIdx + d] * x[baseIdx + d];
            float invNorm = (float)(1.0 / Math.Max(Math.Sqrt(sumSq), 1e-12));
            for (int d = 0; d < dim; d++)
                output[baseIdx + d] = x[baseIdx + d] * invNorm * scale * gamma[d];
        }
        return output;
    }
}
