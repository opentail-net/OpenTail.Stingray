using System;

namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// F5-TTS's `DiTBlock` (modules.py): AdaLN-Zero modulated self-attention (with interleaved
/// "GPT-J style" RoPE, `x_transformers`'s `RotaryEmbedding`/`apply_rotary_pos_emb` convention --
/// NOT the split-half "rotate_half" convention some other RoPE implementations use, confirmed by
/// reading the actual installed `x_transformers` package source) + gated FFN (tanh-approx GELU).
/// This checkpoint has no qk_norm and attn_mask_enabled=False (single-utterance inference, no
/// batch padding to mask), so those code paths are simply omitted rather than implemented unused.
/// </summary>
public static class F5DiTBlock
{
    public static float[] Forward(F5TtsWeights w, F5DiTBlockWeights bw, float[] x, float[] tEmb, int t, float[] rotaryCos, float[] rotarySin)
    {
        int dim = F5TtsWeights.HiddenDim;

        // AdaLayerNorm: emb = linear(silu(tEmb)) -> chunk6 (each size dim), constant across all t.
        var siluT = new float[dim];
        for (int d = 0; d < dim; d++) siluT[d] = F5Kernels.SiLU(tEmb[d]);
        var modulation = F5Kernels.Linear(siluT, 1, dim, bw.AttnNormLinearWeight, bw.AttnNormLinearBias, dim * 6);

        var shiftMsa = new float[dim]; var scaleMsa = new float[dim]; var gateMsa = new float[dim];
        var shiftMlp = new float[dim]; var scaleMlp = new float[dim]; var gateMlp = new float[dim];
        Array.Copy(modulation, 0 * dim, shiftMsa, 0, dim);
        Array.Copy(modulation, 1 * dim, scaleMsa, 0, dim);
        Array.Copy(modulation, 2 * dim, gateMsa, 0, dim);
        Array.Copy(modulation, 3 * dim, shiftMlp, 0, dim);
        Array.Copy(modulation, 4 * dim, scaleMlp, 0, dim);
        Array.Copy(modulation, 5 * dim, gateMlp, 0, dim);

        var norm = F5Kernels.LayerNormNoAffine(x, t, dim);
        for (int ti = 0; ti < t; ti++)
        {
            int off = ti * dim;
            for (int d = 0; d < dim; d++) norm[off + d] = norm[off + d] * (1f + scaleMsa[d]) + shiftMsa[d];
        }

        var attnOut = Attention(w, bw, norm, t, rotaryCos, rotarySin);

        var xAfterAttn = new float[x.Length];
        for (int ti = 0; ti < t; ti++)
        {
            int off = ti * dim;
            for (int d = 0; d < dim; d++) xAfterAttn[off + d] = x[off + d] + gateMsa[d] * attnOut[off + d];
        }

        var ffNorm = F5Kernels.LayerNormNoAffine(xAfterAttn, t, dim);
        for (int ti = 0; ti < t; ti++)
        {
            int off = ti * dim;
            for (int d = 0; d < dim; d++) ffNorm[off + d] = ffNorm[off + d] * (1f + scaleMlp[d]) + shiftMlp[d];
        }

        var ffOut = FeedForward(bw, ffNorm, t);

        var output = new float[x.Length];
        for (int ti = 0; ti < t; ti++)
        {
            int off = ti * dim;
            for (int d = 0; d < dim; d++) output[off + d] = xAfterAttn[off + d] + gateMlp[d] * ffOut[off + d];
        }
        return output;
    }

    private static float[] FeedForward(F5DiTBlockWeights bw, float[] x, int t)
    {
        int dim = F5TtsWeights.HiddenDim;
        int ffn = F5TtsWeights.FfnDim;
        var h = F5Kernels.Linear(x, t, dim, bw.FfInWeight, bw.FfInBias, ffn);
        for (int i = 0; i < h.Length; i++) h[i] = F5Kernels.GeluTanh(h[i]);
        return F5Kernels.Linear(h, t, ffn, bw.FfOutWeight, bw.FfOutBias, dim);
    }

    /// <summary>Interleaved ("GPT-J style") RoPE q/k projection + multi-head self-attention -- see <see cref="F5Kernels.ApplyRotary"/>/<see cref="F5Kernels.MultiHeadSelfAttention"/> (shared with CosyVoice3's tensor-for-tensor-identical DiT).</summary>
    private static float[] Attention(F5TtsWeights w, F5DiTBlockWeights bw, float[] norm, int t, float[] rotaryCos, float[] rotarySin)
    {
        int dim = F5TtsWeights.HiddenDim;
        int heads = F5TtsWeights.NumHeads;
        int headDim = F5TtsWeights.HeadDim;

        var q = F5Kernels.Linear(norm, t, dim, bw.ToQWeight, bw.ToQBias, dim);
        var k = F5Kernels.Linear(norm, t, dim, bw.ToKWeight, bw.ToKBias, dim);
        var v = F5Kernels.Linear(norm, t, dim, bw.ToVWeight, bw.ToVBias, dim);

        F5Kernels.ApplyRotary(q, t, heads, headDim, rotaryCos, rotarySin);
        F5Kernels.ApplyRotary(k, t, heads, headDim, rotaryCos, rotarySin);

        var context = F5Kernels.MultiHeadSelfAttention(q, k, v, t, heads, headDim);
        return F5Kernels.Linear(context, t, dim, bw.ToOutWeight, bw.ToOutBias, dim);
    }
}
