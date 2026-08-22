using System;
using OpenTail.Stingray.Audio.F5TTS;

namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// CosyVoice3's flow-matching DiT backbone. Ports F5-TTS's `DiTBlock`/`DiT.forward` math
/// directly (`F5TTS/F5DiTBlock.cs`, `F5TTS/F5DiTModel.cs` -- real, golden-verified against the
/// actual PyTorch reference) since the two are tensor-for-tensor architecturally identical
/// (confirmed this session, see `CosyVoice3DiTWeights.cs`'s doc comment) -- reuses
/// `F5TTS.F5Kernels`/`F5TTS.F5RotaryEmbedding` directly (already pipeline-agnostic static
/// utilities) rather than re-deriving the same Linear/SiLU/LayerNorm/RoPE math a third time.
///
/// <para><b>NOT a complete pipeline yet.</b> Only the confirmed-identical pieces are
/// implemented: <see cref="RunBackbone"/> takes an ALREADY-embedded hidden state (post
/// `input_embed.proj` + `ConvPositionEmbedding`, i.e. what F5-TTS's `InputEmbedding.Forward`
/// would have produced) and runs the 22-block transformer + final `norm_out`/`proj_out`,
/// returning the predicted velocity in mel-space. The `input_embed` stage itself (concatenate
/// which real tensors into the 320-dim vector `proj` expects: F5's analog is
/// `[x, cond, text_embed]`, but CosyVoice3 has no text embedding in the same sense -- likely
/// some combination of the noisy mel `x`, the flow's own upsampled token embedding, the
/// reference `conds` mel, and the speaker embedding, each contributing 80 dims to reach 320,
/// but this is NOT yet confirmed against `examples/cosyvoice.cpp`'s real estimator-input
/// construction code) is deliberately NOT implemented -- writing it from a plausible-sounding
/// guess would be exactly the failure mode this whole rebuild exists to avoid. See
/// docs/audio-review-progress.md's CosyVoice3 section for the concrete next step (read
/// `examples/cosyvoice.cpp`'s `CausalConditionalCFM`/`build_cgraph_one_step` and
/// `CausalMaskedDiffWithDiT::build_cgraph_encode` in full) before completing this.</para>
/// </summary>
public static class CosyVoice3DiTModel
{
    /// <summary>h is the already-embedded hidden state [numFrames, HiddenDim] (post input_embed -- see this class's doc comment for what's NOT yet implemented upstream of this call). Returns the predicted velocity in mel-space [numFrames, MelDim].</summary>
    public static float[] RunBackbone(CosyVoice3DiTWeights w, float[] h, float timestep, int numFrames)
    {
        int dim = CosyVoice3DiTWeights.HiddenDim;

        var tEmb = TimestepEmbedding(w, timestep);
        var (rotaryCos, rotarySin) = F5RotaryEmbedding.Precompute(RotaryInvFreq(), numFrames);

        for (int layer = 0; layer < w.NumLayers; layer++)
            h = DiTBlock(w.Blocks[layer], h, tEmb, numFrames, rotaryCos, rotarySin);

        var siluT = new float[dim];
        for (int d = 0; d < dim; d++) siluT[d] = F5Kernels.SiLU(tEmb[d]);
        var modulation = F5Kernels.Linear(siluT, 1, dim, w.NormOutLinearWeight, w.NormOutLinearBias, dim * 2);
        var scale = new float[dim];
        var shift = new float[dim];
        Array.Copy(modulation, 0, scale, 0, dim);
        Array.Copy(modulation, dim, shift, 0, dim);

        var normOut = F5Kernels.LayerNormNoAffine(h, numFrames, dim);
        for (int ti = 0; ti < numFrames; ti++)
        {
            int off = ti * dim;
            for (int d = 0; d < dim; d++) normOut[off + d] = normOut[off + d] * (1f + scale[d]) + shift[d];
        }

        return F5Kernels.Linear(normOut, numFrames, dim, w.ProjOutWeight, w.ProjOutBias, CosyVoice3DiTWeights.MelDim);
    }

    /// <summary>proj(concat) + ConvPositionEmbedding(kernel=31, groups=16) -- the mechanical part of F5's InputEmbedding that IS confirmed (kernel/groups/dims match exactly), given an already-formed concatenated input of the right width.</summary>
    public static float[] InputEmbed(CosyVoice3DiTWeights w, float[] concatInput, int concatDim, int numFrames)
    {
        int hidden = CosyVoice3DiTWeights.HiddenDim;
        var h = F5Kernels.Linear(concatInput, numFrames, concatDim, w.InputProjWeight, w.InputProjBias, hidden);

        var pos = F5Kernels.GroupedConv1dSamePad(h, numFrames, hidden, w.ConvPos1Weight, w.ConvPos1Bias, CosyVoice3DiTWeights.ConvPosKernel, CosyVoice3DiTWeights.ConvPosGroups);
        for (int i = 0; i < pos.Length; i++) pos[i] = F5Kernels.Mish(pos[i]);
        pos = F5Kernels.GroupedConv1dSamePad(pos, numFrames, hidden, w.ConvPos2Weight, w.ConvPos2Bias, CosyVoice3DiTWeights.ConvPosKernel, CosyVoice3DiTWeights.ConvPosGroups);
        for (int i = 0; i < pos.Length; i++) pos[i] = F5Kernels.Mish(pos[i]);

        var output = new float[h.Length];
        for (int i = 0; i < output.Length; i++) output[i] = h[i] + pos[i];
        return output;
    }

    private static float[] TimestepEmbedding(CosyVoice3DiTWeights w, float timestep)
    {
        int freqDim = CosyVoice3DiTWeights.TimeFreqDim;
        int halfDim = freqDim / 2;

        var sinusEmbed = new float[freqDim];
        float embConst = MathF.Log(10000f) / (halfDim - 1);
        for (int k = 0; k < halfDim; k++)
        {
            float freq = MathF.Exp(-k * embConst);
            float angle = 1000f * timestep * freq;
            sinusEmbed[k] = MathF.Sin(angle);
            sinusEmbed[halfDim + k] = MathF.Cos(angle);
        }

        var h = F5Kernels.Linear(sinusEmbed, 1, freqDim, w.TimeMlp0Weight, w.TimeMlp0Bias, CosyVoice3DiTWeights.HiddenDim);
        for (int i = 0; i < h.Length; i++) h[i] = F5Kernels.SiLU(h[i]);
        return F5Kernels.Linear(h, 1, CosyVoice3DiTWeights.HiddenDim, w.TimeMlp2Weight, w.TimeMlp2Bias, CosyVoice3DiTWeights.HiddenDim);
    }

    /// <summary>Same RoPE base/formula F5-TTS's `F5RotaryEmbedding` uses (theta=10000, standard `x_transformers` convention) -- confirmed applicable since head_dim (64) matches exactly; not yet cross-checked against `examples/cosyvoice.cpp`'s own RoPE construction, flagged alongside the input_embed gap above.</summary>
    private static float[] RotaryInvFreq()
    {
        int halfHead = CosyVoice3DiTWeights.HeadDim / 2;
        var invFreq = new float[halfHead];
        for (int k = 0; k < halfHead; k++)
            invFreq[k] = 1f / MathF.Pow(10000f, (float)(2 * k) / CosyVoice3DiTWeights.HeadDim);
        return invFreq;
    }

    private static float[] DiTBlock(CosyVoice3DiTBlockWeights bw, float[] x, float[] tEmb, int t, float[] rotaryCos, float[] rotarySin)
    {
        int dim = CosyVoice3DiTWeights.HiddenDim;

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

        var attnOut = Attention(bw, norm, t, rotaryCos, rotarySin);

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

    private static float[] FeedForward(CosyVoice3DiTBlockWeights bw, float[] x, int t)
    {
        int dim = CosyVoice3DiTWeights.HiddenDim;
        int ffn = CosyVoice3DiTWeights.FfnDim;
        var h = F5Kernels.Linear(x, t, dim, bw.FfInWeight, bw.FfInBias, ffn);
        for (int i = 0; i < h.Length; i++) h[i] = F5Kernels.GeluTanh(h[i]);
        return F5Kernels.Linear(h, t, ffn, bw.FfOutWeight, bw.FfOutBias, dim);
    }

    private static float[] Attention(CosyVoice3DiTBlockWeights bw, float[] norm, int t, float[] rotaryCos, float[] rotarySin)
    {
        int dim = CosyVoice3DiTWeights.HiddenDim;
        int heads = CosyVoice3DiTWeights.NumHeads;
        int headDim = CosyVoice3DiTWeights.HeadDim;
        float scale = 1f / MathF.Sqrt(headDim);

        var q = F5Kernels.Linear(norm, t, dim, bw.ToQWeight, bw.ToQBias, dim);
        var k = F5Kernels.Linear(norm, t, dim, bw.ToKWeight, bw.ToKBias, dim);
        var v = F5Kernels.Linear(norm, t, dim, bw.ToVWeight, bw.ToVBias, dim);

        ApplyRotary(q, t, heads, headDim, rotaryCos, rotarySin);
        ApplyRotary(k, t, heads, headDim, rotaryCos, rotarySin);

        var context = new float[t * dim];
        System.Threading.Tasks.Parallel.For(0, heads, h =>
        {
            int hOff = h * headDim;
            var scores = new float[t];
            for (int i = 0; i < t; i++)
            {
                int qOff = i * dim + hOff;
                for (int j = 0; j < t; j++)
                {
                    int kOff = j * dim + hOff;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += q[qOff + d] * k[kOff + d];
                    scores[j] = dot * scale;
                }
                F5Kernels.SoftmaxInPlace(scores, 0, t);

                int cOff = i * dim + hOff;
                for (int j = 0; j < t; j++)
                {
                    float p = scores[j];
                    if (p == 0f) continue;
                    int vOff = j * dim + hOff;
                    for (int d = 0; d < headDim; d++) context[cOff + d] += p * v[vOff + d];
                }
            }
        });

        return F5Kernels.Linear(context, t, dim, bw.ToOutWeight, bw.ToOutBias, dim);
    }

    private static void ApplyRotary(float[] x, int t, int heads, int headDim, float[] rotaryCos, float[] rotarySin)
    {
        int dim = heads * headDim;
        int halfHead = headDim / 2;
        for (int ti = 0; ti < t; ti++)
        {
            int angleBase = ti * halfHead;
            for (int h = 0; h < heads; h++)
            {
                int hOff = ti * dim + h * headDim;
                for (int k = 0; k < halfHead; k++)
                {
                    float cos = rotaryCos[angleBase + k];
                    float sin = rotarySin[angleBase + k];
                    float x0 = x[hOff + 2 * k];
                    float x1 = x[hOff + 2 * k + 1];
                    x[hOff + 2 * k] = x0 * cos - x1 * sin;
                    x[hOff + 2 * k + 1] = x1 * cos + x0 * sin;
                }
            }
        }
    }
}
