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
/// <para><b>Input composition confirmed directly from `examples/cosyvoice.cpp`</b>
/// (`InputEmbedding::build_cgraph`, `cosyvoice-graph.cpp` line ~278, and its caller
/// `DiT::build_cgraph` line ~446): `x = concat(x, cond, text_embed, spks)` in exactly that
/// order, where `text_embed` is CosyVoice's own repurposing of a parameter slot inherited
/// from a text-conditioned DiT template -- it actually receives `mu` (the flow's own
/// upsampled 80-dim speech-token embedding from `CausalMaskedDiffWithDiT::
/// build_cgraph_encode`'s `pre_lookahead_layer` + 2x upsample, NOT a text embedding at all;
/// CosyVoice3 has no Conformer flow encoder, see this class's earlier doc history in
/// docs/audio-review-progress.md). `cond` is the padded reference/prompt mel (`conds` in that
/// same function). `spks` is `spk_embed_affine_layer(l2_norm(campplus_embedding))`,
/// broadcast (`ggml_repeat`) across every frame before concatenation, not per-frame data.
/// All four are `MelDim` (80) wide, concatenating to the confirmed 320-wide `input_embed.
/// proj` input.</para>
/// </summary>
public static class CosyVoice3DiTModel
{
    /// <summary>
    /// Full DiT forward pass. x/cond/mu/spks are channel-last [numFrames, MelDim] (spks is
    /// logically per-utterance but passed pre-broadcast to every frame here, matching
    /// `ggml_repeat`'s effect in the real graph -- callers should broadcast a single [MelDim]
    /// speaker vector across all frames before calling). Returns predicted velocity,
    /// [numFrames, MelDim].
    /// </summary>
    public static float[] ForwardVelocity(CosyVoice3DiTWeights w, float[] x, float[] cond, float[] mu, float[] spks, float timestep, int numFrames)
    {
        var h = InputEmbed(w, x, cond, mu, spks, numFrames);
        return RunBackbone(w, h, timestep, numFrames);
    }

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

    /// <summary>
    /// concat(x, cond, mu, spks) [confirmed order/composition, see this class's doc comment]
    /// -> proj -> + ConvPositionEmbedding(kernel=31, groups=16). x/cond/mu/spks are each
    /// channel-last [numFrames, MelDim]; spks is expected already broadcast to every frame.
    /// </summary>
    public static float[] InputEmbed(CosyVoice3DiTWeights w, float[] x, float[] cond, float[] mu, float[] spks, int numFrames)
    {
        int melDim = CosyVoice3DiTWeights.MelDim;
        int concatDim = melDim * 4;
        var concatInput = new float[numFrames * concatDim];
        for (int ti = 0; ti < numFrames; ti++)
        {
            int outOff = ti * concatDim;
            int melOff = ti * melDim;
            Array.Copy(x, melOff, concatInput, outOff, melDim);
            Array.Copy(cond, melOff, concatInput, outOff + melDim, melDim);
            Array.Copy(mu, melOff, concatInput, outOff + 2 * melDim, melDim);
            Array.Copy(spks, melOff, concatInput, outOff + 3 * melDim, melDim);
        }

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
