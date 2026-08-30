
namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// F5-TTS's `DiTBlock` (modules.py): AdaLN-Zero modulated self-attention (with interleaved
/// "GPT-J style" RoPE, `x_transformers`'s `RotaryEmbedding`/`apply_rotary_pos_emb` convention --
/// NOT the split-half "rotate_half" convention some other RoPE implementations use, confirmed by
/// reading the actual installed `x_transformers` package source) + gated FFN (tanh-approx GELU).
/// This checkpoint has no qk_norm and attn_mask_enabled=False (single-utterance inference, no
/// batch padding to mask), so those code paths are simply omitted rather than implemented unused.
///
/// <para><b>Critical, previously-missed real config value</b>: the real, official `F5TTS_Base`
/// checkpoint's own `F5TTS_Base.yaml` sets `pe_attn_head: 1` -- RoPE is applied to only the FIRST
/// attention head, NOT uniformly to all 16 (confirmed directly in `modules.py`'s `AttnProcessor.
/// forward`: `query[:, :pn, :, :] = apply_rotary_pos_emb(...)` when `pe_attn_head` is set). This
/// checkpoint's `model.safetensors` was independently confirmed byte-identical to the canonical
/// `SWivid/F5-TTS`/`F5TTS_Base/model_1200000.safetensors` release (not a mismatched/wrong file).
/// Applying RoPE to all heads (this port's original assumption, matching the DIFFERENT
/// `F5TTS_v1_Base` checkpoint's real `pe_attn_head: null`) still runs without error -- same
/// tensor shapes -- but silently computes structurally wrong attention for 15 of 16 heads,
/// producing fluent-sounding but content-incorrect speech (confirmed: this exact symptom
/// persisted identically through the real, unmodified, official Python reference across two
/// independently-pinned dependency environments before this was found -- it was never a C#-
/// specific porting bug). See `docs/audio-review-progress.md`'s F5-TTS entry for the full
/// investigation trail.</para>
/// </summary>
public static class F5DiTBlock
{
    public static float[] Forward(F5TtsWeights w, F5DiTBlockWeights bw, float[] x, float[] tEmb, int t, float[] rotaryCos, float[] rotarySin)
    {
        int dim = F5TtsWeights.HiddenDim;

        var siluT = new float[dim];
        for (int d = 0; d < dim; d++) siluT[d] = F5Kernels.SiLU(tEmb[d]);
        var modulation = F5Kernels.Linear(siluT, 1, dim, bw.AttnNormLinearWeight, bw.AttnNormLinearBias, dim * 6);

        var norm = F5Kernels.LayerNormNoAffine(x, t, dim);
        ApplyAffineModulationSlice(norm, norm, modulation, 1 * dim, 0 * dim, t, dim);

        var attnOut = Attention(w, bw, norm, t, rotaryCos, rotarySin);

        var xAfterAttn = new float[x.Length];
        ApplyGatedResidualSlice(xAfterAttn, x, modulation, 2 * dim, attnOut, t, dim);

        var ffNorm = F5Kernels.LayerNormNoAffine(xAfterAttn, t, dim);
        ApplyAffineModulationSlice(ffNorm, ffNorm, modulation, 4 * dim, 3 * dim, t, dim);

        var ffOut = FeedForward(bw, ffNorm, t);

        var output = new float[x.Length];
        ApplyGatedResidualSlice(output, xAfterAttn, modulation, 5 * dim, ffOut, t, dim);
        return output;
    }

    private static unsafe void ApplyAffineModulationSlice(float[] dst, float[] src, float[] modulation, int scaleOffset, int shiftOffset, int t, int dim)
    {
        int vecSize = System.Numerics.Vector<float>.Count;
        fixed (float* dp = dst, sp = src, mp = modulation)
        {
            float* dpLocal = dp;
            float* spLocal = sp;
            float* scpLocal = mp + scaleOffset;
            float* shpLocal = mp + shiftOffset;
            Parallel.For(0, t, ti =>
            {
                int off = ti * dim;
                float* dRow = dpLocal + off;
                float* sRow = spLocal + off;
                int d = 0;
                for (; d <= dim - vecSize; d += vecSize)
                {
                    var vs = new System.Numerics.Vector<float>(new ReadOnlySpan<float>(sRow + d, vecSize));
                    var vScale = new System.Numerics.Vector<float>(new ReadOnlySpan<float>(scpLocal + d, vecSize));
                    var vShift = new System.Numerics.Vector<float>(new ReadOnlySpan<float>(shpLocal + d, vecSize));
                    var vr = vs * (System.Numerics.Vector<float>.One + vScale) + vShift;
                    vr.CopyTo(new Span<float>(dRow + d, vecSize));
                }
                for (; d < dim; d++) dRow[d] = sRow[d] * (1f + scpLocal[d]) + shpLocal[d];
            });
        }
    }

    private static unsafe void ApplyGatedResidualSlice(float[] dst, float[] residual, float[] modulation, int gateOffset, float[] update, int t, int dim)
    {
        int vecSize = System.Numerics.Vector<float>.Count;
        fixed (float* dp = dst, rp = residual, mp = modulation, up = update)
        {
            float* dpLocal = dp;
            float* rpLocal = rp;
            float* gpLocal = mp + gateOffset;
            float* upLocal = up;
            Parallel.For(0, t, ti =>
            {
                int off = ti * dim;
                float* dRow = dpLocal + off;
                float* rRow = rpLocal + off;
                float* uRow = upLocal + off;
                int d = 0;
                for (; d <= dim - vecSize; d += vecSize)
                {
                    var vr = new System.Numerics.Vector<float>(new ReadOnlySpan<float>(rRow + d, vecSize));
                    var vg = new System.Numerics.Vector<float>(new ReadOnlySpan<float>(gpLocal + d, vecSize));
                    var vu = new System.Numerics.Vector<float>(new ReadOnlySpan<float>(uRow + d, vecSize));
                    var vRes = vr + vg * vu;
                    vRes.CopyTo(new Span<float>(dRow + d, vecSize));
                }
                for (; d < dim; d++) dRow[d] = rRow[d] + gpLocal[d] * uRow[d];
            });
        }
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

        // Real F5TTS_Base.yaml: pe_attn_head=1 -- RoPE applies to only the FIRST attention head, not all 16 (confirmed via modules.py's AttnProcessor.forward, `query[:, :pn, :, :] = apply_rotary_pos_emb(...)`).
        F5Kernels.ApplyRotary(q, t, heads, headDim, rotaryCos, rotarySin, numRopeHeads: 1);
        F5Kernels.ApplyRotary(k, t, heads, headDim, rotaryCos, rotarySin, numRopeHeads: 1);

        var context = F5Kernels.MultiHeadSelfAttention(q, k, v, t, heads, headDim);
        return F5Kernels.Linear(context, t, dim, bw.ToOutWeight, bw.ToOutBias, dim);
    }
}
