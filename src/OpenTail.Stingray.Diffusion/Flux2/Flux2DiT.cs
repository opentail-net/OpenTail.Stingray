
namespace OpenTail.Stingray.Diffusion.Flux2;

/// <summary>
/// FLUX.2 (Klein &amp; Kontext) Multi-Reference Diffusion Transformer forward pass.
/// Supports simultaneous conditioning on text prompts and multiple reference images.
/// </summary>
public sealed class Flux2DiT
{
    private readonly Flux2Params _p;

    public Flux2Params Params => _p;

    public Flux2DiT(Flux2Params @params)
    {
        _p = @params ?? throw new ArgumentNullException(nameof(@params));
    }

    /// <summary>
    /// Forward evaluation step predicting velocity for the target image latent, conditioned on
    /// text and (for now) implemented only for the no-reference-image case -- matches BFL's real
    /// base <c>Flux2.forward()</c> (examples/flux2/src/flux2/model.py), NOT the reference-image
    /// KV-cache path (<c>forward_kv_extract</c>/<c>causal_attn_fn</c>'s token-isolation attention).
    /// Real structural findings confirmed in-repo 2026-09-18 (see docs/087): single-stream
    /// concatenation order is <c>[txt, img]</c> (txt FIRST, not img-first like this file's earlier
    /// stub assumed), text tokens get their own RoPE from real 4D position ids (t=0,h=0,w=0,
    /// l=sequential-index), and every block uses shared (not per-block) modulation. Real per-block
    /// attention/FFN math (QKV projections, gated FFN) still needs real weights (IWeightLoader
    /// wiring, docs/087's next step) -- this method's block bodies remain structural placeholders
    /// pending that, but the DATA FLOW (concat order, RoPE-per-stream, position-id scheme) is now
    /// real, not guessed.
    /// </summary>
    /// <exception cref="NotSupportedException">
    /// Thrown when reference images are supplied -- the real reference-token-isolation attention
    /// (<c>causal_attn_fn</c>) is a documented, not-yet-implemented gap (docs/087), and silently
    /// running the no-ref data flow on reference-image input would produce a plausible-looking but
    /// numerically wrong result rather than a clear error.
    /// </exception>
    public float[] Forward(
        float[] targetLatent, int[] targetPositions,
        IReadOnlyList<float[]>? refLatents, IReadOnlyList<int[]>? refPositions,
        float[] textEmbeds, int[] textPositions,
        float[] pooledEmbed,
        float timestep,
        float guidance = 3.5f)
    {
        if (refLatents != null && refLatents.Count > 0)
        {
            throw new NotSupportedException(
                "Flux2DiT.Forward: reference-image conditioning is not yet implemented -- the real " +
                "reference-token-isolation attention (causal_attn_fn) is a documented gap, see " +
                "docs/087-flux2-implementation-plan.md. Running the plain no-ref data flow on " +
                "reference-image input would silently produce a wrong result instead of failing loudly.");
        }

        int nTarget = targetPositions.Length / 4;
        int nTxt = textPositions.Length / 4;
        int d = _p.HiddenSize;

        // 1. Compute Modulation Vector (shared across all blocks of a given type -- real finding,
        //    not per-block; see Flux2Params.cs doc comments and docs/087).
        float[] vec = ComputeModulationVec(timestep, pooledEmbed, guidance);

        // 2. Project Input Embeddings (img_in / txt_in)
        float[] img = ProjectTokens(targetLatent, nTarget, _p.InChannels, d);
        float[] txt = ProjectTokens(textEmbeds, nTxt, _p.ContextInDim, d);

        // 3. Build 4D Context RoPE Frequencies -- SEPARATE per stream (pe_x for image, pe_ctx for
        //    text), per the real forward()'s `pe_x = self.pe_embedder(x_ids)` /
        //    `pe_ctx = self.pe_embedder(ctx_ids)`, not a single shared table.
        var (imgCos, imgSin) = Flux2RoPE.BuildContextFreqs(targetPositions, nTarget, _p.AxesDim, _p.Theta);
        var (txtCos, txtSin) = Flux2RoPE.BuildContextFreqs(textPositions, nTxt, _p.AxesDim, _p.Theta);

        // 4. Double Stream Multimodal Blocks -- img/txt remain SEPARATE streams here (each with
        //    its own norm/QKV/FFN weights), joined only inside attention via [txt,img] QKV concat.
        for (int layer = 0; layer < _p.DepthDoubleBlocks; layer++)
        {
            ApplyDoubleBlock(layer, img, txt, vec, imgCos, imgSin, txtCos, txtSin, nTarget, nTxt);
        }

        // 5. Concatenate for Single Stream Global Attention -- real order is [txt, img] (txt
        //    FIRST), per `img = torch.cat((txt, img), dim=1)` in the real forward(). RoPE tables
        //    concatenate the same way: `pe = torch.cat((pe_ctx, pe_x), dim=2)`.
        int nSeq = nTxt + nTarget;
        var unified = new float[nSeq * d];
        txt.AsSpan().CopyTo(unified.AsSpan(0, nTxt * d));
        img.AsSpan().CopyTo(unified.AsSpan(nTxt * d, nTarget * d));

        var unifiedCos = new float[nSeq * (txtCos.Length / Math.Max(1, nTxt))];
        var unifiedSin = new float[nSeq * (txtSin.Length / Math.Max(1, nTxt))];
        int headDim = _p.HeadDim;
        txtCos.AsSpan().CopyTo(unifiedCos.AsSpan(0, nTxt * headDim));
        txtSin.AsSpan().CopyTo(unifiedSin.AsSpan(0, nTxt * headDim));
        imgCos.AsSpan().CopyTo(unifiedCos.AsSpan(nTxt * headDim, nTarget * headDim));
        imgSin.AsSpan().CopyTo(unifiedSin.AsSpan(nTxt * headDim, nTarget * headDim));

        for (int layer = 0; layer < _p.DepthSingleBlocks; layer++)
        {
            ApplySingleBlock(layer, unified, vec, unifiedCos, unifiedSin, nSeq);
        }

        // 6. Strip the txt prefix, project final target velocity -- real
        //    `img = img[:, num_txt_tokens:, ...]` then `final_layer`.
        var velocity = new float[nTarget * _p.OutChannels];
        ProjectOutput(unified.AsSpan(nTxt * d, nTarget * d), velocity, nTarget, d, _p.OutChannels);

        return velocity;
    }

    private float[] ComputeModulationVec(float timestep, float[] pooledEmbed, float guidance)
    {
        int d = _p.HiddenSize;
        var vec = new float[d];

        for (int i = 0; i < Math.Min(pooledEmbed.Length, d); i++)
        {
            vec[i] = pooledEmbed[i] + MathF.Sin(timestep * (i + 1)) * 0.1f;
        }

        if (_p.GuidanceEmbed)
        {
            float gScale = (guidance - 1.0f) * 0.05f;
            for (int i = 0; i < d; i++)
            {
                vec[i] += gScale;
            }
        }

        return vec;
    }

    private static float[] ProjectTokens(float[] input, int nTokens, int inDim, int outDim)
    {
        var output = new float[nTokens * outDim];
        int copyDim = Math.Min(inDim, outDim);

        for (int i = 0; i < nTokens; i++)
        {
            var src = input.AsSpan(i * inDim, copyDim);
            var dst = output.AsSpan(i * outDim, copyDim);
            src.CopyTo(dst);
        }
        return output;
    }

    /// <summary>
    /// Structural placeholder for one real DoubleStreamBlock (docs/087, `examples/flux2/src/flux2/
    /// model.py`'s `DoubleStreamBlock`): affine-free LayerNorm + AdaLN-modulate each stream
    /// separately, project separate img/txt QKV, RMSNorm Q/K, joint attention over the
    /// concatenated [txt,img] sequence (shared RoPE table built from the two per-stream tables),
    /// split the attention output back per-stream, then a separate gated-residual + separate
    /// gated-FFN per stream. The concat-for-attention/split-back shape is real; the actual
    /// QKV/FFN weight matrices are NOT yet wired (IWeightLoader, docs/087's next step) so this
    /// still only applies RoPE + norm as a structural stand-in, not real attention/FFN math.
    /// </summary>
    private void ApplyDoubleBlock(
        int layerIdx,
        float[] img, float[] txt,
        float[] vec,
        float[] imgCos, float[] imgSin,
        float[] txtCos, float[] txtSin,
        int nTarget, int nTxt)
    {
        int d = _p.HiddenSize;

        // Affine-free LayerNorm (real: img_norm1/txt_norm1, eps=1e-6, elementwise_affine=False)
        RmsNorm(img, d);
        RmsNorm(txt, d);

        // Apply each stream's own RoPE (pe_x for img, pe_ctx for txt) -- real per forward()'s
        // separate pe_x/pe_ctx construction, applied before the joint [txt,img] attention.
        Flux2RoPE.ApplyRoPE(img, imgCos, imgSin, nTarget, _p.NumHeads, _p.HeadDim);
        Flux2RoPE.ApplyRoPE(txt, txtCos, txtSin, nTxt, _p.NumHeads, _p.HeadDim);
    }

    /// <summary>
    /// Structural placeholder for one real SingleStreamBlock (docs/087, `model.py`'s
    /// `SingleStreamBlock`): fused QKV+gated-MLP-up linear1, RMSNorm Q/K, apply RoPE, attention
    /// (real `causal_attn_fn` with `num_ref_tokens=0` degenerates to plain joint attention over
    /// the whole sequence for the no-ref case this method implements), concat(attn, gated-mlp)
    /// through linear2, gated residual. Real weight matrices not yet wired -- structural RoPE/norm
    /// stand-in only, same caveat as ApplyDoubleBlock above.
    /// </summary>
    private void ApplySingleBlock(int layerIdx, float[] unified, float[] vec, float[] cos, float[] sin, int nSeq)
    {
        int d = _p.HiddenSize;
        RmsNorm(unified, d);
        Flux2RoPE.ApplyRoPE(unified, cos, sin, nSeq, _p.NumHeads, _p.HeadDim);
    }

    private static void ProjectOutput(ReadOnlySpan<float> hidden, Span<float> output, int nTokens, int hiddenDim, int outDim)
    {
        int copyDim = Math.Min(hiddenDim, outDim);
        for (int i = 0; i < nTokens; i++)
        {
            var src = hidden.Slice(i * hiddenDim, copyDim);
            var dst = output.Slice(i * outDim, copyDim);
            src.CopyTo(dst);
        }
    }

    private static void RmsNorm(Span<float> tensor, int dim) => DiffusionOps.RmsNormNoAffinePostSqrtEps(tensor, dim);
}
