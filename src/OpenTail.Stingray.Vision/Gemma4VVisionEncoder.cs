using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// The Gemma 4 E-model (E4B) <c>gemma4v</c> ViT encoder + token-reduction pooler +
/// projector. Turns a preprocessed image (<see cref="Gemma4VImagePreprocessor"/>) into LLM
/// soft-token embeddings.
///
/// Ported stage-by-stage from the real reference (<c>examples/llama.cpp/llama.cpp</c>,
/// <c>tools/mtmd/models/gemma4v.cpp</c> plus the shared <c>clip_graph::build_vit</c>/
/// <c>build_attn</c>/<c>build_ffn</c>/<c>build_mm</c> helpers in <c>tools/mtmd/clip.cpp</c>), NOT
/// guessed from tensor names -- see <c>docs/03-gemma4-e4b-vision-plan.md</c> for the full
/// derivation and the direct verification against the real E4B mmproj
/// (<c>models/gemma-4-E4B-it-mmproj.gguf</c>) that pinned every constant below.
///
/// <para>Per-patch pipeline: <c>patches = 2*x-1</c> (the mmproj's declared mean=[0,0,0]/std=[1,1,1]
/// preprocessing emits [0,1]; the encoder itself does the range fix) -&gt; conv2d patch embed
/// (stride=patch_size, no bias) -&gt; add learned 2D position embeddings (two stacked per-axis
/// lookup tables, NOT a square grid or RoPE) -&gt; 16 transformer blocks -&gt; average-pool
/// token reduction -&gt; scale by sqrt(embd) -&gt; unweighted RMSNorm -&gt; linear projection.</para>
///
/// <para>Per block (sandwich-normed, NOT CLIP-standard pre-norm-only): RMSNorm(ln1) -&gt; separate
/// (not fused) Q/K/V projections, each a real INT8-range-clamped linear (<c>build_mm</c>'s
/// <c>clamp_info_map</c> -- confirmed present for every one of the seven per-block linears in the
/// real mmproj) -&gt; per-head QK-norm (weighted RMSNorm, applied AFTER splitting into heads)
/// -&gt; 2D RoPE on Q/K only (NEOX convention, <c>rope_theta=100</c> -- a `gemma4v`-specific
/// hardcoded constant, NOT the paired text model's own theta -- leading half of each head's
/// dim rotated using the patch's COLUMN as position, trailing half using its ROW) -&gt; RMSNorm on
/// V (gemma4v-specific, unweighted, applied per-head -- easy to miss since it lives in the SHARED
/// <c>build_vit</c> helper gated on projector type, not in <c>gemma4v.cpp</c> itself) -&gt;
/// UNSCALED attention (<c>kq_scale=1.0</c>, not the usual <c>1/sqrt(head_dim)</c>) -&gt; output
/// projection (clamped) -&gt; RMSNorm(attn_post_norm) -&gt; residual -&gt; RMSNorm(ln2) -&gt;
/// gated FFN (clamped gate/up/down; quick-GELU <c>x*sigmoid(1.702x)</c>, NOT the tanh-approximation
/// <see cref="SimdKernels.GeluInPlace"/> already used elsewhere in this codebase -- the mmproj
/// declares neither <c>use_gelu</c> nor <c>use_silu</c>, so <c>clip.cpp</c>'s
/// <c>FFN_GELU_QUICK</c> default applies) -&gt; RMSNorm(ffn_post_norm) -&gt; residual. There is no
/// final post-layernorm for this projector (no <c>v.post_ln</c> tensor).</para>
///
/// <para><b>Unverified end-to-end</b>: every constant and mechanism here is sourced from the real
/// reference graph and the real mmproj's tensor/metadata inventory, not inferred, but there is
/// still no working oracle on this machine to run <c>gemma4v</c> end-to-end for numerical parity
/// (the paired text architecture <c>gemma4</c> is not admitted by the local llama.cpp build). Do
/// not treat this as parity-verified until that comparison exists.</para>
/// </summary>
public sealed unsafe class Gemma4VVisionEncoder
{
    private readonly Gemma4VVisionModel _m;
    private readonly int _embd, _heads, _headDim, _halfHeadDim, _quarterHeadDim, _ffLen, _blockCount, _patch, _imageSize, _projDim;
    private readonly float _eps;
    private readonly int _gridSize;   // patches per side (imageSize / patch)
    private readonly int _posSize;    // per-axis learned position-table length
    private readonly int _nMerge;

    private readonly byte* _patchEmbdW;    // F32 [embd, patchVec], patchVec = patch*patch*3
    private readonly float[] _posTable;    // [2 * posSize * embd] : x-table then y-table
    private readonly byte* _mmProjW;       // BF16 [embd, projDim] -- NOT clamped

    private sealed class BlockData
    {
        public required float[] Ln1, Ln2, AttnQNorm, AttnKNorm, AttnPostNorm, FfnPostNorm;
        public required byte* AttnQ, AttnK, AttnV, AttnOut, FfnGate, FfnUp, FfnDown;
        public required Gemma4VClamp AttnQClamp, AttnKClamp, AttnVClamp, AttnOutClamp, FfnGateClamp, FfnUpClamp, FfnDownClamp;
    }

    private readonly BlockData[] _blocks;

    /// <remarks>
    /// Borrows <paramref name="model"/>'s memory-mapped weights (caches raw pointers into them),
    /// so <paramref name="model"/> must outlive this encoder and must not be disposed before the
    /// last <see cref="Forward"/> call. <see cref="Forward"/> guards via
    /// <see cref="Gemma4VVisionModel.EnsureNotDisposed"/>.
    /// </remarks>
    public Gemma4VVisionEncoder(Gemma4VVisionModel model)
    {
        _m = model;
        _embd = model.EmbeddingLength;
        _heads = model.HeadCount;
        _headDim = _embd / _heads;
        _halfHeadDim = _headDim / 2;
        _quarterHeadDim = _headDim / 4;
        _ffLen = model.FeedForwardLength;
        _blockCount = model.BlockCount;
        _patch = model.PatchSize;
        _imageSize = model.ImageSize;
        _projDim = model.ProjectionDim;
        _eps = model.LayerNormEps;
        _gridSize = _imageSize / _patch;
        _posSize = (int)model.PositionEmbedding.Dimensions[1];
        _nMerge = model.NMerge;

        _patchEmbdW = model.Gguf.GetTensorDataPtr(model.PatchEmbedding);
        _posTable = model.LoadFloats(model.PositionEmbedding);
        _mmProjW = model.Gguf.GetTensorDataPtr(model.InputProjection);

        _blocks = new BlockData[_blockCount];
        for (var i = 0; i < _blockCount; i++)
        {
            var b = model.Blocks[i];
            _blocks[i] = new BlockData
            {
                Ln1 = model.LoadFloats(b.Ln1),
                Ln2 = model.LoadFloats(b.Ln2),
                AttnQNorm = model.LoadFloats(b.AttnQNorm),
                AttnKNorm = model.LoadFloats(b.AttnKNorm),
                AttnPostNorm = model.LoadFloats(b.AttnPostNorm),
                FfnPostNorm = model.LoadFloats(b.FfnPostNorm),
                AttnQ = model.Gguf.GetTensorDataPtr(b.AttnQ),
                AttnK = model.Gguf.GetTensorDataPtr(b.AttnK),
                AttnV = model.Gguf.GetTensorDataPtr(b.AttnV),
                AttnOut = model.Gguf.GetTensorDataPtr(b.AttnOut),
                FfnGate = model.Gguf.GetTensorDataPtr(b.FfnGate),
                FfnUp = model.Gguf.GetTensorDataPtr(b.FfnUp),
                FfnDown = model.Gguf.GetTensorDataPtr(b.FfnDown),
                AttnQClamp = b.AttnQClamp,
                AttnKClamp = b.AttnKClamp,
                AttnVClamp = b.AttnVClamp,
                AttnOutClamp = b.AttnOutClamp,
                FfnGateClamp = b.FfnGateClamp,
                FfnUpClamp = b.FfnUpClamp,
                FfnDownClamp = b.FfnDownClamp,
            };
        }
    }

    /// <summary>Soft-token count this encoder produces for its fixed input grid.</summary>
    public int TokenCount => ((_gridSize - _nMerge) / _nMerge + 1) * ((_gridSize - _nMerge) / _nMerge + 1);

    /// <summary>
    /// Run the encoder. <paramref name="chw"/> is <see cref="Gemma4VImagePreprocessor"/>'s output:
    /// planar CHW, already resized to <c>ImageSize x ImageSize</c> and channel-normalized (values
    /// in [0,1] for this mmproj's declared mean=[0,0,0]/std=[1,1,1]). Returns
    /// <see cref="TokenCount"/> x <c>ProjectionDim</c> soft-token embeddings, row-major.
    /// </summary>
    public float[] Forward(ReadOnlySpan<float> chw)
    {
        _m.EnsureNotDisposed();
        var plane = _imageSize * _imageSize;
        if (chw.Length < 3L * plane)
            throw new ArgumentException($"chw length ({chw.Length}) is too small for a 3x{_imageSize}x{_imageSize} image.");

        var nPatches = _gridSize * _gridSize;
        var patchVec = _patch * _patch * 3;

        // 1. range fix (2*x-1) + im2col: per patch a patchVec vector, order [c*patch*patch + ky*patch + kx]
        //    (matches patch_embd.weight's ne=[patch,patch,3,embd] -- kx fastest, then ky, then c --
        //    so each output row (one embd channel) is contiguous over exactly this order).
        var patches = new float[nPatches * patchVec];
        for (var p = 0; p < nPatches; p++)
        {
            var pr = p / _gridSize;
            var pc = p % _gridSize;
            var baseY = pr * _patch;
            var baseX = pc * _patch;
            var dst = p * patchVec;
            for (var c = 0; c < 3; c++)
            {
                var cPlane = c * plane;
                var cOff = dst + c * _patch * _patch;
                for (var ky = 0; ky < _patch; ky++)
                {
                    var srcRow = cPlane + (baseY + ky) * _imageSize + baseX;
                    var dstRow = cOff + ky * _patch;
                    for (var kx = 0; kx < _patch; kx++)
                        patches[dstRow + kx] = 2f * chw[srcRow + kx] - 1f;
                }
            }
        }

        // 2. patch embed (F32 matvec, no bias) + learned 2D position embeddings
        var hidden = new float[nPatches * _embd];
        var yBase = _posSize * _embd;
        fixed (float* pPatches = patches, pHidden = hidden, pPos = _posTable)
        {
            for (var p = 0; p < nPatches; p++)
                SimdKernels.MatVec(pHidden + p * _embd, _patchEmbdW, pPatches + p * patchVec, _embd, patchVec, DType.Float32);
            for (var p = 0; p < nPatches; p++)
            {
                var pr = p / _gridSize;
                var pc = p % _gridSize;
                var row = pHidden + p * _embd;
                var xRow = pPos + pc * _embd;
                var yRow = pPos + yBase + pr * _embd;
                for (var c = 0; c < _embd; c++)
                    row[c] += xRow[c] + yRow[c];
            }
        }

        // 3. transformer blocks
        var normed = new float[_embd];
        var q = new float[nPatches * _embd];
        var k = new float[nPatches * _embd];
        var v = new float[nPatches * _embd];
        var attnConcat = new float[nPatches * _embd];
        var attnProj = new float[_embd];
        var scores = new float[nPatches];
        var ffnIn = new float[_embd];
        var gate = new float[_ffLen];
        var up = new float[_ffLen];
        var ffnOut = new float[_embd];
        var clampScratch = new float[Math.Max(_embd, _ffLen)];

        fixed (float* pHidden = hidden, pNormed = normed, pQ = q, pK = k, pV = v,
               pAttnConcat = attnConcat, pAttnProj = attnProj, pScores = scores,
               pFfnIn = ffnIn, pGate = gate, pUp = up, pFfnOut = ffnOut, pScratch = clampScratch)
        {
            for (var layer = 0; layer < _blockCount; layer++)
            {
                var blk = _blocks[layer];
                fixed (float* ln1 = blk.Ln1, ln2 = blk.Ln2, qNorm = blk.AttnQNorm, kNorm = blk.AttnKNorm,
                       attnPostNorm = blk.AttnPostNorm, ffnPostNorm = blk.FfnPostNorm)
                {
                    // ln1 -> Q/K/V (each independently clamped) -> per-head QK-norm -> 2D RoPE on Q/K
                    // -> unweighted per-head V-norm.
                    for (var p = 0; p < nPatches; p++)
                    {
                        var hp = pHidden + p * _embd;
                        SimdKernels.RmsNorm(pNormed, hp, ln1, _embd, _eps);

                        var qp = pQ + p * _embd;
                        var kp = pK + p * _embd;
                        var vp = pV + p * _embd;
                        ClampedMatVec(qp, blk.AttnQ, pNormed, _embd, _embd, blk.AttnQClamp, pScratch);
                        ClampedMatVec(kp, blk.AttnK, pNormed, _embd, _embd, blk.AttnKClamp, pScratch);
                        ClampedMatVec(vp, blk.AttnV, pNormed, _embd, _embd, blk.AttnVClamp, pScratch);

                        var col = p % _gridSize;
                        var row = p / _gridSize;
                        for (var h = 0; h < _heads; h++)
                        {
                            var qh = qp + h * _headDim;
                            var khh = kp + h * _headDim;
                            var vh = vp + h * _headDim;

                            SimdKernels.RmsNorm(qh, qh, qNorm, _headDim, _eps);
                            SimdKernels.RmsNorm(khh, khh, kNorm, _headDim, _eps);

                            ApplyRope2DHalf(qh, _quarterHeadDim, col, Gemma4VVisionModel.RopeTheta);
                            ApplyRope2DHalf(qh + _halfHeadDim, _quarterHeadDim, row, Gemma4VVisionModel.RopeTheta);
                            ApplyRope2DHalf(khh, _quarterHeadDim, col, Gemma4VVisionModel.RopeTheta);
                            ApplyRope2DHalf(khh + _halfHeadDim, _quarterHeadDim, row, Gemma4VVisionModel.RopeTheta);

                            SimdKernels.PureRmsNorm(vh, vh, _headDim, _eps);
                        }
                    }

                    // Bidirectional multi-head attention, unscaled (kq_scale = 1.0 for gemma4v).
                    for (var h = 0; h < _heads; h++)
                    {
                        var off = h * _headDim;
                        for (var i = 0; i < nPatches; i++)
                        {
                            var qi = pQ + i * _embd + off;
                            for (var j = 0; j < nPatches; j++)
                            {
                                var kj = pK + j * _embd + off;
                                float dot = 0f;
                                for (var d = 0; d < _headDim; d++) dot += qi[d] * kj[d];
                                pScores[j] = dot;
                            }
                            SimdKernels.SoftmaxInPlace(pScores, nPatches);

                            var outp = pAttnConcat + i * _embd + off;
                            for (var d = 0; d < _headDim; d++) outp[d] = 0f;
                            for (var j = 0; j < nPatches; j++)
                            {
                                var vj = pV + j * _embd + off;
                                var w = pScores[j];
                                for (var d = 0; d < _headDim; d++) outp[d] += w * vj[d];
                            }
                        }
                    }

                    // Output projection (clamped) -> attn_post_norm -> residual.
                    for (var p = 0; p < nPatches; p++)
                    {
                        ClampedMatVec(pAttnProj, blk.AttnOut, pAttnConcat + p * _embd, _embd, _embd, blk.AttnOutClamp, pScratch);
                        SimdKernels.RmsNorm(pAttnProj, pAttnProj, attnPostNorm, _embd, _eps);
                        var hp = pHidden + p * _embd;
                        for (var c = 0; c < _embd; c++) hp[c] += pAttnProj[c];
                    }

                    // ln2 -> gated FFN (quick-GELU, clamped gate/up/down) -> ffn_post_norm -> residual.
                    for (var p = 0; p < nPatches; p++)
                    {
                        var hp = pHidden + p * _embd;
                        SimdKernels.RmsNorm(pFfnIn, hp, ln2, _embd, _eps);

                        ClampedMatVec(pGate, blk.FfnGate, pFfnIn, _ffLen, _embd, blk.FfnGateClamp, pScratch);
                        SimdKernels.GeluQuickInPlace(pGate, _ffLen);
                        ClampedMatVec(pUp, blk.FfnUp, pFfnIn, _ffLen, _embd, blk.FfnUpClamp, pScratch);
                        for (var i = 0; i < _ffLen; i++) pGate[i] *= pUp[i];
                        ClampedMatVec(pFfnOut, blk.FfnDown, pGate, _embd, _ffLen, blk.FfnDownClamp, pScratch);

                        SimdKernels.RmsNorm(pFfnOut, pFfnOut, ffnPostNorm, _embd, _eps);
                        for (var c = 0; c < _embd; c++) hp[c] += pFfnOut[c];
                    }
                }
            }
        }

        // 4. token-reduction average pool (kernel = stride = NMerge, no padding -- a non-divisible
        // grid silently drops the trailing patches per axis, matching ggml_pool_2d exactly) ->
        // scale by sqrt(embd) -> unweighted RMSNorm (embedding_pre_projection_norm) -> projection
        // (NOT clamped -- mm.input_projection has no clamp tensors).
        var outSide = (_gridSize - _nMerge) / _nMerge + 1;
        var nOut = outSide * outSide;
        var pooled = new float[nOut * _embd];
        var scale = MathF.Sqrt(_embd);
        var kernelArea = (float)(_nMerge * _nMerge);
        for (var oy = 0; oy < outSide; oy++)
        {
            for (var ox = 0; ox < outSide; ox++)
            {
                var dst = (oy * outSide + ox) * _embd;
                for (var ky = 0; ky < _nMerge; ky++)
                {
                    for (var kx = 0; kx < _nMerge; kx++)
                    {
                        var srcPatch = (oy * _nMerge + ky) * _gridSize + (ox * _nMerge + kx);
                        var src = srcPatch * _embd;
                        for (var c = 0; c < _embd; c++)
                            pooled[dst + c] += hidden[src + c];
                    }
                }
                for (var c = 0; c < _embd; c++)
                    pooled[dst + c] = pooled[dst + c] / kernelArea * scale;
            }
        }

        var result = new float[nOut * _projDim];
        fixed (float* pPooled = pooled, pResult = result)
        {
            for (var t = 0; t < nOut; t++)
                SimdKernels.PureRmsNorm(pPooled + t * _embd, pPooled + t * _embd, _embd, _eps);
            for (var t = 0; t < nOut; t++)
                SimdKernels.MatVec(pResult + t * _projDim, _mmProjW, pPooled + t * _embd, _projDim, _embd, DType.BFloat16);
        }
        return result;
    }

    /// <summary>
    /// One <c>build_mm</c> linear: clamp input to [InputMin,InputMax], matvec (BF16 weight),
    /// clamp output to [OutputMin,OutputMax]. <paramref name="scratch"/> must hold at least
    /// <paramref name="cols"/> floats and is caller-owned so no per-call allocation is needed.
    /// </summary>
    private static void ClampedMatVec(float* dst, byte* weights, float* input, int rows, int cols, in Gemma4VClamp clamp, float* scratch)
    {
        for (var i = 0; i < cols; i++)
            scratch[i] = Math.Clamp(input[i], clamp.InputMin, clamp.InputMax);
        SimdKernels.MatVec(dst, weights, scratch, rows, cols, DType.BFloat16);
        for (var i = 0; i < rows; i++)
            dst[i] = Math.Clamp(dst[i], clamp.OutputMin, clamp.OutputMax);
    }

    /// <summary>
    /// 2D-RoPE half: NEOX pairs <c>(j, j+quarterDim)</c> for <c>j in [0, quarterDim)</c>, rotated
    /// by <c>position * ropeTheta^(-2j/(2*quarterDim))</c>. Applied twice per head -- once to the
    /// leading half with the patch's column as <paramref name="position"/>, once to the trailing
    /// half with its row -- never to V.
    /// </summary>
    private static void ApplyRope2DHalf(float* half, int quarterDim, int position, float ropeTheta)
    {
        var nDims = quarterDim * 2;
        for (var j = 0; j < quarterDim; j++)
        {
            var thetaScale = MathF.Pow(ropeTheta, -2f * j / nDims);
            var angle = position * thetaScale;
            var cos = MathF.Cos(angle);
            var sin = MathF.Sin(angle);
            var x0 = half[j];
            var x1 = half[j + quarterDim];
            half[j] = x0 * cos - x1 * sin;
            half[j + quarterDim] = x0 * sin + x1 * cos;
        }
    }
}
