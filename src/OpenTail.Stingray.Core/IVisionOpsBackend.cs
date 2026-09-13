namespace OpenTail.Stingray.Core;

/// <summary>
/// Extended compute-backend interface for native GPU vision encoder operations (ViT patch downsampling,
/// 2D multimodal position embeddings, layer norms, and specialized activation functions).
/// </summary>
public interface IVisionOpsBackend : IComputeBackend
{
    /// <summary>
    /// PixelShuffle 2x2 spatial downsampler for vision token grids:
    /// Merges 2x2 spatial patch tokens [gridY, gridX, inDim] -> [gridY/2, gridX/2, 4*inDim].
    /// </summary>
    Tensor VisionPixelShuffle2x2(Tensor input, int gridY, int gridX, int inDim);

    /// <summary>
    /// Multimodal 2D Rotary Position Embedding (M-RoPE) for vision encoders.
    /// Rotates quarter sub-bands of Q and K with X and Y patch coordinates.
    /// </summary>
    void VisionMRoPE(Tensor q, Tensor k, int patchesX, int patchesY, int qHeads, int kvHeads, int headDim, float theta = 10000.0f);

    /// <summary>
    /// Continuous 2D Rotary Position Embedding (Pixtral / Gemma style).
    /// </summary>
    void VisionContinuous2DRoPE(Tensor q, Tensor k, int patchesX, int patchesY, int heads, int headDim, float theta = 10000.0f);

    /// <summary>
    /// Vision LayerNorm with optional bias.
    /// </summary>
    void VisionLayerNorm(Tensor output, Tensor input, Tensor weight, Tensor? bias, float eps = 1e-5f);

    /// <summary>
    /// Vision GELU activation in-place (Tanh approximation).
    /// </summary>
    void VisionGeluInPlace(Tensor x);

    /// <summary>
    /// Vision QuickGELU activation in-place (x * sigmoid(1.702 * x)).
    /// </summary>
    void VisionQuickGeluInPlace(Tensor x);

    /// <summary>
    /// Vision Squared ReLU activation in-place (max(0, x)^2).
    /// </summary>
    void VisionSquaredReluInPlace(Tensor x);

    /// <summary>
    /// Fused AdaLN-Zero Modulation:
    /// y = Norm(x, eps) * (1 + scale) + shift
    /// where Norm is RMSNorm (if isRmsNorm=true) or LayerNorm.
    /// </summary>
    void AdaLNModulate(Tensor output, Tensor input, Tensor shift, Tensor scale, int nTokens, int dim, bool isRmsNorm = true, float eps = 1e-5f);

    /// <summary>
    /// Fused AdaLN-Zero Modulation taking a combined modulation tensor with slice offsets.
    /// </summary>
    void AdaLNModulate(Tensor output, Tensor input, Tensor mod, int nTokens, int dim, int shiftOffset = 0, int scaleOffset = -1, bool isRmsNorm = true, float eps = 1e-5f)
        => throw new NotSupportedException();

    /// <summary>
    /// Modulated residual addition:
    /// x = x + proj * gate
    /// </summary>
    void ScaleGateAdd(Tensor x, Tensor proj, Tensor gate, int nTokens, int dim);

    /// <summary>
    /// Modulated residual addition taking a combined modulation tensor with gate offset:
    /// x = x + proj * mod[gateOffset ..]
    /// </summary>
    void ScaleGateAdd(Tensor x, Tensor proj, Tensor mod, int nTokens, int dim, int gateOffset)
        => throw new NotSupportedException();

    /// <summary>
    /// Per-head QK Normalization in VRAM:
    /// q_h = RMSNorm(q_h) * qScale, k_h = RMSNorm(k_h) * kScale
    /// </summary>
    void QKNorm(Tensor q, Tensor k, Tensor qScale, Tensor kScale, int nTokens, int numHeads, int headDim, float eps = 1e-5f);

    /// <summary>
    /// Per-head QK Normalization in VRAM with startToken offset.
    /// </summary>
    void QKNorm(Tensor q, Tensor k, Tensor qScale, Tensor kScale, int nTokens, int numHeads, int headDim, float eps, int startToken)
        => throw new NotSupportedException();

    /// <summary>
    /// 3D Spatio-Temporal Rotary Position Embedding for video DiTs (e.g. Wan2.1, HunyuanVideo).
    /// Rotates Q and K across (temporal, height, width) sub-bands.
    /// </summary>
    void RoPE3D(Tensor q, Tensor k, int numTokens, int numHeads, int headDim, int tDim, int hDim, int wDim, float theta = 10000.0f);

    /// <summary>
    /// FLUX 2D Rotary Position Embedding (GPT-NeoX interleaved pair rotation).
    /// Rotates Q and K in-place using precomputed cos/sin frequency tables.
    /// </summary>
    void Flux2DRoPE(Tensor q, Tensor k, Tensor cos, Tensor sin, int startToken, int tokenCount, int numHeads, int headDim)
        => throw new NotSupportedException();

    /// <summary>
    /// Fused QKV split + per-head RMSNorm + 3D RoPE for Wan 2.1 self-attention.
    /// Unpacks Q, K, V from fused Qkv [numTokens, 3*dim], applies RMSNorm on Q and K,
    /// rotates Q and K with 3D RoPE tables, and writes out Q, K, V ready for attention.
    /// </summary>
    void WanQkvSplitNormRoPE(Tensor qkv, Tensor q, Tensor k, Tensor v, Tensor cos, Tensor sin,
                             Tensor? normQ, Tensor? normK, int numTokens, int numHeads, int headDim, float eps = 1e-6f)
        => throw new NotSupportedException();

    /// <summary>
    /// Unpack fused QKV [nTokens, 3*dim] into separate Q, K, V buffers [nSeq, dim] at dstTokenOffset.
    /// </summary>
    void FluxUnpackQkv(Tensor qkv, Tensor q, Tensor k, Tensor v, int nTokens, int dim, int dstTokenOffset)
        => throw new NotSupportedException();

    /// <summary>
    /// Unpack fused Linear1 [nSeq, 7*dim] into Q, K, V [nSeq, dim] and MLP [nSeq, 4*dim].
    /// </summary>
    void FluxUnpackSingleLin1(Tensor lin1, Tensor q, Tensor k, Tensor v, Tensor mlp, int nSeq, int dim)
        => throw new NotSupportedException();

    /// <summary>
    /// Interleaves Attention output [nSeq, dim] and MLP output [nSeq, 4*dim] per token into Combined [nSeq, 5*dim].
    /// </summary>
    void FluxConcatAttnMlp(Tensor attnOut, Tensor mlp, Tensor combined, int nSeq, int dim)
        => throw new NotSupportedException();

    /// <summary>
    /// Concatenates text [nTxt, dim] and image [nImg, dim] token sequences into x [nSeq, dim].
    /// </summary>
    void FluxConcatTxtImg(Tensor txt, Tensor img, Tensor x, int nTxt, int nImg, int dim)
        => throw new NotSupportedException();

    /// <summary>
    /// Slices out the image portion [nImg, dim] from combined sequence x [nSeq, dim] (skipping nTxt tokens).
    /// </summary>
    void FluxSliceImg(Tensor x, Tensor img, int nTxt, int nImg, int dim)
        => throw new NotSupportedException();

    /// <summary>
    /// In-place Euler flow matching step on GPU: x[i] += signDt * v[i].
    /// </summary>
    void FluxEulerStep(Tensor x, Tensor v, float signDt, int count)
        => throw new NotSupportedException();

    /// <summary>
    /// T5 multi-head self-attention with relative position bias:
    /// scores[h, i, j] = dot(q, k) * (1/sqrt(headDim)) + relBias[h, i, j], output = softmax(scores) * v.
    /// Q, K, V are [seq, numHeads * headDim] (row-major per token).
    /// RelBias is [numHeads, qSeq, kvSeq].
    /// </summary>
    void T5MultiHeadAttentionRelBias(Tensor output, Tensor q, Tensor k, Tensor v, Tensor relBias, int qSeq, int kvSeq, int numHeads, int headDim)
        => throw new NotSupportedException();

    /// <summary>
    /// Batched RMSNorm over numTokens rows of dimension rowDim:
    /// output[t, :] = RMSNorm(x[t, :], weight, eps).
    /// </summary>
    void RmsNormBatched(Tensor output, Tensor x, Tensor weight, int rowDim, int numTokens, float eps = 1e-5f)
        => throw new NotSupportedException();
}

