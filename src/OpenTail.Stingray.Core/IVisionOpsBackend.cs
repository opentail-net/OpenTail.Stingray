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
}
