
namespace OpenTail.Stingray.Diffusion.Flux2;

/// <summary>
/// 3D Contextual Rotary Position Embedding (RoPE) generator for FLUX.2 (Klein &amp; Kontext).
/// Positions tokens in 3D space (image_index, y, x) to allow multi-reference image conditioning without coordinate collision.
/// Reference: diffusers:src/diffusers/models/transformers/transformer_flux2.py
/// </summary>
public static class Flux2RoPE
{
    /// <summary>
    /// Computes cosine and sine frequency matrices for 3D contextual tokens (image_index, y, x).
    /// Shape: [nTokens * headDim].
    /// </summary>
    /// <param name="positions">Array of (img_idx, y, x) integer triplets, length = nTokens * 3.</param>
    /// <param name="nTokens">Number of patch/text tokens.</param>
    /// <param name="axesDim">Axial split [dimImageIdx, dimY, dimX], e.g. [16, 56, 56] (sum must equal headDim).</param>
    /// <param name="theta">Base frequency theta (default: 10000.0).</param>
    public static (float[] cos, float[] sin) BuildContextFreqs(
        ReadOnlySpan<int> positions,
        int nTokens,
        int[] axesDim,
        float theta = 10000.0f)
    {
        if (axesDim.Length != 3)
            throw new ArgumentException("FLUX.2 axesDim must contain 3 elements: [dimImageIdx, dimY, dimX].", nameof(axesDim));

        int dimImg = axesDim[0];
        int dimY = axesDim[1];
        int dimX = axesDim[2];
        int headDim = dimImg + dimY + dimX;

        var cos = new float[nTokens * headDim];
        var sin = new float[nTokens * headDim];

        float[] invFreqImg = InterleavedRoPE.ComputeInvFreqs(dimImg, theta);
        float[] invFreqY = InterleavedRoPE.ComputeInvFreqs(dimY, theta);
        float[] invFreqX = InterleavedRoPE.ComputeInvFreqs(dimX, theta);

        for (int i = 0; i < nTokens; i++)
        {
            int imgIdx = positions[i * 3 + 0];
            int y = positions[i * 3 + 1];
            int x = positions[i * 3 + 2];

            int outOffset = i * headDim;

            // 1. Image Index Axis (Multi-Image ID)
            InterleavedRoPE.FillAxisFreqs(cos.AsSpan(outOffset, dimImg), sin.AsSpan(outOffset, dimImg), imgIdx, invFreqImg);

            // 2. Vertical Axis (Y)
            InterleavedRoPE.FillAxisFreqs(cos.AsSpan(outOffset + dimImg, dimY), sin.AsSpan(outOffset + dimImg, dimY), y, invFreqY);

            // 3. Horizontal Axis (X)
            InterleavedRoPE.FillAxisFreqs(cos.AsSpan(outOffset + dimImg + dimY, dimX), sin.AsSpan(outOffset + dimImg + dimY, dimX), x, invFreqX);
        }

        return (cos, sin);
    }

    /// <summary>
    /// Applies rotary embeddings in-place to Q or K tensor [nTokens, numHeads, headDim].
    /// </summary>
    public static void ApplyRoPE(Span<float> tensor, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin, int nTokens, int numHeads, int headDim)
        => InterleavedRoPE.ApplyRoPE(tensor, cos, sin, nTokens, numHeads, headDim);
}
