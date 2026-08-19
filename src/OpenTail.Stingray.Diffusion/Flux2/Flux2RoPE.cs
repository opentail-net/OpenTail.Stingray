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

        float[] invFreqImg = ComputeInvFreqs(dimImg, theta);
        float[] invFreqY = ComputeInvFreqs(dimY, theta);
        float[] invFreqX = ComputeInvFreqs(dimX, theta);

        for (int i = 0; i < nTokens; i++)
        {
            int imgIdx = positions[i * 3 + 0];
            int y = positions[i * 3 + 1];
            int x = positions[i * 3 + 2];

            int outOffset = i * headDim;

            // 1. Image Index Axis (Multi-Image ID)
            FillAxisFreqs(cos.AsSpan(outOffset, dimImg), sin.AsSpan(outOffset, dimImg), imgIdx, invFreqImg);

            // 2. Vertical Axis (Y)
            FillAxisFreqs(cos.AsSpan(outOffset + dimImg, dimY), sin.AsSpan(outOffset + dimImg, dimY), y, invFreqY);

            // 3. Horizontal Axis (X)
            FillAxisFreqs(cos.AsSpan(outOffset + dimImg + dimY, dimX), sin.AsSpan(outOffset + dimImg + dimY, dimX), x, invFreqX);
        }

        return (cos, sin);
    }

    /// <summary>
    /// Applies rotary embeddings in-place to Q or K tensor [nTokens, numHeads, headDim].
    /// </summary>
    public static void ApplyRoPE(Span<float> tensor, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin, int nTokens, int numHeads, int headDim)
    {
        int halfHead = headDim / 2;

        for (int i = 0; i < nTokens; i++)
        {
            var cosToken = cos.Slice(i * headDim, headDim);
            var sinToken = sin.Slice(i * headDim, headDim);

            for (int h = 0; h < numHeads; h++)
            {
                int offset = (i * numHeads + h) * headDim;
                var headSpan = tensor.Slice(offset, headDim);

                for (int d = 0; d < halfHead; d++)
                {
                    float x0 = headSpan[d];
                    float x1 = headSpan[d + halfHead];
                    float c = cosToken[d * 2];
                    float s = sinToken[d * 2];

                    headSpan[d] = x0 * c - x1 * s;
                    headSpan[d + halfHead] = x0 * s + x1 * c;
                }
            }
        }
    }

    private static float[] ComputeInvFreqs(int dim, float theta)
    {
        int half = dim / 2;
        var invFreq = new float[half];
        for (int i = 0; i < half; i++)
        {
            invFreq[i] = (float)(1.0 / Math.Pow(theta, (2.0 * i) / dim));
        }
        return invFreq;
    }

    private static void FillAxisFreqs(Span<float> cosOut, Span<float> sinOut, int pos, ReadOnlySpan<float> invFreq)
    {
        int half = invFreq.Length;
        for (int i = 0; i < half; i++)
        {
            float angle = pos * invFreq[i];
            float c = MathF.Cos(angle);
            float s = MathF.Sin(angle);

            cosOut[i * 2 + 0] = c;
            cosOut[i * 2 + 1] = c;
            sinOut[i * 2 + 0] = s;
            sinOut[i * 2 + 1] = s;
        }
    }
}
