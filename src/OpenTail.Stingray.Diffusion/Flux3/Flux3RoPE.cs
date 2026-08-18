using System.Numerics.Tensors;

namespace OpenTail.Stingray.Diffusion.Flux3;

/// <summary>
/// 3D/4D Spatiotemporal and Audio Rotary Position Embedding (RoPE) generator for FLUX 3.
/// Generates cross-modal rotary cosine/sine matrices aligning temporal frames, spatial grids, and audio spectrograms.
/// </summary>
public static class Flux3RoPE
{
    /// <summary>
    /// Computes cosine and sine frequency matrices for 3D video tokens (t, y, x).
    /// Shape of returned matrices: [nTokens * headDim].
    /// </summary>
    /// <param name="positions">Array of (t, y, x) integer triplets, length = nTokens * 3.</param>
    /// <param name="nTokens">Number of video patch tokens.</param>
    /// <param name="axesDim">Axial split [dimT, dimY, dimX], e.g. [32, 48, 48] (sum must equal headDim).</param>
    /// <param name="theta">Base frequency theta (default: 10000.0).</param>
    public static (float[] cos, float[] sin) BuildVideoFreqs(
        ReadOnlySpan<int> positions,
        int nTokens,
        int[] axesDim,
        float theta = 10000.0f)
    {
        if (axesDim.Length != 3)
            throw new ArgumentException("Video axesDim must contain exactly 3 elements: [dimT, dimY, dimX].", nameof(axesDim));

        int dimT = axesDim[0];
        int dimY = axesDim[1];
        int dimX = axesDim[2];
        int headDim = dimT + dimY + dimX;

        var cos = new float[nTokens * headDim];
        var sin = new float[nTokens * headDim];

        // Precompute frequency inv-scales for each axis
        float[] invFreqT = ComputeInvFreqs(dimT, theta);
        float[] invFreqY = ComputeInvFreqs(dimY, theta);
        float[] invFreqX = ComputeInvFreqs(dimX, theta);

        for (int i = 0; i < nTokens; i++)
        {
            int t = positions[i * 3 + 0];
            int y = positions[i * 3 + 1];
            int x = positions[i * 3 + 2];

            int outOffset = i * headDim;

            // 1. Temporal Axis
            FillAxisFreqs(cos.AsSpan(outOffset, dimT), sin.AsSpan(outOffset, dimT), t, invFreqT);

            // 2. Vertical Axis (Y)
            FillAxisFreqs(cos.AsSpan(outOffset + dimT, dimY), sin.AsSpan(outOffset + dimT, dimY), y, invFreqY);

            // 3. Horizontal Axis (X)
            FillAxisFreqs(cos.AsSpan(outOffset + dimT + dimY, dimX), sin.AsSpan(outOffset + dimT + dimY, dimX), x, invFreqX);
        }

        return (cos, sin);
    }

    /// <summary>
    /// Computes cosine and sine frequency matrices for 2D audio spectrogram tokens (t, freq).
    /// Shape: [nTokens * headDim].
    /// </summary>
    /// <param name="positions">Array of (t, freq) integer pairs, length = nTokens * 2.</param>
    /// <param name="nTokens">Number of acoustic patch tokens.</param>
    /// <param name="axesDim">Axial split [dimT, dimFreq], e.g. [64, 64].</param>
    /// <param name="theta">Base frequency theta.</param>
    public static (float[] cos, float[] sin) BuildAudioFreqs(
        ReadOnlySpan<int> positions,
        int nTokens,
        int[] axesDim,
        float theta = 10000.0f)
    {
        if (axesDim.Length != 2)
            throw new ArgumentException("Audio axesDim must contain exactly 2 elements: [dimT, dimFreq].", nameof(axesDim));

        int dimT = axesDim[0];
        int dimF = axesDim[1];
        int headDim = dimT + dimF;

        var cos = new float[nTokens * headDim];
        var sin = new float[nTokens * headDim];

        float[] invFreqT = ComputeInvFreqs(dimT, theta);
        float[] invFreqF = ComputeInvFreqs(dimF, theta);

        for (int i = 0; i < nTokens; i++)
        {
            int t = positions[i * 2 + 0];
            int f = positions[i * 2 + 1];

            int outOffset = i * headDim;

            FillAxisFreqs(cos.AsSpan(outOffset, dimT), sin.AsSpan(outOffset, dimT), t, invFreqT);
            FillAxisFreqs(cos.AsSpan(outOffset + dimT, dimF), sin.AsSpan(outOffset + dimT, dimF), f, invFreqF);
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
