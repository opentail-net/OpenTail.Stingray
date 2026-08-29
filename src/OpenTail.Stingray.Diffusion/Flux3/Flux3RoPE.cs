
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
        float[] invFreqT = InterleavedRoPE.ComputeInvFreqs(dimT, theta);
        float[] invFreqY = InterleavedRoPE.ComputeInvFreqs(dimY, theta);
        float[] invFreqX = InterleavedRoPE.ComputeInvFreqs(dimX, theta);

        for (int i = 0; i < nTokens; i++)
        {
            int t = positions[i * 3 + 0];
            int y = positions[i * 3 + 1];
            int x = positions[i * 3 + 2];

            int outOffset = i * headDim;

            // 1. Temporal Axis
            InterleavedRoPE.FillAxisFreqs(cos.AsSpan(outOffset, dimT), sin.AsSpan(outOffset, dimT), t, invFreqT);

            // 2. Vertical Axis (Y)
            InterleavedRoPE.FillAxisFreqs(cos.AsSpan(outOffset + dimT, dimY), sin.AsSpan(outOffset + dimT, dimY), y, invFreqY);

            // 3. Horizontal Axis (X)
            InterleavedRoPE.FillAxisFreqs(cos.AsSpan(outOffset + dimT + dimY, dimX), sin.AsSpan(outOffset + dimT + dimY, dimX), x, invFreqX);
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

        float[] invFreqT = InterleavedRoPE.ComputeInvFreqs(dimT, theta);
        float[] invFreqF = InterleavedRoPE.ComputeInvFreqs(dimF, theta);

        for (int i = 0; i < nTokens; i++)
        {
            int t = positions[i * 2 + 0];
            int f = positions[i * 2 + 1];

            int outOffset = i * headDim;

            InterleavedRoPE.FillAxisFreqs(cos.AsSpan(outOffset, dimT), sin.AsSpan(outOffset, dimT), t, invFreqT);
            InterleavedRoPE.FillAxisFreqs(cos.AsSpan(outOffset + dimT, dimF), sin.AsSpan(outOffset + dimT, dimF), f, invFreqF);
        }

        return (cos, sin);
    }

    /// <summary>
    /// Applies rotary embeddings in-place to Q or K tensor [nTokens, numHeads, headDim].
    /// </summary>
    public static void ApplyRoPE(Span<float> tensor, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin, int nTokens, int numHeads, int headDim)
        => InterleavedRoPE.ApplyRoPE(tensor, cos, sin, nTokens, numHeads, headDim);
}
