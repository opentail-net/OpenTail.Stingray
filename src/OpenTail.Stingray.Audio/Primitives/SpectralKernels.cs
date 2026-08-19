namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// Shared low-level spectral primitives (window functions, naive DFT) used identically
/// across the ASR/TTS mel-spectrogram extractors. Each extractor still owns its own
/// framing, filterbank shape, log-compression and normalization, since those differ
/// per model.
/// </summary>
internal static class SpectralKernels
{
    /// <summary>Periodic Hann window: 0.5 * (1 - cos(2*pi*i/N)).</summary>
    public static float[] CreateHannWindow(int size)
    {
        var win = new float[size];
        for (int i = 0; i < size; i++)
        {
            win[i] = 0.5f * (1.0f - MathF.Cos(2.0f * MathF.PI * i / size));
        }
        return win;
    }

    /// <summary>Symmetric Hann window: 0.5 * (1 - cos(2*pi*i/(N-1))).</summary>
    public static float[] CreateSymmetricHannWindow(int size)
    {
        var win = new float[size];
        for (int i = 0; i < size; i++)
        {
            win[i] = 0.5f * (1.0f - MathF.Cos(2.0f * MathF.PI * i / (size - 1)));
        }
        return win;
    }

    /// <summary>
    /// Naive O(bins*n) real-input DFT power spectrum |X[k]|^2 (no 1/N scaling) for k in [0, n/2].
    /// </summary>
    public static void ComputePowerSpectrum(ReadOnlySpan<float> windowed, Span<float> powerSpectrum)
    {
        int n = windowed.Length;
        int half = n / 2 + 1;

        for (int k = 0; k < half; k++)
        {
            float real = 0f;
            float imag = 0f;
            float angleStep = -2.0f * MathF.PI * k / n;

            for (int t = 0; t < n; t++)
            {
                float angle = angleStep * t;
                float s = windowed[t];
                real += s * MathF.Cos(angle);
                imag += s * MathF.Sin(angle);
            }

            powerSpectrum[k] = real * real + imag * imag;
        }
    }
}
