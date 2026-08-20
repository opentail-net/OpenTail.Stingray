using System;
using System.Collections.Concurrent;

namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// Shared low-level spectral primitives (window functions, fast Twiddle-cached DFT) used identically
/// across the ASR/TTS mel-spectrogram extractors. Each extractor still owns its own
/// framing, filterbank shape, log-compression and normalization, since those differ
/// per model.
/// </summary>
internal static unsafe class SpectralKernels
{
    private static readonly ConcurrentDictionary<int, (float[] Cos, float[] Sin)> TwiddleCache = new();

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
    /// Gets or creates precomputed cos/sin twiddle tables for a given FFT window length N.
    /// </summary>
    public static (float[] Cos, float[] Sin) GetTwiddleTables(int n)
    {
        return TwiddleCache.GetOrAdd(n, static nLen =>
        {
            int half = nLen / 2 + 1;
            var cosTab = new float[half * nLen];
            var sinTab = new float[half * nLen];

            for (int k = 0; k < half; k++)
            {
                float angleStep = -2.0f * MathF.PI * k / nLen;
                int row = k * nLen;
                for (int t = 0; t < nLen; t++)
                {
                    float angle = angleStep * t;
                    cosTab[row + t] = MathF.Cos(angle);
                    sinTab[row + t] = MathF.Sin(angle);
                }
            }

            return (cosTab, sinTab);
        });
    }

    /// <summary>
    /// Fast twiddle-accelerated real-input DFT power spectrum |X[k]|^2 (no 1/N scaling) for k in [0, n/2].
    /// </summary>
    public static void ComputePowerSpectrum(ReadOnlySpan<float> windowed, Span<float> powerSpectrum)
    {
        int n = windowed.Length;
        int half = n / 2 + 1;

        var (cosTable, sinTable) = GetTwiddleTables(n);

        fixed (float* pWin = windowed)
        fixed (float* pOut = powerSpectrum)
        fixed (float* pCos = cosTable)
        fixed (float* pSin = sinTable)
        {
            for (int k = 0; k < half; k++)
            {
                float* cosRow = pCos + (long)k * n;
                float* sinRow = pSin + (long)k * n;

                float real = 0f;
                float imag = 0f;

                for (int t = 0; t < n; t++)
                {
                    float s = pWin[t];
                    real += s * cosRow[t];
                    imag += s * sinRow[t];
                }

                pOut[k] = real * real + imag * imag;
            }
        }
    }
}
