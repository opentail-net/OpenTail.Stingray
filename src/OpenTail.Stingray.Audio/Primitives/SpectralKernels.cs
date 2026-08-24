using System;
using System.Collections.Concurrent;
using OpenTail.Stingray.Cpu;

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

                // Was a scalar per-element accumulation; cosRow/sinRow/pWin are all contiguous,
                // so this is a direct SimdKernels.DotF32 (AVX2/FMA) substitution -- shared by
                // Whisper/Parakeet/F5-TTS/QwenASR's mel extractors, all naive O(n) per bin
                // before this change.
                float real = SimdKernels.DotF32(pWin, cosRow, n);
                float imag = SimdKernels.DotF32(pWin, sinRow, n);

                pOut[k] = real * real + imag * imag;
            }
        }
    }

    /// <summary>
    /// Inverse real FFT: reconstructs n real time-domain samples from n/2+1 complex frequency
    /// bins (torch.fft.irfft convention, norm="backward" i.e. 1/n scaling on the inverse).
    /// y[t] = (1/n) * [Re(X[0]) + (-1)^t*Re(X[n/2]) + 2*sum_{k=1}^{n/2-1}(Re(X[k])*cos(2*pi*k*t/n) + Im(X[k])*sin(2*pi*k*t/n))]
    /// (uses the conjugate-symmetric structure of a real signal's spectrum -- X[n-k] = conj(X[k])
    /// -- so only the n/2+1 non-redundant bins are needed). Reuses <see cref="GetTwiddleTables"/>'s
    /// forward-DFT cos/sin tables: cosTab[k*n+t] = cos(2*pi*k*t/n) (cosine is even), sinTab[k*n+t]
    /// = -sin(2*pi*k*t/n) (forward DFT uses the negative-angle convention), so the inverse formula
    /// above needs `-Im(X[k])*sinTab[...]` to recover `+Im(X[k])*sin(2*pi*k*t/n)`.
    /// </summary>
    public static void InverseRealFft(ReadOnlySpan<float> real, ReadOnlySpan<float> imag, Span<float> output)
    {
        int n = output.Length;
        int half = n / 2 + 1;
        var (cosTable, sinTable) = GetTwiddleTables(n);
        float invN = 1f / n;

        for (int t = 0; t < n; t++)
        {
            float sum = real[0] + ((t & 1) == 0 ? real[half - 1] : -real[half - 1]);
            for (int k = 1; k < half - 1; k++)
            {
                int idx = k * n + t;
                sum += 2f * (real[k] * cosTable[idx] - imag[k] * sinTable[idx]);
            }
            output[t] = sum * invN;
        }
    }
}
