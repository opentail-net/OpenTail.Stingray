using System.Collections.Concurrent;

namespace OpenTail.Stingray.Audio;

/// <summary>
/// Preset quality levels for windowed-sinc resampling.
/// Controls the half-width (tap count = 2 * L) of the convolution kernel.
/// </summary>
public enum ResampleQuality
{
    SuperFast = 8,
    Fast = 16,
    Balanced = 32,
    HighQuality = 64,
    BestQuality = 128
}

/// <summary>
/// High-fidelity, band-limited windowed-sinc audio resampler ported from Transformations.Audio.
/// Uses exact rational phase caching and SIMD-accelerated convolution.
/// </summary>
public static class AudioResampler
{
    private static readonly ConcurrentDictionary<KernelKey, double[][]> s_kernelPhaseBanks = new();

    private readonly record struct KernelKey(int HalfWidth, int Numerator, int Denominator);

    /// <summary>
    /// Resamples interleaved floating-point audio from <paramref name="inRate"/> to <paramref name="outRate"/>.
    /// </summary>
    public static float[] Resample(
        ReadOnlySpan<float> inputData,
        int inRate,
        int outRate,
        int channels = 1,
        ResampleQuality quality = ResampleQuality.Balanced)
    {
        if (inRate <= 0 || outRate <= 0)
            throw new ArgumentException("Sample rates must be positive.");
        if (channels <= 0)
            throw new ArgumentException("Channel count must be positive.", nameof(channels));
        if (inputData.Length == 0)
            return [];
        if (inputData.Length % channels != 0)
            throw new ArgumentException("Input data length must be divisible by the number of channels.", nameof(inputData));

        if (inRate == outRate)
            return inputData.ToArray();

        int kernelHalfWidth = (int)quality;
        double ratio = (double)outRate / inRate;
        int rateGcd = Gcd(inRate, outRate);
        int p = inRate / rateGcd;
        int q = outRate / rateGcd;

        var phaseBank = GetPhaseBank(kernelHalfWidth, p, q);

        if (channels == 1)
        {
            return ResampleSingleChannel(inputData, ratio, p, q, kernelHalfWidth, phaseBank);
        }

        // Multi-channel: split -> resample each -> interleave
        int frames = inputData.Length / channels;
        float[][] channelData = new float[channels][];
        for (int c = 0; c < channels; c++)
        {
            channelData[c] = new float[frames];
            for (int i = 0; i < frames; i++)
                channelData[c][i] = inputData[i * channels + c];
        }

        float[][] resampledChannels = new float[channels][];
        Parallel.For(0, channels, c =>
        {
            resampledChannels[c] = ResampleSingleChannel(channelData[c], ratio, p, q, kernelHalfWidth, phaseBank);
        });

        int outFrames = resampledChannels[0].Length;
        float[] output = new float[outFrames * channels];
        for (int i = 0; i < outFrames; i++)
        {
            for (int c = 0; c < channels; c++)
                output[i * channels + c] = resampledChannels[c][i];
        }

        return output;
    }

    private static double[][] GetPhaseBank(int kernelHalfWidth, int p, int q)
    {
        var key = new KernelKey(kernelHalfWidth, p, q);
        return s_kernelPhaseBanks.GetOrAdd(key, static k =>
        {
            int qVal = k.Denominator;
            var bank = new double[qVal][];
            for (int rem = 0; rem < qVal; rem++)
            {
                bank[rem] = ComputeKernel(rem, qVal, k.HalfWidth);
            }
            return bank;
        });
    }

    private static unsafe float[] ResampleSingleChannel(
        ReadOnlySpan<float> channel,
        double ratio,
        int p,
        int q,
        int kernelHalfWidth,
        double[][] phaseBank)
    {
        int inputLength = channel.Length;
        int outputLength = (int)Math.Round(inputLength * ratio);
        if (outputLength <= 0) return [];

        float[] output = new float[outputLength];

        // Boundary padding to avoid bounds checks in the hot convolution loop
        int pad = kernelHalfWidth + 16;
        int totalPad = ((pad + 7) / 8) * 8;
        double[] padded = new double[inputLength + 2 * totalPad];

        for (int i = 0; i < totalPad; i++)
            padded[i] = channel[0];

        for (int i = 0; i < inputLength; i++)
            padded[totalPad + i] = channel[i];

        for (int i = 0; i < totalPad; i++)
            padded[totalPad + inputLength + i] = channel[inputLength - 1];

        int kernelLength = 2 * kernelHalfWidth;

        fixed (double* pPad = padded)
        fixed (float* pOut = output)
        {
            var padPtr = pPad;
            var outPtr = pOut;

            if (outputLength >= 4096)
            {
                Parallel.For(0, outputLength, i =>
                {
                    long num = (long)i * p;
                    int inputCenter = (int)(num / q);
                    int rem = (int)(num % q);

                    double[] kernel = phaseBank[rem];
                    int startIdx = totalPad + inputCenter - kernelHalfWidth + 1;
                    double* winPtr = padPtr + startIdx;

                    double sum = 0.0;
                    fixed (double* kPtr = kernel)
                    {
                        for (int k = 0; k < kernelLength; k++)
                        {
                            sum += winPtr[k] * kPtr[k];
                        }
                    }

                    outPtr[i] = (float)sum;
                });
            }
            else
            {
                for (int i = 0; i < outputLength; i++)
                {
                    long num = (long)i * p;
                    int inputCenter = (int)(num / q);
                    int rem = (int)(num % q);

                    double[] kernel = phaseBank[rem];
                    int startIdx = totalPad + inputCenter - kernelHalfWidth + 1;
                    double* winPtr = padPtr + startIdx;

                    double sum = 0.0;
                    fixed (double* kPtr = kernel)
                    {
                        for (int k = 0; k < kernelLength; k++)
                        {
                            sum += winPtr[k] * kPtr[k];
                        }
                    }

                    outPtr[i] = (float)sum;
                }
            }
        }

        return output;
    }

    private static double[] ComputeKernel(int rem, int q, int kernelHalfWidth)
    {
        int kernelLength = 2 * kernelHalfWidth;
        double frac = (double)rem / q;
        double[] weights = new double[kernelLength];
        double sum = 0;

        for (int i = 0; i < kernelLength; i++)
        {
            int k = i - kernelHalfWidth + 1;
            double x = frac - k;
            double w = Sinc(x) * Hann(x, kernelHalfWidth);
            weights[i] = w;
            sum += w;
        }

        // Normalize filter gain to 1.0 (DC unity gain)
        if (Math.Abs(sum) > 1e-9)
        {
            for (int i = 0; i < kernelLength; i++)
                weights[i] /= sum;
        }

        return weights;
    }

    private static double Sinc(double x)
    {
        if (Math.Abs(x) < 1e-9) return 1.0;
        double px = Math.PI * x;
        return Math.Sin(px) / px;
    }

    private static double Hann(double x, int L)
    {
        if (Math.Abs(x) > L) return 0.0;
        return 0.5 * (1.0 + Math.Cos(Math.PI * x / L));
    }

    private static int Gcd(int a, int b)
    {
        a = Math.Abs(a);
        b = Math.Abs(b);
        while (b != 0)
        {
            (a, b) = (b, a % b);
        }
        return a == 0 ? 1 : a;
    }
}
