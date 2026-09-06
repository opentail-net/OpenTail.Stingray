
namespace OpenTail.Stingray.Audio.Rvc;

/// <summary>
/// Real RMVPE log-mel frontend, transcribed directly from `compute_rmvpe_log_mel` in
/// `examples/audio.cpp/src/framework/modules/pitch_extractors/rmvpe_pitch_extractor.cpp` (not
/// guessed) -- see docs/audio-review-progress.md's RMVPE section for the full derivation.
/// Deliberately DIFFERENT from every other mel extractor already in this codebase: real HTK mel
/// scale (not Slaney/librosa-default), fmin=30Hz (not 0), fmax=8000Hz (Nyquist for 16kHz), and
/// filters applied to the STFT MAGNITUDE spectrum (not power).
/// </summary>
public static class RvcRmvpeMelExtractor
{
    public const int SampleRate = 16000;
    public const int NFft = 1024;
    public const int HopLength = 160;
    public const int MelBins = 128;
    private const float FMin = 30f;
    private const float FMax = 8000f;

    private static readonly float[][] MelFilter = BuildMelFilterbank();

    /// <summary>Returns frame-major [frames, 128] log-mel, time-padded to a multiple of 32
    /// (matching the U-Net's 5 avg-pool-2x2 levels) with the real pad amount also reported so a
    /// caller can trim the network's output back to the real (unpadded) frame count afterward.</summary>
    public static (float[][] Mel, int RealFrames) ExtractPadded(ReadOnlySpan<float> waveform16k)
    {
        var magnitude = Stft.Magnitude(waveform16k, NFft, HopLength, reflectPad: true);
        int freqBins = NFft / 2 + 1;
        int frames = magnitude.Length;

        var mel = new float[frames][];
        for (int f = 0; f < frames; f++)
        {
            var row = new float[MelBins];
            for (int m = 0; m < MelBins; m++)
            {
                double sum = 0;
                var filter = MelFilter[m];
                for (int freq = 0; freq < freqBins; freq++)
                    sum += filter[freq] * magnitude[f][freq];
                row[m] = MathF.Log(MathF.Max((float)sum, 1e-5f));
            }
            mel[f] = row;
        }

        int paddedFrames = 32 * ((frames - 1) / 32 + 1);
        if (paddedFrames == frames) return (mel, frames);

        var padded = new float[paddedFrames][];
        Array.Copy(mel, padded, frames);
        for (int f = frames; f < paddedFrames; f++) padded[f] = new float[MelBins];
        return (padded, frames);
    }

    /// <summary>Real HTK mel scale (`2595*log10(1+hz/700)`) with Slaney-style per-filter energy
    /// normalization (`enorm = 2/(right-left)`), matching the reference's real filterbank
    /// construction exactly (not the Slaney mel-scale formula some other extractors in this
    /// codebase use -- HTK and Slaney differ below ~1kHz).</summary>
    private static float[][] BuildMelFilterbank()
    {
        int freqBins = NFft / 2 + 1;
        static double HzToMel(double hz) => 2595.0 * Math.Log10(1.0 + hz / 700.0);
        static double MelToHz(double mel) => 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);

        double melMin = HzToMel(FMin);
        double melMax = HzToMel(FMax);
        var melPoints = new double[MelBins + 2];
        for (int i = 0; i < MelBins + 2; i++)
            melPoints[i] = melMin + (melMax - melMin) * i / (MelBins + 1);
        var hzPoints = new double[MelBins + 2];
        for (int i = 0; i < MelBins + 2; i++)
            hzPoints[i] = MelToHz(melPoints[i]);

        var filters = new float[MelBins][];
        for (int mel = 0; mel < MelBins; mel++)
        {
            var filter = new float[freqBins];
            double left = hzPoints[mel], center = hzPoints[mel + 1], right = hzPoints[mel + 2];
            double lowerWidth = Math.Max(center - left, 1e-12);
            double upperWidth = Math.Max(right - center, 1e-12);
            double enorm = 2.0 / Math.Max(right - left, 1e-12);
            for (int freq = 0; freq < freqBins; freq++)
            {
                double hz = SampleRate * 0.5 * freq / (freqBins - 1);
                double lower = (hz - left) / lowerWidth;
                double upper = (right - hz) / upperWidth;
                filter[freq] = (float)(Math.Max(0.0, Math.Min(lower, upper)) * enorm);
            }
            filters[mel] = filter;
        }
        return filters;
    }
}

/// <summary>Minimal real STFT magnitude computation (reflect-padded, Hann window, center=True)
/// shared only by <see cref="RvcRmvpeMelExtractor"/> -- deliberately not reusing any other mel
/// extractor's private STFT since RMVPE's real n_fft/hop combination (1024/160) doesn't match any
/// existing one in this codebase.</summary>
internal static class Stft
{
    public static float[][] Magnitude(ReadOnlySpan<float> waveform, int nFft, int hop, bool reflectPad)
    {
        int pad = nFft / 2;
        var padded = reflectPad ? ReflectPad(waveform, pad) : waveform.ToArray();
        var window = HannWindow(nFft);

        int frames = (padded.Length - nFft) / hop + 1;
        var output = new float[frames][];
        int freqBins = nFft / 2 + 1;

        var real = new double[nFft];
        var imag = new double[nFft];
        for (int f = 0; f < frames; f++)
        {
            int start = f * hop;
            for (int i = 0; i < nFft; i++) real[i] = padded[start + i] * window[i];
            Array.Clear(imag);
            Dft(real, imag);

            var mag = new float[freqBins];
            for (int k = 0; k < freqBins; k++)
                mag[k] = (float)Math.Sqrt(real[k] * real[k] + imag[k] * imag[k]);
            output[f] = mag;
        }
        return output;
    }

    private static float[] ReflectPad(ReadOnlySpan<float> x, int pad)
    {
        var output = new float[x.Length + 2 * pad];
        for (int i = 0; i < pad; i++) output[i] = x[Math.Min(pad - i, x.Length - 1)];
        x.CopyTo(output.AsSpan(pad));
        for (int i = 0; i < pad; i++) output[pad + x.Length + i] = x[Math.Max(x.Length - 2 - i, 0)];
        return output;
    }

    private static float[] HannWindow(int n)
    {
        var w = new float[n];
        for (int i = 0; i < n; i++) w[i] = 0.5f - 0.5f * MathF.Cos(2f * MathF.PI * i / n);
        return w;
    }

    /// <summary>Direct O(n^2) DFT (real magnitude spectrum only needed, correctness over speed --
    /// n_fft=1024 is small enough this is not a real bottleneck for pitch extraction's frame rate).</summary>
    private static void Dft(double[] real, double[] imag)
    {
        int n = real.Length;
        var outRe = new double[n];
        var outIm = new double[n];
        for (int k = 0; k < n / 2 + 1; k++)
        {
            double sumRe = 0, sumIm = 0;
            for (int t = 0; t < n; t++)
            {
                double angle = -2.0 * Math.PI * k * t / n;
                double c = Math.Cos(angle), s = Math.Sin(angle);
                sumRe += real[t] * c - imag[t] * s;
                sumIm += real[t] * s + imag[t] * c;
            }
            outRe[k] = sumRe;
            outIm[k] = sumIm;
        }
        Array.Copy(outRe, real, n / 2 + 1);
        Array.Copy(outIm, imag, n / 2 + 1);
    }
}
