
namespace OpenTail.Stingray.Audio.Rvc;

/// <summary>
/// Real RVC content-audio preprocessing, transcribed directly from `high_pass_48hz_in_place` and
/// the `audio_pad_duration_sec` reflect-pad step in
/// `examples/audio.cpp/src/models/rvc/native_pipeline.cpp` (not guessed). Both HuBERT content
/// encoding and RMVPE pitch extraction run on this SAME preprocessed signal (48Hz zero-phase
/// Butterworth high-pass via filtfilt, then reflect-padded by `audio_pad_duration_sec` seconds on
/// each side, default 1s) -- found via a real numeric mismatch: RMVPE's own end-to-end salience
/// stats (mean/std/max) were off by ~40x from the reference until this preprocessing was added,
/// and the reference's own frame counts (796 padded vs our un-padded 596 for the same source clip)
/// confirmed the padding was the missing step.
/// </summary>
public static class RvcAudioPreprocessing
{
    public const int ContentSampleRate = 16000;

    /// <summary>5th-order Butterworth high-pass, cutoff 48Hz @ 16kHz, applied via scipy-style
    /// `filtfilt` (odd reflect-extend by padlen=18, forward+reverse `lfilter`, then trim back) --
    /// coefficients and zi taken directly from the reference's own literal constants.</summary>
    public static void HighPass48HzInPlace(Span<float> samples)
    {
        if (samples.Length == 0) return;
        const int order = 5;
        const int padlen = 18;
        double[] b =
        [
            0.9699606451838447,
            -4.849803225919223,
            9.699606451838447,
            -9.699606451838447,
            4.849803225919223,
            -0.9699606451838447,
        ];
        double[] a =
        [
            1.0,
            -4.939001819168364,
            9.757863526739543,
            -9.639544849413458,
            4.761506797356209,
            -0.9408236532054606,
        ];
        double[] zi =
        [
            -0.9699604796995847,
            3.8798419288925783,
            -5.819762908173043,
            3.879841948472456,
            -0.9699604894923387,
        ];
        if (samples.Length <= padlen)
            throw new ArgumentException("RVC high-pass filtfilt input is too short", nameof(samples));

        var working = new double[samples.Length + 2 * padlen];
        double first = samples[0];
        double last = samples[^1];
        for (int i = 0; i < padlen; i++)
            working[i] = 2.0 * first - samples[padlen - i];
        for (int i = 0; i < samples.Length; i++)
            working[padlen + i] = samples[i];
        for (int i = 0; i < padlen; i++)
            working[padlen + samples.Length + i] = 2.0 * last - samples[samples.Length - 2 - i];

        void LFilter(double[] values)
        {
            Span<double> state = stackalloc double[order];
            for (int i = 0; i < order; i++) state[i] = zi[i] * values[0];
            for (int t = 0; t < values.Length; t++)
            {
                double x = values[t];
                double y = b[0] * x + state[0];
                for (int i = 1; i < order; i++)
                    state[i - 1] = b[i] * x + state[i] - a[i] * y;
                state[order - 1] = b[order] * x - a[order] * y;
                values[t] = y;
            }
        }

        LFilter(working);
        Array.Reverse(working);
        LFilter(working);
        Array.Reverse(working);

        for (int i = 0; i < samples.Length; i++)
            samples[i] = (float)working[padlen + i];
    }

    /// <summary>Reflect-pads without repeating the edge sample (numpy/PyTorch `mode="reflect"`
    /// convention), matching the reference's own `reflect_pad_samples`.</summary>
    public static float[] ReflectPad(ReadOnlySpan<float> samples, long leftPad, long rightPad)
    {
        if (samples.Length == 0) throw new ArgumentException("reflect padding requires non-empty samples");
        if (leftPad < 0 || rightPad < 0) throw new ArgumentException("reflect padding size is invalid");
        long n = samples.Length;
        var output = new float[n + leftPad + rightPad];
        for (long i = 0; i < output.Length; i++)
        {
            long index = i - leftPad;
            while (index < 0 || index >= n)
            {
                if (index < 0) index = -index;
                if (index >= n) index = 2 * n - index - 2;
            }
            output[i] = samples[(int)index];
        }
        return output;
    }

    /// <summary>Full real preprocessing applied before both HuBERT content encoding and RMVPE
    /// pitch extraction: mono 16kHz -> 48Hz high-pass (filtfilt, in place) -> reflect-pad by
    /// `audioPadDurationSec` seconds on each side (default 1, matching the reference CLI's
    /// default `audio_pad_duration_sec` option).</summary>
    public static float[] PrepareContentAudio(ReadOnlySpan<float> waveform16k, int audioPadDurationSec = 1)
    {
        var content = waveform16k.ToArray();
        HighPass48HzInPlace(content);
        long padSamples = (long)ContentSampleRate * audioPadDurationSec;
        return ReflectPad(content, padSamples, padSamples);
    }
}
