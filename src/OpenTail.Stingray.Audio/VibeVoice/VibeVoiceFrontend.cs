namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>Real `audio_processor` config fields, ported from VibeVoice ASR's real config
/// (`assets.processor.audio_processor` in `frontend.cpp`).</summary>
public sealed class VibeVoiceFrontendConfig
{
    public required int SampleRate { get; init; }
    public required bool NormalizeAudio { get; init; }
    public required float TargetDbFs { get; init; }
    public required float Eps { get; init; }
}

/// <summary>
/// Real audio frontend normalization for VibeVoice ASR, ported from `frontend.cpp`'s
/// `VibeVoiceASRFrontend::normalize` (not guessed): mono downmix (average of interleaved
/// channels) -&gt; resample to the tokenizer's real sample rate (this port takes ALREADY-mono,
/// already-at-target-rate input and only applies the loudness normalization stage -- resampling
/// itself is out of scope here, matching this codebase's existing convention of doing
/// resampling once via a shared utility rather than per-pipeline) -&gt; optional RMS-based
/// loudness normalization to a target dBFS, with a real peak-limiting safeguard (`max_abs &gt;
/// 1.0` triggers a second rescale pass) so the output never clips.
/// </summary>
public static class VibeVoiceFrontend
{
    /// <summary>Applies the real RMS-normalize-to-target-dBFS + peak-limit stages to an
    /// already-mono, already-resampled waveform. No-ops (returns a copy) when
    /// <see cref="VibeVoiceFrontendConfig.NormalizeAudio"/> is false.</summary>
    public static float[] Normalize(float[] mono, VibeVoiceFrontendConfig config)
    {
        var output = (float[])mono.Clone();
        if (!config.NormalizeAudio) return output;

        double sumSq = 0.0;
        foreach (float s in output) sumSq += (double)s * s;
        float rms = (float)Math.Sqrt(sumSq / Math.Max(output.Length, 1));
        float target = MathF.Pow(10f, config.TargetDbFs / 20f);
        float gain = target / (rms + config.Eps);

        float maxAbs = 0f;
        for (int i = 0; i < output.Length; i++)
        {
            output[i] *= gain;
            maxAbs = MathF.Max(maxAbs, MathF.Abs(output[i]));
        }

        if (maxAbs > 1.0f)
        {
            float scale = maxAbs + config.Eps;
            for (int i = 0; i < output.Length; i++) output[i] /= scale;
        }

        return output;
    }
}
