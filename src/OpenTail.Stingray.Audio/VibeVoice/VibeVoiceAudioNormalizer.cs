namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>
/// Real RMS-based loudness normalization for VibeVoice ASR input audio, ported from
/// `frontend.cpp`'s `VibeVoiceASRFrontend::normalize` (not guessed). Applied to the already-
/// resampled (target sample rate) mono waveform, BEFORE the tokenizer encoders run: scales the
/// waveform so its RMS matches a target dBFS level, then re-scales down if that pushes any sample
/// past full scale. Real default config values (`VibeVoiceAudioProcessorConfig` in `assets.h`):
/// `normalize_audio=true`, `target_db_fs=-25.0`, `eps=1e-6`.
/// </summary>
public static class VibeVoiceAudioNormalizer
{
    public static float[] Normalize(float[] waveform, float targetDbFs = -25.0f, float eps = 1e-6f)
    {
        double sumSq = 0.0;
        foreach (var s in waveform) sumSq += (double)s * s;
        float rms = (float)Math.Sqrt(sumSq / Math.Max(waveform.Length, 1));
        float target = MathF.Pow(10.0f, targetDbFs / 20.0f);
        float gain = target / (rms + eps);

        var output = new float[waveform.Length];
        float maxAbs = 0.0f;
        for (int i = 0; i < waveform.Length; i++)
        {
            output[i] = waveform[i] * gain;
            maxAbs = Math.Max(maxAbs, Math.Abs(output[i]));
        }
        if (maxAbs > 1.0f)
        {
            float scale = maxAbs + eps;
            for (int i = 0; i < output.Length; i++) output[i] /= scale;
        }
        return output;
    }
}
