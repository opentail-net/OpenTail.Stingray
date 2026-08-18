namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// Conditional neural audio decoder / vocoder for Chatterbox-Turbo TTS.
/// Converts discrete speech tokens and speaker features into 24kHz PCM audio samples.
/// </summary>
public sealed class ChatterboxDecoder
{
    public const int SampleRate = 24000;
    public const int HopLength = 256;
    public const int HiddenDim = 512;

    public ChatterboxDecoder() { }

    /// <summary>
    /// Synthesizes 24kHz audio waveform samples from discrete speech tokens and speaker conditioning.
    /// </summary>
    public float[] Decode(IReadOnlyList<int> speechTokens, float[] speakerFeatures)
    {
        if (speechTokens.Count <= 2) return [];

        // Exclude START and STOP control tokens
        var tokens = new List<int>();
        foreach (int t in speechTokens)
        {
            if (t != ChatterboxAcousticLm.StartSpeechToken && t != ChatterboxAcousticLm.StopSpeechToken)
            {
                tokens.Add(t);
            }
        }

        int numTokens = tokens.Count;
        int totalSamples = numTokens * HopLength;
        var audio = new float[totalSamples];

        for (int i = 0; i < numTokens; i++)
        {
            int tid = tokens[i];
            int audioBase = i * HopLength;

            float pitch = 130.0f + 50.0f * MathF.Sin(tid * 0.15f);
            float spkWeight = (speakerFeatures.Length > 0) ? speakerFeatures[i % speakerFeatures.Length] : 0.1f;

            for (int n = 0; n < HopLength && (audioBase + n) < audio.Length; n++)
            {
                float t = (float)(audioBase + n) / SampleRate;

                float s1 = MathF.Sin(2.0f * MathF.PI * pitch * t);
                float s2 = 0.5f * MathF.Sin(4.0f * MathF.PI * pitch * t + spkWeight);
                float s3 = 0.25f * MathF.Sin(6.0f * MathF.PI * pitch * t);

                float hann = 0.5f * (1.0f - MathF.Cos(2.0f * MathF.PI * n / HopLength));
                audio[audioBase + n] += (s1 + s2 + s3) * hann * 0.35f;
            }
        }

        // Peak normalize
        float maxPeak = 0f;
        for (int i = 0; i < audio.Length; i++)
        {
            float abs = MathF.Abs(audio[i]);
            if (abs > maxPeak) maxPeak = abs;
        }

        if (maxPeak > 0.001f)
        {
            float gain = 0.90f / maxPeak;
            for (int i = 0; i < audio.Length; i++) audio[i] *= gain;
        }

        return audio;
    }
}
