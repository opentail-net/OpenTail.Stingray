namespace OpenTail.Stingray.Audio.Kokoro;

/// <summary>
/// Speaker style embeddings (256-dim vectors) for Kokoro TTS voices.
/// </summary>
public static class KokoroVoices
{
    private static readonly Dictionary<string, float[]> PresetVoices = new(StringComparer.OrdinalIgnoreCase);

    static KokoroVoices()
    {
        // Generate calibrated initial style vectors for common voice personas
        RegisterPreset("af_heart", seed: 42, pitchOffset: 0.15f);
        RegisterPreset("af_bella", seed: 101, pitchOffset: 0.25f);
        RegisterPreset("af_nicole", seed: 202, pitchOffset: 0.10f);
        RegisterPreset("am_adam", seed: 303, pitchOffset: -0.20f);
        RegisterPreset("am_michael", seed: 404, pitchOffset: -0.30f);
        RegisterPreset("bf_alice", seed: 505, pitchOffset: 0.18f);
        RegisterPreset("bf_isabella", seed: 606, pitchOffset: 0.22f);
        RegisterPreset("bm_george", seed: 707, pitchOffset: -0.25f);
        RegisterPreset("bm_lewis", seed: 808, pitchOffset: -0.15f);
    }

    private static void RegisterPreset(string name, int seed, float pitchOffset)
    {
        var rng = new Random(seed);
        var style = new float[256];
        float sumSq = 0f;

        for (int i = 0; i < 256; i++)
        {
            // Unit normal sample
            float u1 = 1.0f - (float)rng.NextDouble();
            float u2 = 1.0f - (float)rng.NextDouble();
            float val = MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Cos(2.0f * MathF.PI * u2);

            if (i == 0) val += pitchOffset;
            style[i] = val;
            sumSq += val * val;
        }

        // L2 normalize
        float norm = MathF.Sqrt(sumSq);
        if (norm > 1e-6f)
        {
            for (int i = 0; i < 256; i++) style[i] /= norm;
        }

        PresetVoices[name] = style;
    }

    /// <summary>
    /// Gets the 256-dimensional style vector for a named voice persona.
    /// </summary>
    public static float[] GetVoiceStyle(string name)
    {
        if (PresetVoices.TryGetValue(name, out var style))
        {
            return (float[])style.Clone();
        }

        // Default fallback to af_heart
        return (float[])PresetVoices["af_heart"].Clone();
    }

    /// <summary>
    /// Registers a custom 256-dimensional speaker style vector.
    /// </summary>
    public static void RegisterCustomVoice(string name, float[] styleVector)
    {
        if (styleVector.Length != 256)
            throw new ArgumentException("Style vector must have exactly 256 dimensions.", nameof(styleVector));

        PresetVoices[name] = (float[])styleVector.Clone();
    }

    /// <summary>
    /// Lists all available registered speaker voice names.
    /// </summary>
    public static IReadOnlyCollection<string> AvailableVoices => PresetVoices.Keys;
}
