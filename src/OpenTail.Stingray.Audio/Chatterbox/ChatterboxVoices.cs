namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// Preset speaker styles and feature banks for Chatterbox-Turbo TTS.
/// </summary>
public static class ChatterboxVoices
{
    public const int FeatureDim = 512;

    public static readonly string[] AvailableVoices =
    [
        "resemble_default",
        "narrator",
        "conversational",
        "emotive"
    ];

    public static float[] GetSpeakerFeatures(string voice)
    {
        var features = new float[FeatureDim];
        int seed = voice.ToLowerInvariant() switch
        {
            "narrator" => 101,
            "conversational" => 202,
            "emotive" => 303,
            _ => 42
        };

        for (int i = 0; i < FeatureDim; i++)
        {
            features[i] = 0.15f * MathF.Sin(seed * 0.314f + i * 0.17f);
        }

        return features;
    }
}
