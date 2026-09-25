namespace OpenTail.Stingray.Audio.MeloTTS;

/// <summary>
/// Preset regional accents and speaker IDs for MeloTTS.
/// </summary>
public static class MeloVoices
{
    public static readonly string[] AvailableVoices =
    [
        "EN-US",
        "EN-BR",
        "EN-INDIA",
        "EN-AU",
        "EN-Default",
        "ZH",
        "ES",
        "FR",
        "JP",
        "KR"
    ];

    public static int GetSpeakerId(string voice)
    {
        return voice.ToUpperInvariant() switch
        {
            // melotts-zh_en has a single speaker, id 1 ("ZH-MIX-EN" in the MeloTTS.cpp reference,
            // src/tts.cpp speaker_ids); ZH used to map to 0, which this checkpoint does not use.
            _ => 1
        };
    }
}
