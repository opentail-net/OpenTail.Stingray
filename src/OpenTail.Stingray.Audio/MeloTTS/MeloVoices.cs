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
            "EN-US" or "EN" or "EN-DEFAULT" => 1,
            "ZH" => 0,
            _ => 1 // Default to active speaker in melotts-zh_en
        };
    }
}
