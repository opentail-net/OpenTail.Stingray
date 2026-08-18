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
            "EN-US" => 0,
            "EN-BR" => 1,
            "EN-INDIA" => 2,
            "EN-AU" => 3,
            "EN-DEFAULT" => 4,
            "ZH" => 10,
            "ES" => 20,
            "FR" => 30,
            "JP" => 40,
            "KR" => 50,
            _ => 0
        };
    }
}
