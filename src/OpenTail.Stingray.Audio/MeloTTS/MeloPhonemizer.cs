namespace OpenTail.Stingray.Audio.MeloTTS;

/// <summary>
/// Multilingual phonemizer for MeloTTS: extracts phone IDs, tone IDs (0..4), and language IDs.
/// </summary>
public sealed class MeloPhonemizer
{
    private readonly Dictionary<char, int> _phoneMap = [];

    public MeloPhonemizer()
    {
        InitializePhoneMap();
    }

    private void InitializePhoneMap()
    {
        _phoneMap['_'] = 0; // Pad
        _phoneMap['^'] = 1; // BOS
        _phoneMap['$'] = 2; // EOS
        _phoneMap[' '] = 3;

        string symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" +
                         "áàâäãåçéèêëíìîïñóòôöõúùûüýÿæœøɑæɐɒɔɕçɗɖðʤəɚɛɜɝɟɡɦɨɪʝɭɬɫɮɱɯŋɳɲɴøɵɸθɹɾɻʁʂʃʈʧʉʊʌʍʎʐʒʔˈˌːˑ";

        int id = 4;
        foreach (char c in symbols)
        {
            if (!_phoneMap.ContainsKey(c))
            {
                _phoneMap[c] = id++;
            }
        }
    }

    /// <summary>
    /// Phonemizes input text and returns phone IDs, tone IDs, and language IDs.
    /// </summary>
    public MeloPhonemeResult Phonemize(string text, string language = "EN-US")
    {
        int langId = language.ToUpperInvariant() switch
        {
            "ZH" => 1,
            "ES" => 2,
            "FR" => 3,
            "JP" => 4,
            "KR" => 5,
            _ => 0 // English (EN-US, EN-BR, etc.)
        };

        if (string.IsNullOrWhiteSpace(text))
        {
            return new MeloPhonemeResult([0, 1, 0, 2, 0], [0, 0, 0, 0, 0], [langId, langId, langId, langId, langId]);
        }

        var rawPhones = new List<int> { 1 }; // BOS
        var rawTones = new List<int> { 0 };

        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            int pid = _phoneMap.TryGetValue(c, out int id) ? id : 3;
            rawPhones.Add(pid);

            // Estimate tone (for tonal languages, numbers 1-4 or vowel position)
            int tone = (char.IsDigit(c) && c >= '1' && c <= '4') ? (c - '0') : 0;
            rawTones.Add(tone);
        }

        rawPhones.Add(2); // EOS
        rawTones.Add(0);

        // Intersperse pad token (0)
        int total = rawPhones.Count * 2 + 1;
        var phones = new int[total];
        var tones = new int[total];
        var langIds = new int[total];

        for (int i = 0; i < rawPhones.Count; i++)
        {
            phones[i * 2] = 0;
            tones[i * 2] = 0;
            langIds[i * 2] = langId;

            phones[i * 2 + 1] = rawPhones[i];
            tones[i * 2 + 1] = rawTones[i];
            langIds[i * 2 + 1] = langId;
        }

        phones[^1] = 0;
        tones[^1] = 0;
        langIds[^1] = langId;

        return new MeloPhonemeResult(phones, tones, langIds);
    }
}

public sealed record MeloPhonemeResult(int[] Phones, int[] Tones, int[] LangIds);
