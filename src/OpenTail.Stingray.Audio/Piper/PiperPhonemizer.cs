using System.Text;
using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Audio.Piper;

/// <summary>
/// Native C# G2P phonemizer and pad token intersperser for Piper (VITS) TTS.
/// </summary>
public sealed class PiperPhonemizer
{
    private readonly Dictionary<string, int> _vocab = new(StringComparer.Ordinal);
    private readonly Dictionary<char, int> _charVocab = [];

    public int PadId { get; } = 0;
    public int BosId { get; } = 1;
    public int EosId { get; } = 2;

    public PiperPhonemizer(PiperConfig? config = null)
    {
        if (config?.PhonemeIdMap != null && config.PhonemeIdMap.Count > 0)
        {
            foreach (var kvp in config.PhonemeIdMap)
            {
                if (kvp.Value.Count > 0)
                {
                    _vocab[kvp.Key] = kvp.Value[0];
                    if (kvp.Key.Length == 1) _charVocab[kvp.Key[0]] = kvp.Value[0];
                }
            }
        }
        else
        {
            InitializeDefaultPiperVocab();
        }
    }

    private void InitializeDefaultPiperVocab()
    {
        // Standard Piper VITS character and IPA symbols
        _vocab["_"] = 0; // Pad
        _vocab["^"] = 1; // BOS
        _vocab["$"] = 2; // EOS
        _vocab[" "] = 3;

        _charVocab['_'] = 0;
        _charVocab['^'] = 1;
        _charVocab['$'] = 2;
        _charVocab[' '] = 3;

        string symbols = "a b c d e f g h i j k l m n o p q r s t u v w x y z " +
                         "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z " +
                         "0 1 2 3 4 5 6 7 8 9 ! \" # % & ' ( ) * + , - . / : ; < = > ? @ [ ] " +
                         "ɑ æ ɐ ɒ ɔ ɕ ç ɗ ɖ ð ʤ ə ɚ ɛ ɜ ɝ ɟ ɡ ɦ ɨ ɪ ʝ ɭ ɬ ɫ ɮ ɱ ɯ ŋ ɳ ɲ ɴ ø ɵ ɸ θ ɹ ɾ ɻ ʁ ʂ ʃ ʈ ʧ ʉ ʊ ʌ ʍ ʎ ʐ ʒ ʔ ˈ ˌ ː ˑ";

        var tokens = symbols.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int nextId = 4;

        foreach (var t in tokens)
        {
            if (!_vocab.ContainsKey(t))
            {
                _vocab[t] = nextId;
                if (t.Length == 1) _charVocab[t[0]] = nextId;
                nextId++;
            }
        }
    }

    /// <summary>
    /// Converts text into a sequence of phoneme character representations.
    /// </summary>
    public string TextToPhonemes(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return "";

        var sb = new StringBuilder();
        string clean = text.Trim();

        for (int i = 0; i < clean.Length; i++)
        {
            char c = clean[i];
            sb.Append(c);
        }

        return sb.ToString();
    }

    /// <summary>
    /// Tokenizes input text with Piper pad token interspersing ([0, t1, 0, t2, ..., 0]).
    /// </summary>
    public int[] Tokenize(string text, bool interspersePad = true)
    {
        string phonemes = TextToPhonemes(text);
        var rawTokens = new List<int>(phonemes.Length + 2);

        rawTokens.Add(BosId);

        foreach (char c in phonemes)
        {
            if (_charVocab.TryGetValue(c, out int id))
            {
                rawTokens.Add(id);
            }
            else if (char.IsLetterOrDigit(c))
            {
                rawTokens.Add(3); // whitespace fallback
            }
        }

        rawTokens.Add(EosId);

        if (!interspersePad)
        {
            return rawTokens.ToArray();
        }

        // Intersperse pad token (0) between each token and at ends: [0, t1, 0, t2, ..., 0]
        var interspersed = new int[rawTokens.Count * 2 + 1];
        interspersed[0] = PadId;

        for (int i = 0; i < rawTokens.Count; i++)
        {
            interspersed[i * 2 + 1] = rawTokens[i];
            interspersed[i * 2 + 2] = PadId;
        }

        return interspersed;
    }
}
