using System.Text;
using System.Text.RegularExpressions;
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.Kokoro;

/// <summary>
/// Native C# Grapheme-to-Phoneme (G2P) and vocabulary tokenizer for Kokoro-82M TTS. Real
/// pronunciation lookup is <see cref="CmuDictG2P"/> (the real ~135k-word CMU Pronouncing
/// Dictionary); <see cref="G2PFallback"/> only ever runs for words outside that dictionary
/// (names, neologisms, typos) -- see CmuDictG2P's own doc comment for why a closed dictionary is
/// the real, standard approach here rather than a hand-written word list.
/// </summary>
public sealed class KokoroPhonemizer
{
    private readonly Dictionary<string, int> _vocab = new(StringComparer.Ordinal);
    private readonly Dictionary<char, int> _charVocab = [];

    public KokoroPhonemizer(Dictionary<string, int>? customVocab = null)
    {
        if (customVocab != null)
        {
            foreach (var kvp in customVocab)
            {
                _vocab[kvp.Key] = kvp.Value;
                if (kvp.Key.Length == 1) _charVocab[kvp.Key[0]] = kvp.Value;
            }
        }
        else
        {
            InitializeDefaultVocab();
        }
    }

    private void InitializeDefaultVocab()
    {
        // Standard Kokoro-82M vocabulary symbols (178 token IDs)
        string symbols = "$ % ( ) , - . / 0 1 2 3 4 5 6 7 8 9 : ; ? A B C D E F G H I J K L M N O P Q R S T U V W X Y Z " +
                         "a b c d e f g h i j k l m n o p q r s t u v w x y z " +
                         "ɑ ɐ ɒ æ ɓ ʙ β ɔ ɕ ç ɗ ɖ ð ʤ ə ɘ ɚ ɛ ɜ ɝ ɞ ɟ ʄ ɡ ɠ ɢ ʛ ɦ ɧ ħ ɥ ʜ ɨ ɪ ʝ ɭ ɬ ɫ ɮ ʟ ɱ ɯ ɰ ŋ ɳ ɲ ɴ ø ɵ ɸ θ œ ɶ ʘ ɹ ɺ ɾ ɻ ʀ ʁ ʂ ʃ ʈ ʧ ʉ ʊ ʋ ⱱ ʌ ɣ ɤ ʍ χ ʎ ʏ ʑ ʐ ʒ ʔ ʡ ʕ ʢ ˈ ˌ ː ˑ ˘ ˞";

        var tokens = symbols.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int id = 1;
        _vocab["$"] = 0; // Pad / silence
        _charVocab['$'] = 0;

        foreach (var t in tokens)
        {
            if (!_vocab.ContainsKey(t))
            {
                _vocab[t] = id;
                if (t.Length == 1) _charVocab[t[0]] = id;
                id++;
            }
        }
    }

    /// <summary>
    /// Converts input plain text into a normalized phoneme string.
    /// </summary>
    public string TextToPhonemes(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return "$";

        var sb = new StringBuilder();
        var words = Regex.Split(text.Trim(), @"(\s+|[.,!?;:])");

        foreach (var word in words)
        {
            if (string.IsNullOrWhiteSpace(word))
            {
                sb.Append(' ');
                continue;
            }

            if (word.Length == 1 && ",.!?;:".Contains(word[0]))
            {
                sb.Append(word);
                continue;
            }

            if (CmuDictG2P.TryLookup(word, out var ipa))
            {
                sb.Append(ipa);
            }
            else
            {
                // Word not in the real cmudict dictionary (name, neologism, typo) -- letter-to-
                // sound approximation is the real fallback here, not a substitute for the
                // dictionary lookup above.
                sb.Append(G2PFallback(word));
            }
        }

        return sb.ToString();
    }

    private static string G2PFallback(string word)
    {
        // Simple letter-to-sound phonetic approximation
        var sb = new StringBuilder(word.Length);
        string lower = word.ToLowerInvariant();

        for (int i = 0; i < lower.Length; i++)
        {
            char c = lower[i];
            if (c == 't' && i + 1 < lower.Length && lower[i + 1] == 'h')
            {
                sb.Append('θ');
                i++;
            }
            else if (c == 's' && i + 1 < lower.Length && lower[i + 1] == 'h')
            {
                sb.Append('ʃ');
                i++;
            }
            else if (c == 'c' && i + 1 < lower.Length && lower[i + 1] == 'h')
            {
                sb.Append('ʧ');
                i++;
            }
            else if (c == 'e' && i == lower.Length - 1 && lower.Length > 2)
            {
                // Silent trailing e
            }
            else
            {
                sb.Append(c);
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Tokenizes a phoneme string into Kokoro vocabulary token indices.
    /// </summary>
    public int[] Tokenize(string phonemes)
    {
        var tokens = new List<int>(phonemes.Length + 2);
        tokens.Add(0); // Start of sequence

        foreach (char c in phonemes)
        {
            if (_charVocab.TryGetValue(c, out int tid))
            {
                tokens.Add(tid);
            }
            else if (c == ' ')
            {
                tokens.Add(0); // Space / pause
            }
        }

        tokens.Add(0); // End of sequence
        return tokens.ToArray();
    }
}
