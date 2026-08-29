
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

    /// <summary>
    /// The REAL Kokoro-82M phoneme-&gt;id vocabulary, transcribed verbatim from the real
    /// <c>hexgrad/Kokoro-82M</c> model's own published <c>config.json</c> ("vocab" key,
    /// <c>huggingface.co/hexgrad/Kokoro-82M/raw/main/config.json</c>, Apache-2.0 -- same license
    /// as the model weights this port already loads). This is NOT the same thing as "every IPA
    /// symbol Kokoro's decoder happens to have an embedding row for" -- id assignment is that
    /// specific model's own training-time tokenizer table, and a locally-guessed enumeration
    /// (this class's prior implementation: walk a hardcoded symbol string, assign sequential ids)
    /// produces a systematically different mapping even when the SET of symbols matches, silently
    /// feeding every input through the wrong embedding rows. id 0 is pad/silence (no vocab entry
    /// for it in the real config; every prior/existing '$' pad convention in this file maps to it).
    /// </summary>
    private static readonly Dictionary<char, int> RealKokoroVocab = new()
    {
        [';'] = 1, [':'] = 2, [','] = 3, ['.'] = 4, ['!'] = 5, ['?'] = 6,
        ['—'] = 9, ['…'] = 10, ['"'] = 11, ['('] = 12, [')'] = 13, ['“'] = 14, ['”'] = 15,
        [' '] = 16, ['̃'] = 17, ['ʣ'] = 18, ['ʥ'] = 19, ['ʦ'] = 20, ['ʨ'] = 21, ['ᵝ'] = 22, ['ꭧ'] = 23,
        ['A'] = 24, ['I'] = 25, ['O'] = 31, ['Q'] = 33, ['S'] = 35, ['T'] = 36, ['W'] = 39, ['Y'] = 41, ['ᵊ'] = 42,
        ['a'] = 43, ['b'] = 44, ['c'] = 45, ['d'] = 46, ['e'] = 47, ['f'] = 48, ['h'] = 50, ['i'] = 51, ['j'] = 52,
        ['k'] = 53, ['l'] = 54, ['m'] = 55, ['n'] = 56, ['o'] = 57, ['p'] = 58, ['q'] = 59, ['r'] = 60, ['s'] = 61,
        ['t'] = 62, ['u'] = 63, ['v'] = 64, ['w'] = 65, ['x'] = 66, ['y'] = 67, ['z'] = 68,
        ['ɑ'] = 69, ['ɐ'] = 70, ['ɒ'] = 71, ['æ'] = 72, ['β'] = 75, ['ɔ'] = 76, ['ɕ'] = 77, ['ç'] = 78, ['ɖ'] = 80,
        ['ð'] = 81, ['ʤ'] = 82, ['ə'] = 83, ['ɚ'] = 85, ['ɛ'] = 86, ['ɜ'] = 87, ['ɟ'] = 90, ['ɡ'] = 92, ['ɥ'] = 99,
        ['ɨ'] = 101, ['ɪ'] = 102, ['ʝ'] = 103, ['ɯ'] = 110, ['ɰ'] = 111, ['ŋ'] = 112, ['ɳ'] = 113, ['ɲ'] = 114,
        ['ɴ'] = 115, ['ø'] = 116, ['ɸ'] = 118, ['θ'] = 119, ['œ'] = 120, ['ɹ'] = 123, ['ɾ'] = 125, ['ɻ'] = 126,
        ['ʁ'] = 128, ['ɽ'] = 129, ['ʂ'] = 130, ['ʃ'] = 131, ['ʈ'] = 132, ['ʧ'] = 133, ['ʊ'] = 135, ['ʋ'] = 136,
        ['ʌ'] = 138, ['ɣ'] = 139, ['ɤ'] = 140, ['χ'] = 142, ['ʎ'] = 143, ['ʒ'] = 147, ['ʔ'] = 148,
        ['ˈ'] = 156, ['ˌ'] = 157, ['ː'] = 158, ['ʰ'] = 162, ['ʲ'] = 164,
        ['↓'] = 169, ['→'] = 171, ['↗'] = 172, ['↘'] = 173, ['ᵻ'] = 177,
    };

    private void InitializeDefaultVocab()
    {
        _vocab["$"] = 0; // Pad / silence -- no real vocab entry, matches this class's own convention
        _charVocab['$'] = 0;

        foreach (var (c, id) in RealKokoroVocab)
        {
            _vocab[c.ToString()] = id;
            _charVocab[c] = id;
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
