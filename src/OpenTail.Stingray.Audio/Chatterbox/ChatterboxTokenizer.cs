namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// Text tokenizer for Chatterbox-Turbo speech synthesis.
/// </summary>
public sealed class ChatterboxTokenizer
{
    private readonly Dictionary<string, int> _vocab = new(StringComparer.Ordinal);
    private readonly Dictionary<char, int> _charVocab = [];

    public ChatterboxTokenizer()
    {
        InitializeVocab();
    }

    private void InitializeVocab()
    {
        _vocab["<pad>"] = 0;
        _vocab["<s>"] = 1;
        _vocab["</s>"] = 2;
        _vocab["<unk>"] = 3;

        string symbols = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
        int id = 4;
        foreach (char c in symbols)
        {
            string s = c.ToString();
            if (!_vocab.ContainsKey(s))
            {
                _vocab[s] = id;
                _charVocab[c] = id;
                id++;
            }
        }
    }

    /// <summary>
    /// Encodes input text into token IDs.
    /// </summary>
    public int[] Encode(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return [1, 2];

        var tokens = new List<int>(text.Length + 2) { 1 }; // <s>

        foreach (char c in text)
        {
            if (_charVocab.TryGetValue(c, out int id))
            {
                tokens.Add(id);
            }
            else
            {
                tokens.Add(3); // <unk>
            }
        }

        tokens.Add(2); // </s>
        return tokens.ToArray();
    }
}
