
namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// Text tokenizer for Chatterbox-Turbo speech synthesis. When constructed with real GGUF weights
/// (<see cref="ChatterboxWeights"/>), wraps the real BPE tokenizer the GGUF carries in its
/// tokenizer.ggml.tokens/merges metadata (<see cref="GgufTokenizer"/> -- the same tokenizer type
/// used for this codebase's text LLMs), applies the reference's punc_norm cleanup
/// (examples/chatterbox-tts-py/chatterbox/tts_turbo.py's punc_norm), and wraps the result in T3's
/// start/stop text tokens. Without weights, falls back to a minimal char-level tokenizer (used only
/// by the no-model test/demo path).
/// </summary>
public sealed class ChatterboxTokenizer
{
    private readonly Dictionary<string, int> _vocab = new(StringComparer.Ordinal);
    private readonly Dictionary<char, int> _charVocab = [];

    private readonly GgufTokenizer? _real;
    private readonly int _startTextToken;
    private readonly int _stopTextToken;

    public ChatterboxTokenizer()
    {
        InitializeVocab();
    }

    public ChatterboxTokenizer(ChatterboxWeights weights)
    {
        _real = weights.Tokenizer;
        _startTextToken = weights.StartTextToken;
        _stopTextToken = weights.StopTextToken;
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
    /// Cleans up punctuation the way LLM-generated or otherwise irregular input text tends to use,
    /// ported directly from tts_turbo.py's punc_norm.
    /// </summary>
    public static string PuncNorm(string text)
    {
        if (text.Length == 0) return "You need to add some text for me to talk.";

        if (char.IsLower(text[0])) text = char.ToUpperInvariant(text[0]) + text[1..];

        text = string.Join(" ", text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));

        (string, string)[] replacements =
        [
            ("…", ", "), (":", ","), ("—", "-"), ("–", "-"), (" ,", ","),
            ("“", "\""), ("”", "\""), ("‘", "'"), ("’", "'"),
        ];
        foreach (var (oldSeq, newChar) in replacements)
            text = text.Replace(oldSeq, newChar);

        text = text.TrimEnd(' ');
        char[] sentenceEnders = ['.', '!', '?', '-', ','];
        if (text.Length == 0 || Array.IndexOf(sentenceEnders, text[^1]) < 0)
            text += ".";

        return text;
    }

    /// <summary>
    /// Encodes input text into token IDs, including the model's start/stop wrapping.
    /// </summary>
    public int[] Encode(string text)
    {
        if (_real is { } tok)
        {
            string normalized = PuncNorm(text);
            var ids = tok.Encode(normalized);
            var result = new int[ids.Count + 2];
            result[0] = _startTextToken;
            for (int i = 0; i < ids.Count; i++) result[i + 1] = ids[i];
            result[^1] = _stopTextToken;
            return result;
        }

        if (string.IsNullOrWhiteSpace(text)) return [1, 2];

        var tokens = new List<int>(text.Length + 2) { 1 };
        foreach (char c in text)
        {
            tokens.Add(_charVocab.TryGetValue(c, out int id) ? id : 3 /* <unk> */);
        }
        tokens.Add(2);
        return tokens.ToArray();
    }
}
