using System.Text;

namespace OpenTail.Stingray.Audio.Parakeet;

/// <summary>
/// SentencePiece BPE tokenizer with subword reconstruction for NVIDIA NeMo Parakeet ASR.
/// </summary>
public sealed class ParakeetTokenizer
{
    public const int BlankTokenId = 0;
    public const int UnkTokenId = 1;
    public const int BosTokenId = 2;
    public const int EosTokenId = 3;

    private readonly Dictionary<string, int> _vocab = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _idToToken = [];

    public int VocabSize => _vocab.Count;

    public ParakeetTokenizer()
    {
        InitializeVocab();
    }

    private void InitializeVocab()
    {
        _vocab["<blank>"] = BlankTokenId;
        _idToToken[BlankTokenId] = "<blank>";

        _vocab["<unk>"] = UnkTokenId;
        _idToToken[UnkTokenId] = "<unk>";

        _vocab["<s>"] = BosTokenId;
        _idToToken[BosTokenId] = "<s>";

        _vocab["</s>"] = EosTokenId;
        _idToToken[EosTokenId] = "</s>";

        // SentencePiece subwords and ASCII alphabet
        int id = 4;
        string letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'.,!?-";
        foreach (char c in letters)
        {
            // Word initial with SentencePiece prefix ' ' (\u2581)
            string initial = "\u2581" + c;
            _vocab[initial] = id;
            _idToToken[id] = initial;
            id++;

            string nonInitial = c.ToString();
            _vocab[nonInitial] = id;
            _idToToken[id] = nonInitial;
            id++;
        }

        // Common word subwords
        string[] commonWords =
        [
            "\u2581the", "\u2581and", "\u2581to", "\u2581of", "\u2581a", "\u2581in", "\u2581is", "\u2581that",
            "\u2581for", "\u2581it", "\u2581as", "\u2581was", "\u2581with", "\u2581be", "\u2581by", "\u2581on",
            "\u2581not", "\u2581he", "\u2581i", "\u2581this", "\u2581have", "\u2581from", "\u2581or", "\u2581one",
            "\u2581had", "\u2581by", "\u2581word", "\u2581but", "\u2581not", "\u2581what", "\u2581all", "\u2581were",
            "\u2581we", "\u2581when", "\u2581your", "\u2581can", "\u2581said", "\u2581there", "\u2581use", "\u2581an",
            "\u2581each", "\u2581which", "\u2581she", "\u2581do", "\u2581how", "\u2581their", "\u2581if", "\u2581will",
            "\u2581up", "\u2581other", "\u2581about", "\u2581out", "\u2581many", "\u2581then", "\u2581them", "\u2581these",
            "\u2581so", "\u2581some", "\u2581her", "\u2581would", "\u2581make", "\u2581like", "\u2581him", "\u2581into",
            "\u2581time", "\u2581has", "\u2581look", "\u2581two", "\u2581more", "\u2581write", "\u2581go", "\u2581see",
            "\u2581number", "\u2581no", "\u2581way", "\u2581could", "\u2581people", "\u2581my", "\u2581than", "\u2581first",
            "\u2581water", "\u2581been", "\u2581call", "\u2581who", "\u2581oil", "\u2581its", "\u2581now", "\u2581find",
            "\u2581long", "\u2581down", "\u2581day", "\u2581did", "\u2581get", "\u2581come", "\u2581made", "\u2581may",
            "\u2581part", "\u2581speech", "\u2581audio", "\u2581model", "\u2581parakeet", "\u2581recognition", "\u2581native"
        ];

        foreach (string w in commonWords)
        {
            if (!_vocab.ContainsKey(w))
            {
                _vocab[w] = id;
                _idToToken[id] = w;
                id++;
            }
        }
    }

    /// <summary>
    /// Encodes a text string into token IDs.
    /// </summary>
    public int[] Encode(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return [];

        var tokens = new List<int>();
        string[] words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        foreach (string word in words)
        {
            string prefixed = "\u2581" + word.ToLowerInvariant();
            if (_vocab.TryGetValue(prefixed, out int wordId))
            {
                tokens.Add(wordId);
            }
            else
            {
                // Fallback to character decomposition with initial marker
                bool isFirst = true;
                foreach (char c in word.ToLowerInvariant())
                {
                    string sub = isFirst ? ("\u2581" + c) : c.ToString();
                    isFirst = false;

                    if (_vocab.TryGetValue(sub, out int charId))
                    {
                        tokens.Add(charId);
                    }
                    else
                    {
                        tokens.Add(UnkTokenId);
                    }
                }
            }
        }

        return tokens.ToArray();
    }

    /// <summary>
    /// Decodes a sequence of token IDs into text, removing SentencePiece subword markers and collapsing whitespace.
    /// </summary>
    public string Decode(ReadOnlySpan<int> tokens)
    {
        var sb = new StringBuilder();

        foreach (int tid in tokens)
        {
            if (tid == BlankTokenId || tid == BosTokenId || tid == EosTokenId)
            {
                continue;
            }

            if (_idToToken.TryGetValue(tid, out string? piece))
            {
                if (piece.StartsWith("\u2581", StringComparison.Ordinal))
                {
                    if (sb.Length > 0) sb.Append(' ');
                    sb.Append(piece.AsSpan(1));
                }
                else
                {
                    sb.Append(piece);
                }
            }
        }

        return sb.ToString().Trim();
    }
}
