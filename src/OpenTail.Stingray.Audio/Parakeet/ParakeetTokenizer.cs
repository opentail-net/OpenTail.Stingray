
namespace OpenTail.Stingray.Audio.Parakeet;

/// <summary>
/// SentencePiece BPE tokenizer with subword reconstruction for NVIDIA NeMo Parakeet ASR.
/// Supports both standard procedural vocab and GGUF header tokens via <see cref="FromGguf"/>.
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

    /// <summary>
    /// Constructs a Parakeet tokenizer by ingesting vocabulary tokens directly from GGUF metadata.
    /// </summary>
    public static ParakeetTokenizer FromGguf(GgufModel model)
    {
        var tokenizer = new ParakeetTokenizer();
        if (model.Metadata.TryGetValue("tokenizer.ggml.tokens", out var obj) && obj is object[] tokens)
        {
            tokenizer._vocab.Clear();
            tokenizer._idToToken.Clear();

            for (int i = 0; i < tokens.Length; i++)
            {
                string tok = tokens[i]?.ToString() ?? string.Empty;
                tokenizer._vocab[tok] = i;
                tokenizer._idToToken[i] = tok;
            }
        }
        return tokenizer;
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
            "\u2581speech", "\u2581recognition", "\u2581model", "\u2581fast", "\u2581accurate"
        ];

        foreach (var word in commonWords)
        {
            if (!_vocab.ContainsKey(word))
            {
                _vocab[word] = id;
                _idToToken[id] = word;
                id++;
            }
        }
    }

    /// <summary>
    /// Encodes a reference text string into a sequence of SentencePiece token IDs.
    /// </summary>
    public int[] Encode(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return [];

        var tokens = new List<int>();
        string[] words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        for (int w = 0; w < words.Length; w++)
        {
            string piece = "\u2581" + words[w];
            if (_vocab.TryGetValue(piece, out int tid))
            {
                tokens.Add(tid);
            }
            else
            {
                for (int c = 0; c < words[w].Length; c++)
                {
                    string charPiece = (c == 0) ? ("\u2581" + words[w][c]) : words[w][c].ToString();
                    if (_vocab.TryGetValue(charPiece, out int cid))
                    {
                        tokens.Add(cid);
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
    /// Decodes a sequence of emitted token IDs into a reconstructed text string.
    /// Handles SentencePiece prefix '\u2581' space boundary replacement.
    /// </summary>
    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        if (tokenIds.IsEmpty) return string.Empty;

        var sb = new StringBuilder();
        foreach (int tid in tokenIds)
        {
            if (tid == BlankTokenId || tid == BosTokenId || tid == EosTokenId)
            {
                continue;
            }

            if (_idToToken.TryGetValue(tid, out var token))
            {
                if (token.StartsWith("\u2581"))
                {
                    if (sb.Length > 0) sb.Append(' ');
                    sb.Append(token[1..]);
                }
                else
                {
                    sb.Append(token);
                }
            }
        }

        return sb.ToString().Trim();
    }

    public string GetToken(int tokenId)
    {
        return _idToToken.TryGetValue(tokenId, out var t) ? t : "<unk>";
    }
}
