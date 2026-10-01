namespace OpenTail.Stingray.Core;

/// <summary>
/// The RWKV "world" tokenizer (<c>tokenizer.ggml.model = rwkv</c>, LLAMA_VOCAB_TYPE_RWKV): greedy
/// longest-match over raw UTF-8 bytes against a byte trie of the vocabulary. There are no merges
/// and no scores. Port of llama-vocab.cpp's <c>llm_tokenizer_rwkv</c> /
/// <c>llm_tokenizer_rwkv_session</c>: vocab strings are stored escaped in the GGUF and unescaped
/// with <see cref="Unescape"/> (<c>llama_unescape_rwkv_token</c>) both to build the trie and to
/// decode a token to bytes. No BOS/EOS is added and no space prefix is applied.
/// </summary>
public sealed class RwkvTokenizer
{
    // Trie edges keyed by (node << 8) | byte; node 0 is the root.
    private readonly Dictionary<int, int> _edges = new();
    private readonly List<int> _value = [-1];
    private readonly byte[][] _tokenBytes;
    private readonly int _unknownTokenId;

    public RwkvTokenizer(IReadOnlyList<string> tokens, int unknownTokenId)
    {
        _unknownTokenId = unknownTokenId;
        _tokenBytes = new byte[tokens.Count][];
        for (int id = 0; id < tokens.Count; id++)
        {
            var bytes = Unescape(tokens[id]);
            _tokenBytes[id] = bytes;
            int node = 0;
            foreach (byte b in bytes)
            {
                int key = (node << 8) | b;
                if (!_edges.TryGetValue(key, out int child))
                {
                    child = _value.Count;
                    _value.Add(-1);
                    _edges[key] = child;
                }
                node = child;
            }
            // naive_trie::insert overwrites, so a later duplicate id wins, as in llama.cpp.
            if (bytes.Length > 0) _value[node] = id;
        }
    }

    /// <summary>Greedy longest-match tokenization of <paramref name="text"/>'s UTF-8 bytes.</summary>
    public IReadOnlyList<int> Encode(string text)
    {
        byte[] bytes = System.Text.Encoding.UTF8.GetBytes(text);
        var output = new List<int>(bytes.Length / 3 + 1);
        int position = 0;
        while (position < bytes.Length)
        {
            int node = 0, tokenId = -1, tokenEnd = position;
            for (int p = position; p < bytes.Length && _edges.TryGetValue((node << 8) | bytes[p], out node); p++)
            {
                if (_value[node] >= 0) { tokenId = _value[node]; tokenEnd = p + 1; }
            }
            if (tokenId < 0)
            {
                // llama.cpp emits UNK and advances one byte when no vocab entry starts here.
                output.Add(_unknownTokenId);
                position++;
                continue;
            }
            output.Add(tokenId);
            position = tokenEnd;
        }
        return output;
    }

    /// <summary>The raw bytes a token stands for (llama_vocab::token_to_piece for RWKV).</summary>
    public byte[] DecodeBytes(int token) =>
        (uint)token < (uint)_tokenBytes.Length ? _tokenBytes[token] : [];

    /// <summary>
    /// <c>llama_unescape_rwkv_token</c>: <c>\t \n \r</c>, <c>\xHH</c> (lowercase hex) and
    /// <c>\c</c> → <c>c</c>; everything else is copied as its UTF-8 bytes.
    /// </summary>
    public static byte[] Unescape(string escaped)
    {
        byte[] input = System.Text.Encoding.UTF8.GetBytes(escaped);
        var output = new List<byte>(input.Length);
        bool escaping = false;
        int hexRemaining = 0, hexAcc = 0;
        foreach (byte c in input)
        {
            if (hexRemaining != 0)
            {
                int value = c >= 'a' ? c - 'a' + 10 : c - '0';
                hexAcc = (hexAcc << 4) + value;
                if (--hexRemaining == 0) { output.Add((byte)hexAcc); hexAcc = 0; }
                continue;
            }
            if (escaping)
            {
                switch (c)
                {
                    case (byte)'t': output.Add((byte)'\t'); break;
                    case (byte)'n': output.Add((byte)'\n'); break;
                    case (byte)'r': output.Add((byte)'\r'); break;
                    case (byte)'x': hexRemaining = 2; break;
                    default: output.Add(c); break;
                }
                escaping = false;
                continue;
            }
            if (c == '\\') { escaping = true; continue; }
            output.Add(c);
        }
        return output.ToArray();
    }
}
