namespace OpenTail.Stingray.Audio.SenseVoice;

/// <summary>
/// Real `tokens.txt` vocabulary loader for SenseVoice, matching sherpa-onnx's real
/// `SymbolTable` format (`sherpa-onnx/csrc/symbol-table.cc`): one `"{token} {id}"` pair per line,
/// space-separated, id is the vocabulary index used directly for lookup (not necessarily equal to
/// line number, though in practice the real generator (`export-onnx.py`'s `generate_tokens`)
/// writes them in id order 0..vocab_size-1). Tokens use SentencePiece's `▁` (U+2581) word-boundary
/// marker, which is replaced with a plain space at decode time (SentencePiece's own convention),
/// matching sherpa-onnx's own text post-processing for SentencePiece-based symbol tables.
/// </summary>
public sealed class SenseVoiceVocab
{
    private readonly string[] _idToToken;

    private SenseVoiceVocab(string[] idToToken) => _idToToken = idToToken;

    public int VocabSize => _idToToken.Length;

    public static SenseVoiceVocab Load(string tokensPath)
    {
        var lines = File.ReadAllLines(tokensPath);
        var map = new Dictionary<int, string>(lines.Length);
        int maxId = -1;
        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            int sep = line.LastIndexOf(' ');
            if (sep < 0) continue;
            string token = line[..sep];
            if (!int.TryParse(line[(sep + 1)..], out int id)) continue;
            map[id] = token;
            if (id > maxId) maxId = id;
        }

        var arr = new string[maxId + 1];
        for (int i = 0; i <= maxId; i++)
            arr[i] = map.TryGetValue(i, out var tok) ? tok : string.Empty;
        return new SenseVoiceVocab(arr);
    }

    public string TokenAt(int id) => (uint)id < (uint)_idToToken.Length ? _idToToken[id] : string.Empty;

    /// <summary>Joins decoded token ids into real transcript text, replacing SentencePiece's `▁`
    /// word-boundary marker with a literal space (standard SentencePiece detokenization -- the
    /// real reference simply concatenates token pieces raw, so `▁` characters embedded in the
    /// piece text themselves become spaces once concatenated and trimmed).</summary>
    public string Decode(IReadOnlyList<int> ids)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var id in ids) sb.Append(TokenAt(id));
        return sb.ToString().Replace('▁', ' ').Trim();
    }
}
