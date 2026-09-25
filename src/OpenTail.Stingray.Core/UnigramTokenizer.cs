
namespace OpenTail.Stingray.Core;

/// <summary>
/// Real SentencePiece/Hugging Face Unigram tokenizer -- inference-time segmentation ONLY (no
/// training/EM). Built to unblock Parler-TTS's T5 encoder, whose real <c>tokenizer.json</c>
/// declares <c>"model": {"type": "Unigram", ...}</c> -- a genuinely different segmentation
/// algorithm from this codebase's existing BPE tokenizer (<see cref="GgufTokenizer"/>/
/// <see cref="HuggingFaceTokenizerSource"/>), not something that can be reused by swapping the
/// vocabulary. Also serves the XLM-RoBERTa family (xlm-roberta, multilingual-e5, bge-m3,
/// paraphrase-multilingual-MiniLM).
///
/// <para><b>Real algorithm, transcribed from the authoritative sources (not guessed) -- Google's
/// `sentencepiece` `src/unigram_model.cc` (`Model::EncodeOptimized`, `Model::PopulateNodes`,
/// `Lattice::Viterbi`) and Hugging Face's `tokenizers` Rust `models/unigram/model.rs`
/// (`populate_nodes`/`encode_optimized`), cross-checked against each other</b>: build a lattice by
/// UTF-8-byte-trie prefix search of the vocabulary against the normalized input; find the
/// maximum-additive-log-score path via a single-pass Viterbi DP (`ADD` scores, `MAX` at each
/// position -- NOT probability multiplication, NOT logsumexp, which are for the training-time
/// forward/backward/sampling algorithms this class deliberately does not implement); fall back to
/// a one-Unicode-scalar UNK edge, scored `(min score among NORMAL pieces) - 10.0`, wherever no
/// single-character vocabulary piece exists at a position -- both the `-10.0` constant and
/// restricting the minimum to NORMAL-type pieces are literal real behaviors, not approximations.
/// Ties are broken by strict `&gt;` (first-encountered path wins), matching the reference's own
/// `if (best_node == nullptr || score &gt; best_score)`.</para>
///
/// <para><b>Preprocessing follows the <c>tokenizer.json</c>'s own <c>normalizer</c> and
/// <c>pre_tokenizer</c></b>: <c>Precompiled</c> (SentencePiece's charsmap, see
/// <see cref="PrecompiledCharsmap"/>), <c>Strip</c> and <c>Replace</c> normalizers (alone or in a
/// <c>Sequence</c>), then an optional <c>WhitespaceSplit</c> and a <c>Metaspace</c> pre-tokenizer
/// (spaces become `▁`, a `▁` is prepended when the piece doesn't already start with one, and with
/// <c>split</c> each `▁` starts a new segment). The GGUF factory has no such config and keeps the
/// SentencePiece default: whitespace runs collapse to one `▁` and one `▁` is always prepended.
/// Id parity for the XLM-R vocab is checked against llama.cpp's UGM tokenizer
/// (<c>XlmRobertaUnigramGoldenTests</c>).</para>
/// </summary>
public sealed class UnigramTokenizer
{
    private readonly string[] _pieces;
    private readonly float[] _scores;
    private readonly int _unkId;
    private readonly float _minNormalScore;
    private readonly TrieNode _root = new();
    private readonly List<Func<string, string>> _normalizers = [];
    private bool _legacyPreprocess = true;
    private bool _whitespaceSplit;
    private bool _metaspacePrepend = true;
    private bool _metaspaceSplit;

    private const char MetaspaceChar = '▁'; // ▁

    private UnigramTokenizer(string[] pieces, float[] scores, int unkId, bool[] isNormal)
    {
        _pieces = pieces;
        _scores = scores;
        _unkId = unkId;

        float minNormal = float.PositiveInfinity;
        for (int i = 0; i < pieces.Length; i++)
        {
            if (isNormal[i] && scores[i] < minNormal) minNormal = scores[i];
            if (pieces[i].Length > 0) Insert(pieces[i], i);
        }
        _minNormalScore = float.IsPositiveInfinity(minNormal) ? 0f : minNormal;
    }

    /// <summary>
    /// Loads a real Unigram model from a Hugging Face <c>tokenizer.json</c>'s
    /// <c>"model": {"type": "Unigram", "vocab": [[piece, score], ...], "unk_id": N}</c> section.
    /// </summary>
    public static UnigramTokenizer FromTokenizerJson(string tokenizerJsonPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(tokenizerJsonPath));
        var root = doc.RootElement;
        var model = root.GetProperty("model");
        // Older tokenizers releases (e.g. FacebookAI/xlm-roberta-base) omit model.type; a [piece, score]
        // vocab array is the Unigram shape.
        bool isUnigram = model.TryGetProperty("type", out var type)
            ? type.GetString() == "Unigram"
            : model.TryGetProperty("vocab", out var v) && v.ValueKind == JsonValueKind.Array;
        if (!isUnigram)
            throw new InvalidDataException($"'{tokenizerJsonPath}' model type is not Unigram.");

        var vocab = model.GetProperty("vocab");
        int n = vocab.GetArrayLength();
        var pieces = new string[n];
        var scores = new float[n];
        var isNormal = new bool[n];
        // Real convention (confirmed via HF's Unigram model.rs): entries beyond the raw
        // SentencePiece vocab that were added as "added_tokens" (control/user-defined, e.g. T5's
        // <extra_id_N>) carry score 0.0 and are NOT ordinary NORMAL segmentation candidates for
        // the purpose of computing the UNK fallback score -- approximated here by excluding any
        // piece bracketed by '<' and '>' from the NORMAL-minimum computation (real SentencePiece
        // distinguishes via an explicit per-piece `type` field on the raw .model, not present in
        // this simplified tokenizer.json vocab array -- see class doc's "Known gap" note; this
        // heuristic only affects the derived UNK score, not the trie/Viterbi correctness for any
        // input that GetActualSegmentation would produce via ordinary NORMAL pieces).
        int i = 0;
        foreach (var entry in vocab.EnumerateArray())
        {
            var arr = entry;
            pieces[i] = arr[0].GetString() ?? "";
            scores[i] = arr[1].GetSingle();
            isNormal[i] = !(pieces[i].Length >= 2 && pieces[i][0] == '<' && pieces[i][^1] == '>');
            i++;
        }
        int unkId = model.TryGetProperty("unk_id", out var u) ? u.GetInt32() : 0;

        var tok = new UnigramTokenizer(pieces, scores, unkId, isNormal);
        if (root.TryGetProperty("normalizer", out var norm) && norm.ValueKind == JsonValueKind.Object)
            tok.AddNormalizer(norm, tokenizerJsonPath);
        if (root.TryGetProperty("pre_tokenizer", out var pre) && pre.ValueKind == JsonValueKind.Object)
        {
            tok._legacyPreprocess = false;
            tok.AddPreTokenizer(pre, tokenizerJsonPath);
        }
        return tok;
    }

    private void AddNormalizer(JsonElement n, string path)
    {
        switch (n.GetProperty("type").GetString())
        {
            case "Sequence":
                foreach (var child in n.GetProperty("normalizers").EnumerateArray()) AddNormalizer(child, path);
                break;
            case "Precompiled":
                var b64 = n.TryGetProperty("precompiled_charsmap", out var pc) ? pc.GetString() : null;
                if (!string.IsNullOrEmpty(b64))
                {
                    var map = new PrecompiledCharsmap(Convert.FromBase64String(b64));
                    _normalizers.Add(map.Normalize);
                }
                break;
            case "Strip":
                bool left = n.TryGetProperty("strip_left", out var sl) && sl.GetBoolean();
                bool right = n.TryGetProperty("strip_right", out var sr) && sr.GetBoolean();
                _normalizers.Add(s => left && right ? s.Trim() : left ? s.TrimStart() : right ? s.TrimEnd() : s);
                break;
            case "Replace":
                var pattern = n.GetProperty("pattern");
                string content = n.GetProperty("content").GetString() ?? "";
                if (pattern.TryGetProperty("Regex", out var rx))
                {
                    var regex = new Regex(rx.GetString()!, RegexOptions.CultureInvariant);
                    _normalizers.Add(s => regex.Replace(s, content));
                }
                else
                {
                    string literal = pattern.GetProperty("String").GetString()!;
                    _normalizers.Add(s => s.Replace(literal, content, StringComparison.Ordinal));
                }
                break;
            default:
                throw new NotSupportedException($"'{path}': Unigram normalizer type '{n.GetProperty("type").GetString()}' is not supported.");
        }
    }

    private void AddPreTokenizer(JsonElement p, string path)
    {
        switch (p.GetProperty("type").GetString())
        {
            case "Sequence":
                foreach (var child in p.GetProperty("pretokenizers").EnumerateArray()) AddPreTokenizer(child, path);
                break;
            case "WhitespaceSplit":
                _whitespaceSplit = true;
                break;
            case "Metaspace":
                if (p.TryGetProperty("replacement", out var rep) && rep.GetString() != "▁")
                    throw new NotSupportedException($"'{path}': Metaspace replacement other than U+2581 is not supported.");
                // Legacy configs carry add_prefix_space (true == prepend_scheme "always"); newer ones
                // carry prepend_scheme. "first" only differs from "always" for pre-split inputs.
                if (p.TryGetProperty("prepend_scheme", out var ps))
                    _metaspacePrepend = ps.GetString() != "never";
                else if (p.TryGetProperty("add_prefix_space", out var aps))
                    _metaspacePrepend = aps.GetBoolean();
                _metaspaceSplit = !p.TryGetProperty("split", out var sp) || sp.GetBoolean();
                break;
            default:
                throw new NotSupportedException($"'{path}': Unigram pre-tokenizer type '{p.GetProperty("type").GetString()}' is not supported.");
        }
    }

    /// <summary>
    /// Loads a real Unigram model from GGUF's own vocab arrays (<c>tokenizer.ggml.tokens</c>,
    /// <c>tokenizer.ggml.scores</c>), used when <c>tokenizer.ggml.model=t5</c> (real llama.cpp's
    /// <c>LLAMA_VOCAB_TYPE_UGM</c> trigger). Unlike <see cref="FromTokenizerJson"/>'s bracket
    /// heuristic (no better data available in a plain HF <c>tokenizer.json</c>), GGUF actually
    /// carries the real per-piece type array (<c>tokenizer.ggml.token_type</c>, GGUF/llama.cpp's
    /// <c>llama_token_type</c> enum: 1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 5=UNUSED,
    /// 6=BYTE) -- used here directly for the NORMAL-vs-not distinction that feeds the UNK fallback
    /// score, when present, instead of guessing from bracket punctuation.
    /// </summary>
    public static UnigramTokenizer FromGgufVocab(string[] tokens, float[] scores, int unkId, int[]? tokenTypes,
        byte[]? precompiledCharsmap = null)
    {
        var isNormal = new bool[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
        {
            isNormal[i] = tokenTypes is not null
                ? tokenTypes[i] == 1 // llama_token_type.LLAMA_TOKEN_TYPE_NORMAL
                : !(tokens[i].Length >= 2 && tokens[i][0] == '<' && tokens[i][^1] == '>');
        }
        var tok = new UnigramTokenizer(tokens, scores, unkId, isNormal);
        if (precompiledCharsmap is { Length: > 0 })
            tok._normalizers.Add(new PrecompiledCharsmap(precompiledCharsmap).Normalize);
        return tok;
    }

    private void Insert(string piece, int id)
    {
        var bytes = Encoding.UTF8.GetBytes(piece);
        var node = _root;
        foreach (var b in bytes)
        {
            node.Children ??= new Dictionary<byte, TrieNode>();
            if (!node.Children.TryGetValue(b, out var next))
            {
                next = new TrieNode();
                node.Children[b] = next;
            }
            node = next;
        }
        node.PieceId = id;
    }

    /// <summary>
    /// Normalizers, then pre-tokenization into the segments Viterbi runs on independently (see the
    /// class doc). Each returned segment already carries its `▁` markers.
    /// </summary>
    private List<string> Preprocess(string text)
    {
        foreach (var normalize in _normalizers) text = normalize(text);
        if (_legacyPreprocess) return [LegacyMetaspace(text)];

        var segments = new List<string>();
        var words = _whitespaceSplit
            ? text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
            : text.Length > 0 ? [text] : [];
        foreach (var word in words)
        {
            string w = word.Replace(' ', MetaspaceChar);
            if (_metaspacePrepend && w[0] != MetaspaceChar) w = MetaspaceChar + w;
            if (!_metaspaceSplit) { segments.Add(w); continue; }
            // Split with MergedWithNext: every ▁ starts a new segment.
            int start = 0;
            for (int i = 1; i < w.Length; i++)
            {
                if (w[i] != MetaspaceChar) continue;
                segments.Add(w[start..i]);
                start = i;
            }
            segments.Add(w[start..]);
        }
        return segments;
    }

    /// <summary>SentencePiece default (no tokenizer.json config): collapse whitespace runs to a
    /// single `▁` and always start with exactly one `▁` (the "dummy prefix").</summary>
    private static string LegacyMetaspace(string normalized)
    {
        var sb = new StringBuilder(normalized.Length + 1);
        bool prevWasSpace = false;
        foreach (char c in normalized)
        {
            if (char.IsWhiteSpace(c))
            {
                if (!prevWasSpace) sb.Append(MetaspaceChar);
                prevWasSpace = true;
            }
            else
            {
                sb.Append(c);
                prevWasSpace = false;
            }
        }
        // Real "dummy prefix" (prepend_scheme="always"): always start with exactly one ▁, whether
        // or not the input itself began with whitespace (avoid a doubled ▁▁ when it did).
        if (sb.Length == 0 || sb[0] != MetaspaceChar) sb.Insert(0, MetaspaceChar);
        return sb.ToString();
    }

    /// <summary>Real Unigram Viterbi segmentation (no special tokens added -- see the class's golden test, which compares against the real reference's `add_special_tokens=False` output).</summary>
    public List<int> Encode(string text)
    {
        var ids = new List<int>();
        foreach (var segment in Preprocess(text)) EncodeSegment(Encoding.UTF8.GetBytes(segment), ids);
        return ids;
    }

    private void EncodeSegment(byte[] bytes, List<int> output)
    {
        int n = bytes.Length;

        var bestScore = new float[n + 1];
        var bestStart = new int[n + 1];
        var bestId = new int[n + 1];
        for (int p = 1; p <= n; p++) bestScore[p] = float.NegativeInfinity;

        for (int start = 0; start < n; start++)
        {
            if (float.IsNegativeInfinity(bestScore[start])) continue;
            float baseScore = bestScore[start];

            // Common-prefix search via the byte trie: walk as far as pieces match, relaxing at every terminal node.
            var node = _root;
            bool hasSingleByteMatch = false;
            for (int end = start; end < n; end++)
            {
                if (node.Children is null || !node.Children.TryGetValue(bytes[end], out var next)) break;
                node = next;
                if (node.PieceId >= 0)
                {
                    int len = end - start + 1;
                    if (len == 1) hasSingleByteMatch = true;
                    float candidate = baseScore + _scores[node.PieceId];
                    int pos = end + 1;
                    if (candidate > bestScore[pos])
                    {
                        bestScore[pos] = candidate;
                        bestStart[pos] = start;
                        bestId[pos] = node.PieceId;
                    }
                }
            }

            if (!hasSingleByteMatch)
            {
                // UNK edge spans exactly one Unicode scalar (1-4 UTF-8 bytes), matching real PopulateNodes.
                int scalarLen = Utf8ScalarByteLength(bytes, start);
                int pos2 = start + scalarLen;
                float unkScore = baseScore + (_minNormalScore - 10.0f);
                if (unkScore > bestScore[pos2])
                {
                    bestScore[pos2] = unkScore;
                    bestStart[pos2] = start;
                    bestId[pos2] = -1; // UNK edge, mapped to _unkId on backtrack
                }
            }
        }

        int first = output.Count;
        int cur = n;
        while (cur > 0)
        {
            output.Add(bestId[cur]);
            cur = bestStart[cur];
        }
        output.Reverse(first, output.Count - first);

        // SentencePiece (and llama.cpp UGM / HF Unigram) merge a run of consecutive unknown pieces
        // into one UNK token, e.g. two unknown emoji in a row.
        int w = first;
        for (int r = first; r < output.Count; r++)
        {
            if (output[r] < 0 && w > first && output[w - 1] < 0) continue;
            output[w++] = output[r];
        }
        output.RemoveRange(w, output.Count - w);
        for (int i = first; i < output.Count; i++)
            if (output[i] < 0) output[i] = _unkId;
    }

    private static int Utf8ScalarByteLength(byte[] bytes, int start)
    {
        byte b = bytes[start];
        int len = b < 0x80 ? 1 : b >> 5 == 0b110 ? 2 : b >> 4 == 0b1110 ? 3 : b >> 3 == 0b11110 ? 4 : 1;
        return Math.Min(len, bytes.Length - start);
    }

    public IReadOnlyList<string> Pieces => _pieces;

    private sealed class TrieNode
    {
        public Dictionary<byte, TrieNode>? Children;
        public int PieceId = -1;
    }
}
