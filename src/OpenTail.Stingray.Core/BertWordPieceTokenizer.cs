using System.Globalization;
using System.Text;

namespace OpenTail.Stingray.Core;

/// <summary>
/// Real BERT-family WordPiece tokenizer — a faithful port of HuggingFace `transformers`'
/// <c>BasicTokenizer</c> + <c>WordpieceTokenizer</c> (the real algorithm every BERT/MiniLM/BGE
/// checkpoint's <c>tokenizer_config.json</c> declares as <c>"tokenizer_class": "BertTokenizer"</c>),
/// not an approximation. Loads a real <c>vocab.txt</c> (one token per line, line number = token id,
/// the standard BERT vocab file format) and reproduces: control-character cleaning, optional CJK
/// character spacing, lowercasing + accent stripping (both gated on <c>do_lower_case</c>, matching
/// the real `strip_accents = do_lower_case` default when `strip_accents` is unset), punctuation
/// splitting, and greedy longest-match-first WordPiece subword segmentation with a "##" continuation
/// prefix. Replaces the previous char-per-token placeholder in `EmbedCommand`'s ONNX embedding path.
/// </summary>
public sealed class BertWordPieceTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly bool _doLowerCase;
    private readonly bool _stripAccents;
    private readonly bool _tokenizeChineseChars;
    private readonly bool _cleanText;
    private readonly string _unkToken;
    private readonly string _continuingPrefix;
    private readonly int _maxInputCharsPerWord;

    public int ClsTokenId { get; }
    public int SepTokenId { get; }
    public int UnkTokenId { get; }
    public int PadTokenId { get; }

    private BertWordPieceTokenizer(Dictionary<string, int> vocab, bool doLowerCase, bool tokenizeChineseChars,
        bool? stripAccents = null, bool cleanText = true, string unkToken = "[UNK]", string continuingPrefix = "##",
        int maxInputCharsPerWord = 100)
    {
        _vocab = vocab;
        _doLowerCase = doLowerCase;
        // HF BertNormalizer / BasicTokenizer: strip_accents unset (null) follows lowercase.
        _stripAccents = stripAccents ?? doLowerCase;
        _tokenizeChineseChars = tokenizeChineseChars;
        _cleanText = cleanText;
        _unkToken = unkToken;
        _continuingPrefix = continuingPrefix;
        _maxInputCharsPerWord = maxInputCharsPerWord;

        ClsTokenId = _vocab.TryGetValue("[CLS]", out var cls) ? cls : 0;
        SepTokenId = _vocab.TryGetValue("[SEP]", out var sep) ? sep : 0;
        UnkTokenId = _vocab.TryGetValue(unkToken, out var unk) ? unk : 0;
        PadTokenId = _vocab.TryGetValue("[PAD]", out var pad) ? pad : 0;
    }

    /// <summary>
    /// Loads a real BERT WordPiece vocab file (one token per line, line index = token id).
    /// <paramref name="doLowerCase"/>/<paramref name="tokenizeChineseChars"/> should come from the
    /// checkpoint's own real <c>tokenizer_config.json</c> when available; both default to the
    /// standard BERT-uncased convention (true) otherwise, matching every checkpoint actually
    /// wired against this loader so far (all-MiniLM-L6-v2, BGE small/base/large — all
    /// `do_lower_case: true`, `tokenize_chinese_chars: true`).
    /// </summary>
    public static BertWordPieceTokenizer LoadVocabFile(string vocabPath, bool doLowerCase = true, bool tokenizeChineseChars = true)
    {
        var lines = File.ReadAllLines(vocabPath);
        var vocab = new Dictionary<string, int>(lines.Length, StringComparer.Ordinal);
        for (int i = 0; i < lines.Length; i++)
        {
            // Real vocab.txt lines are the token verbatim, including a trailing newline that
            // ReadAllLines already strips -- no further trimming (a real token can legitimately
            // be a single space or other whitespace-adjacent piece for some checkpoints).
            vocab[lines[i]] = i;
        }
        return new BertWordPieceTokenizer(vocab, doLowerCase, tokenizeChineseChars);
    }

    /// <summary>
    /// Loads a HF <c>tokenizer.json</c> whose model is <c>WordPiece</c>: the vocab dict, <c>unk_token</c>,
    /// <c>continuing_subword_prefix</c> and <c>max_input_chars_per_word</c>, plus the <c>BertNormalizer</c>
    /// flags (<c>lowercase</c>, <c>strip_accents</c>, <c>handle_chinese_chars</c>, <c>clean_text</c>).
    /// Special-token templates are applied by <see cref="EncoderTokenizer"/>, not here.
    /// </summary>
    public static BertWordPieceTokenizer FromTokenizerJson(string tokenizerJsonPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(tokenizerJsonPath));
        var root = doc.RootElement;
        var model = root.GetProperty("model");
        bool isWordPiece = model.TryGetProperty("type", out var type)
            ? type.GetString() == "WordPiece"
            : model.TryGetProperty("continuing_subword_prefix", out _) && model.TryGetProperty("max_input_chars_per_word", out _);
        if (!isWordPiece)
            throw new InvalidDataException($"'{tokenizerJsonPath}' model type is not WordPiece.");

        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var entry in model.GetProperty("vocab").EnumerateObject()) vocab[entry.Name] = entry.Value.GetInt32();

        bool lower = true, chinese = true, clean = true;
        bool? strip = null;
        if (root.TryGetProperty("normalizer", out var n) && n.ValueKind == JsonValueKind.Object)
        {
            if (n.GetProperty("type").GetString() != "BertNormalizer")
                throw new NotSupportedException($"'{tokenizerJsonPath}': WordPiece normalizer '{n.GetProperty("type").GetString()}' is not supported.");
            lower = !n.TryGetProperty("lowercase", out var lc) || lc.GetBoolean();
            chinese = !n.TryGetProperty("handle_chinese_chars", out var hc) || hc.GetBoolean();
            clean = !n.TryGetProperty("clean_text", out var ct) || ct.GetBoolean();
            if (n.TryGetProperty("strip_accents", out var sa) && sa.ValueKind != JsonValueKind.Null) strip = sa.GetBoolean();
        }
        string unk = model.TryGetProperty("unk_token", out var u) && u.ValueKind == JsonValueKind.String ? u.GetString()! : "[UNK]";
        string prefix = model.TryGetProperty("continuing_subword_prefix", out var p) && p.ValueKind == JsonValueKind.String ? p.GetString()! : "##";
        int maxChars = model.TryGetProperty("max_input_chars_per_word", out var mc) && mc.ValueKind == JsonValueKind.Number ? mc.GetInt32() : 100;
        return new BertWordPieceTokenizer(vocab, lower, chinese, strip, clean, unk, prefix, maxChars);
    }

    /// <summary>Vocab id of <paramref name="token"/>, or null when it is not in the vocab.</summary>
    public int? TokenToId(string token) => _vocab.TryGetValue(token, out var id) ? id : null;

    /// <summary>Ids for <paramref name="text"/> with no special tokens.</summary>
    public List<int> EncodeIds(string text)
    {
        var tokens = Tokenize(text);
        var ids = new List<int>(tokens.Count);
        foreach (var t in tokens) ids.Add(_vocab.TryGetValue(t, out var id) ? id : UnkTokenId);
        return ids;
    }

    /// <summary>Basic tokenization: clean, optional CJK spacing, lowercase+strip-accents, whitespace
    /// split, then punctuation split -- matches the real `BasicTokenizer.tokenize`.</summary>
    private List<string> BasicTokenize(string text)
    {
        if (_cleanText) text = CleanText(text);
        if (_tokenizeChineseChars)
            text = TokenizeChineseChars(text);

        var origTokens = WhitespaceTokenize(text);
        var splitTokens = new List<string>();
        foreach (var token in origTokens)
        {
            string t = token;
            if (_doLowerCase) t = t.ToLowerInvariant();
            if (_stripAccents) t = StripAccents(t);
            splitTokens.AddRange(SplitOnPunctuation(t));
        }
        return splitTokens;
    }

    /// <summary>Removes control characters (per the real reference: ord==0, ord==0xFFFD, or a
    /// Unicode "control"/"format" category that isn't whitespace) and normalizes every whitespace
    /// character to a single ASCII space.</summary>
    private static string CleanText(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (var c in text)
        {
            int cp = c;
            if (cp == 0 || cp == 0xFFFD || IsControl(c))
                continue;
            if (IsWhitespace(c))
                sb.Append(' ');
            else
                sb.Append(c);
        }
        return sb.ToString();
    }

    private static bool IsWhitespace(char c)
    {
        if (c is ' ' or '\t' or '\n' or '\r')
            return true;
        var cat = CharUnicodeInfo.GetUnicodeCategory(c);
        return cat == UnicodeCategory.SpaceSeparator;
    }

    private static bool IsControl(char c)
    {
        if (c is '\t' or '\n' or '\r')
            return false;
        var cat = CharUnicodeInfo.GetUnicodeCategory(c);
        return cat is UnicodeCategory.Control or UnicodeCategory.Format;
    }

    private static string TokenizeChineseChars(string text)
    {
        var sb = new StringBuilder(text.Length * 2);
        var enumerator = StringInfo.GetTextElementEnumerator(text);
        // Iterate by UTF-32 code point (CJK Extension B+ chars are surrogate pairs).
        for (int i = 0; i < text.Length; i++)
        {
            int cp = char.ConvertToUtf32(text, i);
            bool isSurrogatePair = char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]);
            if (IsChineseChar(cp))
            {
                sb.Append(' ');
                sb.Append(isSurrogatePair ? text.Substring(i, 2) : text[i].ToString());
                sb.Append(' ');
            }
            else
            {
                sb.Append(text[i]);
            }
            if (isSurrogatePair) i++;
        }
        return sb.ToString();
    }

    /// <summary>Real CJK ranges from `BasicTokenizer._is_chinese_char` (CJK Unified Ideographs,
    /// Extension A, Extensions B-F, Compatibility Ideographs, Compatibility Supplement).</summary>
    private static bool IsChineseChar(int cp) =>
        (cp >= 0x4E00 && cp <= 0x9FFF) ||
        (cp >= 0x3400 && cp <= 0x4DBF) ||
        (cp >= 0x20000 && cp <= 0x2A6DF) ||
        (cp >= 0x2A700 && cp <= 0x2B73F) ||
        (cp >= 0x2B740 && cp <= 0x2B81F) ||
        (cp >= 0x2B820 && cp <= 0x2CEAF) ||
        (cp >= 0xF900 && cp <= 0xFAFF) ||
        (cp >= 0x2F800 && cp <= 0x2FA1F);

    // NFD + drop Mn via a culture-free table: string.Normalize(FormD) does not decompose non-ASCII
    // under InvariantGlobalization (see UnicodeAccentStrip).
    private static string StripAccents(string text) => UnicodeAccentStrip.StripAccents(text);

    private static List<string> WhitespaceTokenize(string text) =>
        text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).ToList();

    /// <summary>Real `_is_punctuation`: explicit ASCII punctuation ranges plus any Unicode
    /// category starting with "P" (connector/dash/open/close/initial/final/other punctuation).</summary>
    private static bool IsPunctuation(char c)
    {
        int cp = c;
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126))
            return true;
        var cat = CharUnicodeInfo.GetUnicodeCategory(c);
        return cat is UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DashPunctuation
            or UnicodeCategory.OpenPunctuation or UnicodeCategory.ClosePunctuation
            or UnicodeCategory.InitialQuotePunctuation or UnicodeCategory.FinalQuotePunctuation
            or UnicodeCategory.OtherPunctuation;
    }

    private static List<string> SplitOnPunctuation(string text)
    {
        var output = new List<string>();
        var current = new StringBuilder();
        foreach (var c in text)
        {
            if (IsPunctuation(c))
            {
                if (current.Length > 0) { output.Add(current.ToString()); current.Clear(); }
                output.Add(c.ToString());
            }
            else
            {
                current.Append(c);
            }
        }
        if (current.Length > 0) output.Add(current.ToString());
        return output;
    }

    /// <summary>Real greedy longest-match-first WordPiece segmentation of one basic-tokenized
    /// piece, matching `WordpieceTokenizer.tokenize` exactly (including the "##" continuation
    /// prefix and the whole-word-becomes-[UNK] fallback when no valid split exists).</summary>
    private List<string> WordpieceTokenize(string token)
    {
        if (token.Length > _maxInputCharsPerWord)
            return [_unkToken];

        var subTokens = new List<string>();
        int start = 0;
        while (start < token.Length)
        {
            int end = token.Length;
            string? curSubstr = null;
            while (start < end)
            {
                string substr = token[start..end];
                if (start > 0) substr = _continuingPrefix + substr;
                if (_vocab.ContainsKey(substr))
                {
                    curSubstr = substr;
                    break;
                }
                end--;
            }
            if (curSubstr is null)
                return [_unkToken];
            subTokens.Add(curSubstr);
            start = end;
        }
        return subTokens;
    }

    /// <summary>Full tokenize: basic tokenization then WordPiece on each resulting piece.</summary>
    public List<string> Tokenize(string text)
    {
        var output = new List<string>();
        foreach (var basicToken in BasicTokenize(text))
            output.AddRange(WordpieceTokenize(basicToken));
        return output;
    }

    /// <summary>Encodes a single sequence with [CLS]/[SEP] special tokens, matching real
    /// single-sequence `BertTokenizer.encode`/`__call__` output (no `[SEP]`-pair handling needed
    /// for the embedding use case this loader serves).</summary>
    public long[] Encode(string text)
    {
        var tokens = Tokenize(text);
        var ids = new long[tokens.Count + 2];
        ids[0] = ClsTokenId;
        for (int i = 0; i < tokens.Count; i++)
            ids[i + 1] = _vocab.TryGetValue(tokens[i], out var id) ? id : UnkTokenId;
        ids[^1] = SepTokenId;
        return ids;
    }
}
