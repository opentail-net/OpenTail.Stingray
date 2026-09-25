namespace OpenTail.Stingray.Core;

/// <summary>Token ids plus per-token segment (token type) ids for an encoder model.</summary>
public readonly record struct EncodedInput(int[] Ids, int[] TypeIds);

/// <summary>
/// Tokenizer front end for encoder models (BERT/MiniLM/BGE/Nomic/MPNet/ELECTRA WordPiece, XLM-R-family
/// Unigram), built from a HF <c>tokenizer.json</c>. Segmentation is delegated to
/// <see cref="BertWordPieceTokenizer"/> or <see cref="UnigramTokenizer"/>; this class adds the file's
/// <c>post_processor</c>:
/// <list type="bullet">
/// <item><c>TemplateProcessing</c>: the <c>single</c>/<c>pair</c> templates as written, e.g. BERT
/// <c>[CLS] A [SEP] B [SEP]</c> with type ids 0/1.</item>
/// <item><c>RobertaProcessing</c>/<c>BertProcessing</c>: <c>cls A sep (sep) B sep</c>. Type ids are
/// all 0 for RoBERTa-style models (none of them has a second token type) and 0/1 for BERT.</item>
/// </list>
/// Truncation to <c>maxLength</c> counts the special tokens and, for pairs, removes tokens from the
/// longer sequence first (HF's default <c>longest_first</c> strategy).
/// </summary>
public sealed class EncoderTokenizer
{
    private readonly Func<string, List<int>> _segment;
    private readonly TemplatePiece[] _single;
    private readonly TemplatePiece[] _pair;

    /// <summary>Id used to pad batches (the model's <c>[PAD]</c>/<c>&lt;pad&gt;</c>).</summary>
    public int PadTokenId { get; }

    private readonly record struct TemplatePiece(int[]? SpecialIds, bool IsSequenceB, int TypeId);

    private EncoderTokenizer(Func<string, List<int>> segment, TemplatePiece[] single, TemplatePiece[] pair, int padId)
    {
        _segment = segment;
        _single = single;
        _pair = pair;
        PadTokenId = padId;
    }

    public static EncoderTokenizer FromTokenizerJson(string tokenizerJsonPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(tokenizerJsonPath));
        var root = doc.RootElement;
        var model = root.GetProperty("model");
        bool unigram = model.TryGetProperty("type", out var type)
            ? type.GetString() == "Unigram"
            : model.GetProperty("vocab").ValueKind == JsonValueKind.Array;

        Func<string, List<int>> segment;
        Func<string, int?> tokenToId;
        if (unigram)
        {
            var u = UnigramTokenizer.FromTokenizerJson(tokenizerJsonPath);
            segment = u.Encode;
            tokenToId = u.TokenToId;
        }
        else
        {
            var w = BertWordPieceTokenizer.FromTokenizerJson(tokenizerJsonPath);
            segment = w.EncodeIds;
            tokenToId = w.TokenToId;
        }

        int padId = root.TryGetProperty("padding", out var pad) && pad.ValueKind == JsonValueKind.Object
            && pad.TryGetProperty("pad_id", out var pid)
            ? pid.GetInt32()
            : tokenToId("[PAD]") ?? tokenToId("<pad>") ?? 0;

        if (!root.TryGetProperty("post_processor", out var pp) || pp.ValueKind != JsonValueKind.Object)
            return new EncoderTokenizer(segment, [new(null, false, 0)], [new(null, false, 0), new(null, true, 1)], padId);

        string ppType = pp.GetProperty("type").GetString() ?? "";
        switch (ppType)
        {
            case "TemplateProcessing":
            {
                var specials = new Dictionary<string, int[]>(StringComparer.Ordinal);
                if (pp.TryGetProperty("special_tokens", out var st))
                    foreach (var s in st.EnumerateObject())
                        specials[s.Name] = s.Value.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                TemplatePiece[] Parse(JsonElement arr) => arr.EnumerateArray().Select(e =>
                {
                    if (e.TryGetProperty("SpecialToken", out var sp))
                    {
                        string id = sp.GetProperty("id").GetString()!;
                        int[] ids = specials.TryGetValue(id, out var known) ? known
                            : tokenToId(id) is int t ? [t]
                            : throw new InvalidDataException($"'{tokenizerJsonPath}': template special token '{id}' has no id.");
                        return new TemplatePiece(ids, false, sp.GetProperty("type_id").GetInt32());
                    }
                    var seq = e.GetProperty("Sequence");
                    return new TemplatePiece(null, seq.GetProperty("id").GetString() == "B", seq.GetProperty("type_id").GetInt32());
                }).ToArray();
                return new EncoderTokenizer(segment, Parse(pp.GetProperty("single")), Parse(pp.GetProperty("pair")), padId);
            }
            case "RobertaProcessing":
            case "BertProcessing":
            {
                int[] cls = [pp.GetProperty("cls")[1].GetInt32()];
                int[] sep = [pp.GetProperty("sep")[1].GetInt32()];
                bool roberta = ppType == "RobertaProcessing";
                int typeB = roberta ? 0 : 1;
                TemplatePiece[] single = [new(cls, false, 0), new(null, false, 0), new(sep, false, 0)];
                TemplatePiece[] pair = roberta
                    ? [new(cls, false, 0), new(null, false, 0), new(sep, false, 0), new(sep, false, 0), new(null, true, 0), new(sep, false, 0)]
                    : [new(cls, false, 0), new(null, false, 0), new(sep, false, 0), new(null, true, typeB), new(sep, false, typeB)];
                return new EncoderTokenizer(segment, single, pair, padId);
            }
            default:
                throw new NotSupportedException($"'{tokenizerJsonPath}': post_processor '{ppType}' is not supported.");
        }
    }

    /// <summary>Single sequence with its special tokens, truncated to <paramref name="maxLength"/> ids.</summary>
    public EncodedInput Encode(string text, int maxLength = int.MaxValue)
    {
        var a = _segment(text);
        Truncate(a, null, maxLength - SpecialCount(_single));
        return Build(_single, a, null);
    }

    /// <summary>Sequence pair (query/document for cross-encoders), truncated longest-first.</summary>
    public EncodedInput EncodePair(string first, string second, int maxLength = int.MaxValue)
    {
        var a = _segment(first);
        var b = _segment(second);
        Truncate(a, b, maxLength - SpecialCount(_pair));
        return Build(_pair, a, b);
    }

    private static int SpecialCount(TemplatePiece[] template) => template.Sum(p => p.SpecialIds?.Length ?? 0);

    // HF tokenizers TruncationStrategy::LongestFirst: the shorter sequence keeps min(len, budget - other)
    // and, when both exceed half, the shorter gets floor(budget/2) and the longer the remainder.
    private static void Truncate(List<int> a, List<int>? b, int budget)
    {
        if (budget < 0) budget = 0;
        if (b is null)
        {
            if (a.Count > budget) a.RemoveRange(budget, a.Count - budget);
            return;
        }
        if (a.Count + b.Count <= budget) return;
        bool swap = a.Count > b.Count;
        int n1 = swap ? b.Count : a.Count, n2;
        n2 = n1 > budget ? n1 : Math.Max(n1, budget - n1);
        if (n1 + n2 > budget) { n1 = budget / 2; n2 = n1 + budget % 2; }
        if (swap) (n1, n2) = (n2, n1);
        if (a.Count > n1) a.RemoveRange(n1, a.Count - n1);
        if (b.Count > n2) b.RemoveRange(n2, b.Count - n2);
    }

    private static EncodedInput Build(TemplatePiece[] template, List<int> a, List<int>? b)
    {
        var ids = new List<int>(a.Count + (b?.Count ?? 0) + 4);
        var types = new List<int>(ids.Capacity);
        foreach (var piece in template)
        {
            IReadOnlyList<int> part = piece.SpecialIds is { } special ? special : piece.IsSequenceB ? b! : a;
            foreach (int id in part)
            {
                ids.Add(id);
                types.Add(piece.TypeId);
            }
        }
        return new EncodedInput([.. ids], [.. types]);
    }
}
