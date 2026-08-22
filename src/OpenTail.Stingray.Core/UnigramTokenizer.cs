using System.Text;
using System.Text.Json;

namespace OpenTail.Stingray.Core;

/// <summary>
/// Real SentencePiece/Hugging Face Unigram tokenizer -- inference-time segmentation ONLY (no
/// training/EM). Built to unblock Parler-TTS's T5 encoder, whose real <c>tokenizer.json</c>
/// declares <c>"model": {"type": "Unigram", ...}</c> -- a genuinely different segmentation
/// algorithm from this codebase's existing BPE tokenizer (<see cref="GgufTokenizer"/>/
/// <see cref="HuggingFaceTokenizerSource"/>), not something that can be reused by swapping the
/// vocabulary.
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
/// <para><b>Real preprocessing, from the same `tokenizer.json`'s `normalizer`/`pre_tokenizer`
/// sections (confirmed via direct inspection of Parler-TTS's real
/// `scratch-llamacpp-ref/parler-tokenizer/tokenizer.json`)</b>: a `Sequence` normalizer whose
/// first stage is `Precompiled` (SentencePiece's custom `nmt_nfkc`-family normalization, compiled
/// into a binary `precompiled_charsmap` darts-trie blob embedded in the model) -- <b>NOT yet
/// implemented here, a real and precisely documented gap, see the class-level "Known gap" note
/// below</b> -- followed by a `Replace` stage, then a `Metaspace` pre-tokenizer (`▁` replacement,
/// `prepend_scheme="always"`, `split=true`): every run of whitespace collapses and is represented
/// by `▁` (U+2581), and a `▁` is always prepended even at the very start of the input (SentencePiece's
/// real "dummy prefix" behavior).</para>
///
/// <para><b>Known gap, not worked around</b>: the `precompiled_charsmap` binary format (a compiled
/// darts double-array trie mapping arbitrary input substrings to Unicode-normalized replacement
/// strings -- covers things like fullwidth-character folding and various dash/space-equivalent
/// normalization beyond plain NFKC) is NOT implemented. This class instead applies plain Unicode
/// NFKC normalization as a stand-in. For plain-ASCII input (confirmed against a real golden
/// oracle, see <c>UnigramTokenizerTests</c>) this produces IDENTICAL output to the real
/// `precompiled_charsmap` pipeline, because the charsmap's real behavior only diverges from plain
/// NFKC on non-ASCII/exotic input (the specific substitution rules it encodes). Text containing
/// such characters may therefore segment differently from the real reference until this gap is
/// closed -- documented precisely per this project's blocker-honesty discipline, not silently
/// approximated.</para>
/// </summary>
public sealed class UnigramTokenizer
{
    private readonly string[] _pieces;
    private readonly float[] _scores;
    private readonly int _unkId;
    private readonly float _minNormalScore;
    private readonly TrieNode _root = new();

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
        if (model.GetProperty("type").GetString() != "Unigram")
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

        return new UnigramTokenizer(pieces, scores, unkId, isNormal);
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
    /// Real preprocessing: NFKC (stand-in for the real `precompiled_charsmap`, see class doc's
    /// "Known gap") -&gt; Metaspace (collapse whitespace runs to a single `▁`, always prepend one
    /// at the start -- SentencePiece's real "dummy prefix" behavior).
    /// </summary>
    private static string Preprocess(string text)
    {
        string normalized = text.Normalize(NormalizationForm.FormKC);
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
        string prepped = Preprocess(text);
        var bytes = Encoding.UTF8.GetBytes(prepped);
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
                    bestId[pos2] = _unkId;
                }
            }
        }

        var ids = new List<int>();
        int cur = n;
        while (cur > 0)
        {
            ids.Add(bestId[cur]);
            cur = bestStart[cur];
        }
        ids.Reverse();
        return ids;
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
