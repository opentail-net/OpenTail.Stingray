using System.Text.Json;
using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 tokenizer, a plain char-level BPE ported directly from the real HuggingFace
/// `tokenizers`-library `vocab.json` (`TTS/tts/layers/xtts/tokenizer.py`'s `VoiceBpeTokenizer`,
/// which wraps `tokenizers.Tokenizer.from_file`). Confirmed real config from `vocab.json`:
/// `model.type="BPE"`, no `continuing_subword_prefix`/`end_of_word_suffix` (plain char-level, no
/// word-boundary markers), `pre_tokenizer.type="Whitespace"` (regex `\w+|[^\w\s]+` -- splits on
/// alphanumeric vs. punctuation runs, silently DROPS raw whitespace), `normalizer=null`,
/// `fuse_unk=false`. Real `added_tokens` (`[STOP]`, `[UNK]`, `[SPACE]`, language tags like `[en]`,
/// `[START]`) are matched as literal substrings BEFORE pre-tokenization/BPE (HF's
/// `AddedVocabulary`, longest-match-first scan) -- this is how a literal `"[SPACE]"` substring
/// (which `VoiceBpeTokenizer.encode` substitutes for every real space before calling the
/// tokenizer, since the `Whitespace` pre-tokenizer alone drops raw whitespace) becomes a single
/// token instead of three (`[`,`SPACE`,`]`).
///
/// <para>This class ports ONLY the real `tokenizers.Tokenizer.encode` algorithm itself -- verified
/// bit-exact against the real Python `Tokenizer.from_file(...).encode(text).ids` for representative
/// English/French sentences (a hand-written Python reimplementation of this exact algorithm matched
/// real `tokenizers` output exactly before this port was written). It does NOT implement the
/// separate, large, per-language `multilingual_cleaners`/number-expansion text-normalization pass
/// (`VoiceBpeTokenizer.preprocess_text`) that the real pipeline runs BEFORE tokenization --
/// callers must pass already-normalized (lowercase, spelled-out-numbers) text, matching this
/// port's established practice of using the real Python tokenizer's own output ids as the
/// correctness bar for the tokenizer stage in isolation.</para>
/// </summary>
public sealed class XttsBpeTokenizer
{
    private static readonly Regex WhitespacePreTokenizer = new(@"\w+|[^\w\s]+", RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<(string, string), int> _mergeRank;
    private readonly string[] _addedTokensByLengthDesc;
    private readonly Dictionary<string, int> _addedTokenIds;
    private readonly int _unkId;

    public XttsBpeTokenizer(string vocabJsonPath)
    {
        using var stream = File.OpenRead(vocabJsonPath);
        using var doc = JsonDocument.Parse(stream);
        var root = doc.RootElement;

        var model = root.GetProperty("model");
        _vocab = new Dictionary<string, int>();
        foreach (var prop in model.GetProperty("vocab").EnumerateObject())
            _vocab[prop.Name] = prop.Value.GetInt32();

        _mergeRank = new Dictionary<(string, string), int>();
        int rank = 0;
        foreach (var m in model.GetProperty("merges").EnumerateArray())
        {
            string pair = m.GetString()!;
            int sp = pair.IndexOf(' ');
            _mergeRank[(pair[..sp], pair[(sp + 1)..])] = rank++;
        }

        _unkId = _vocab.TryGetValue(model.GetProperty("unk_token").GetString() ?? "[UNK]", out int u) ? u : 0;

        _addedTokenIds = new Dictionary<string, int>();
        if (root.TryGetProperty("added_tokens", out var addedTokens))
        {
            foreach (var t in addedTokens.EnumerateArray())
                _addedTokenIds[t.GetProperty("content").GetString()!] = t.GetProperty("id").GetInt32();
        }
        _addedTokensByLengthDesc = [.. _addedTokenIds.Keys.OrderByDescending(k => k.Length)];
    }

    /// <summary>Real `Tokenizer.encode(text).ids`: scans for added-token literal matches first
    /// (longest match wins), then runs the `Whitespace` pre-tokenizer + char-level BPE merge over
    /// the plain-text spans between them. Text must already be real-pipeline-normalized (see
    /// class doc) -- this method does not itself lowercase or clean the input.</summary>
    public List<int> Encode(string text)
    {
        var ids = new List<int>();
        int i = 0, n = text.Length;
        while (i < n)
        {
            string? matched = MatchAddedTokenAt(text, i);
            if (matched is not null)
            {
                ids.Add(_addedTokenIds[matched]);
                i += matched.Length;
                continue;
            }

            int next = n;
            for (int j = i; j < n; j++)
            {
                if (MatchAddedTokenAt(text, j) is not null) { next = j; break; }
            }

            string span = text[i..next];
            foreach (Match m in WhitespacePreTokenizer.Matches(span))
                foreach (string piece in BpeWord(m.Value))
                    ids.Add(_vocab.TryGetValue(piece, out int id) ? id : _unkId);

            i = next;
        }
        return ids;
    }

    private string? MatchAddedTokenAt(string text, int pos)
    {
        foreach (string tok in _addedTokensByLengthDesc)
        {
            if (pos + tok.Length <= text.Length && string.CompareOrdinal(text, pos, tok, 0, tok.Length) == 0)
                return tok;
        }
        return null;
    }

    private List<string> BpeWord(string word)
    {
        var symbols = new List<string>(word.Length);
        foreach (var rune in word.EnumerateRunes()) symbols.Add(rune.ToString());

        while (symbols.Count > 1)
        {
            int bestIndex = -1, bestRank = int.MaxValue;
            for (int i = 0; i < symbols.Count - 1; i++)
            {
                if (_mergeRank.TryGetValue((symbols[i], symbols[i + 1]), out int r) && r < bestRank)
                {
                    bestRank = r;
                    bestIndex = i;
                }
            }
            if (bestIndex < 0) break;
            symbols[bestIndex] += symbols[bestIndex + 1];
            symbols.RemoveAt(bestIndex + 1);
        }
        return symbols;
    }
}
