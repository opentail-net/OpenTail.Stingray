
namespace OpenTail.Stingray.Diffusion.TextEncoders;

/// <summary>
/// SentencePiece-compatible T5 Unigram tokenizer.
/// Loads vocabulary from a tokenizer.json (HuggingFace format) distributed
/// alongside t5xxl_fp16.safetensors as tokenizer.json / spiece.model.
///
/// BOS = 0, EOS = 1, UNK = 2, PAD = 0.
/// Maximum output length is configurable via <see cref="MaxLen"/>/`FromFile`'s `maxLen` param --
/// 77 for FLUX's T5, 226 for Wan's UMT5 (both real conventions from their own real pipelines).
///
/// <para><b>Real Viterbi-optimal segmentation</b> (fixed 2026-09-01, replacing a greedy
/// longest-match approximation found to diverge from HuggingFace's real `T5TokenizerFast` on real
/// prompts -- e.g. "fox" split as a single wrong greedy token instead of the real two-token split;
/// see `LtxT5EncoderGoldenParityTests`' now-passing exact-match test). Real SentencePiece Unigram
/// tokenization is NOT greedy: each candidate piece carries a log-probability `score` (the
/// vocab's `[token, score]` pairs), and the correct segmentation is the one maximizing the SUM of
/// scores across the whole string -- found via a standard Viterbi dynamic program over the
/// "which vocab entries end at position i" lattice, not by repeatedly taking the longest match at
/// the current position (a genuinely different, and sometimes wrong, greedy strategy).</para>
/// </summary>
public sealed class T5Tokenizer
{
    private readonly string[] _idToToken;
    private readonly Dictionary<string, (int Id, double Score)> _tokenToEntry;
    private readonly int _maxTokenLen;
    private const int EosToken = 1;
    private const int UnkToken = 2;
    private const double UnkScore = -100.0; // real sentencepiece: heavily penalized single-char fallback

    /// <summary>Max output length (including the trailing EOS token). FLUX's own T5 uses 77;
    /// Wan's UMT5 encoder uses 226 (real `max_sequence_length` in `WanPipeline._get_t5_prompt_embeds`).</summary>
    public int MaxLen { get; init; } = 77;

    private T5Tokenizer(string[] idToToken, Dictionary<string, (int, double)> tokenToEntry, int maxTokenLen)
    {
        _idToToken = idToToken;
        _tokenToEntry = tokenToEntry;
        _maxTokenLen = maxTokenLen;
    }

    /// <summary>Load from a HuggingFace tokenizer.json file. <paramref name="maxLen"/> defaults to FLUX's 77; pass 226 for Wan's UMT5 encoder.</summary>
    public static T5Tokenizer FromFile(string path, int maxLen = 77)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var model   = doc.RootElement.GetProperty("model");
        var vocabEl = model.GetProperty("vocab");

        // vocab: array of [token_string, score] pairs (score = real Unigram log-probability)
        var tokens = new List<string> { "<pad>", "</s>", "<unk>" };
        var tokenToEntry = new Dictionary<string, (int, double)>(StringComparer.Ordinal)
        {
            ["<pad>"]  = (0, 0.0),
            ["</s>"]   = (1, 0.0),
            ["<unk>"]  = (2, 0.0),
        };

        int maxTokenLen = 1;
        foreach (var entry in vocabEl.EnumerateArray())
        {
            string tok = entry[0].GetString()!;
            double score = entry[1].GetDouble();
            if (!tokenToEntry.ContainsKey(tok))
            {
                tokenToEntry[tok] = (tokens.Count, score);
                tokens.Add(tok);
                if (tok.Length > maxTokenLen) maxTokenLen = tok.Length;
            }
        }

        return new T5Tokenizer(tokens.ToArray(), tokenToEntry, maxTokenLen) { MaxLen = maxLen };
    }

    /// <summary>
    /// Tokenize text using real Viterbi-optimal SentencePiece Unigram segmentation (maximizes total
    /// log-probability across the whole string, matching the real reference tokenizer -- NOT a
    /// greedy longest-match approximation).
    /// Returns token ids (no padding — caller truncates/pads as needed).
    /// </summary>
    public int[] Tokenize(string text)
    {
        // Normalize: T5 prepends a space to the input
        string normalized = "▁" + text.Replace(" ", "▁");
        int n = normalized.Length;

        // dp[i] = best cumulative log-prob to reach position i; back[i] = (start, tokenId) of the
        // best edge ending at i. dp[0] = 0 (empty prefix), everything else starts at -infinity.
        var dp = new double[n + 1];
        var backStart = new int[n + 1];
        var backId = new int[n + 1];
        Array.Fill(dp, double.NegativeInfinity);
        dp[0] = 0.0;

        for (int i = 1; i <= n; i++)
        {
            int minJ = Math.Max(0, i - _maxTokenLen);
            for (int j = i - 1; j >= minJ; j--)
            {
                if (double.IsNegativeInfinity(dp[j])) continue;
                string sub = normalized.Substring(j, i - j);
                if (_tokenToEntry.TryGetValue(sub, out var entry))
                {
                    double candidate = dp[j] + entry.Score;
                    if (candidate > dp[i])
                    {
                        dp[i] = candidate;
                        backStart[i] = j;
                        backId[i] = entry.Id;
                    }
                }
            }

            // Real sentencepiece guarantees full single-character coverage during training; as a
            // safety net for characters genuinely absent from this vocab, fall back to a
            // heavily-penalized single-character UNK edge from i-1, so segmentation can always
            // proceed rather than failing outright.
            if (double.IsNegativeInfinity(dp[i]) && !double.IsNegativeInfinity(dp[i - 1]))
            {
                dp[i] = dp[i - 1] + UnkScore;
                backStart[i] = i - 1;
                backId[i] = UnkToken;
            }
        }

        // Backtrack from n to reconstruct the FULL optimal segmentation (no early stop -- Viterbi
        // must run to completion to be optimal), then reverse and truncate from the END (keep the
        // leftmost tokens), matching real tokenizer truncation convention.
        var idsReversed = new List<int>();
        int pos = n;
        while (pos > 0)
        {
            idsReversed.Add(backId[pos]);
            pos = backStart[pos];
        }
        idsReversed.Reverse();

        var ids = idsReversed.Count > MaxLen - 1 ? idsReversed.GetRange(0, MaxLen - 1) : idsReversed;
        ids.Add(EosToken);
        return ids.ToArray();
    }
}
