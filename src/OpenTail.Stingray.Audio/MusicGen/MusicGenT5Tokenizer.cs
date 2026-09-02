
namespace OpenTail.Stingray.Audio.MusicGen;

/// <summary>
/// SentencePiece-compatible T5 Unigram tokenizer for MusicGen's `t5-base` text conditioner.
/// Same real Viterbi-optimal segmentation algorithm as
/// <see cref="Diffusion.TextEncoders.T5Tokenizer"/> (duplicated here rather than cross-referenced,
/// matching this codebase's existing per-domain-pipeline convention -- see
/// <see cref="MusicGenTextEncoderWeights"/>'s doc comment for why Parler similarly keeps its own
/// T5 encoder rather than sharing Diffusion's). BOS=0(unused, T5 has no BOS), EOS=1, UNK=2, PAD=0.
/// </summary>
public sealed class MusicGenT5Tokenizer
{
    private readonly Dictionary<string, (int Id, double Score)> _tokenToEntry;
    private readonly int _maxTokenLen;
    private const int EosToken = 1;
    private const int UnkToken = 2;
    private const double UnkScore = -100.0;

    /// <summary>Max output length (including the trailing EOS token). t5-base's own `n_positions` is 512; MusicGen prompts are short, so a much smaller practical cap is used.</summary>
    public int MaxLen { get; init; } = 256;

    private MusicGenT5Tokenizer(Dictionary<string, (int, double)> tokenToEntry, int maxTokenLen)
    {
        _tokenToEntry = tokenToEntry;
        _maxTokenLen = maxTokenLen;
    }

    /// <summary>Load from a HuggingFace `tokenizer.json` file (t5-base's own, NOT the raw `spiece.model` binary).</summary>
    public static MusicGenT5Tokenizer FromFile(string path, int maxLen = 256)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var model = doc.RootElement.GetProperty("model");
        var vocabEl = model.GetProperty("vocab");

        var tokenToEntry = new Dictionary<string, (int, double)>(StringComparer.Ordinal)
        {
            ["<pad>"] = (0, 0.0),
            ["</s>"] = (1, 0.0),
            ["<unk>"] = (2, 0.0),
        };

        int maxTokenLen = 1;
        foreach (var entry in vocabEl.EnumerateArray())
        {
            string tok = entry[0].GetString()!;
            double score = entry[1].GetDouble();
            if (!tokenToEntry.ContainsKey(tok))
            {
                tokenToEntry[tok] = (tokenToEntry.Count, score);
                if (tok.Length > maxTokenLen) maxTokenLen = tok.Length;
            }
        }

        return new MusicGenT5Tokenizer(tokenToEntry, maxTokenLen) { MaxLen = maxLen };
    }

    /// <summary>Tokenize via real Viterbi-optimal SentencePiece Unigram segmentation. Returns token ids including a trailing EOS.</summary>
    public int[] Tokenize(string text)
    {
        string normalized = "▁" + text.Replace(" ", "▁");
        int n = normalized.Length;

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

            if (double.IsNegativeInfinity(dp[i]) && !double.IsNegativeInfinity(dp[i - 1]))
            {
                dp[i] = dp[i - 1] + UnkScore;
                backStart[i] = i - 1;
                backId[i] = UnkToken;
            }
        }

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
