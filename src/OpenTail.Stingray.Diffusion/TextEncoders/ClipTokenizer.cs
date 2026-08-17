using System.Text;
using System.Text.Json;

namespace OpenTail.Stingray.Diffusion.TextEncoders;

/// <summary>
/// CLIP BPE tokenizer compatible with OpenAI CLIP / Stable Diffusion.
/// Loads vocabulary and merges from a tokenizer.json (HuggingFace format).
///
/// Output tokens are in range [0, 49407] with:
///   BOS = 49406 (start of text)
///   EOS = 49407 (end of text / padding token)
/// Maximum sequence length: 77.
/// </summary>
public sealed class ClipTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly List<(string first, string second)> _merges;
    public const int BosToken = 49406;
    public const int EosToken = 49407;
    public const int MaxLen   = 77;

    private ClipTokenizer(Dictionary<string, int> vocab, List<(string, string)> merges)
    {
        _vocab  = vocab;
        _merges = merges;
    }

    /// <summary>Load from a HuggingFace tokenizer.json file.</summary>
    public static ClipTokenizer FromFile(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var model  = doc.RootElement.GetProperty("model");
        var vocabEl = model.GetProperty("vocab");
        var mergesEl = model.GetProperty("merges");

        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var kv in vocabEl.EnumerateObject())
            vocab[kv.Name] = kv.Value.GetInt32();

        var merges = new List<(string, string)>(mergesEl.GetArrayLength());
        foreach (var m in mergesEl.EnumerateArray())
        {
            var parts = m.GetString()!.Split(' ', 2);
            merges.Add((parts[0], parts[1]));
        }

        return new ClipTokenizer(vocab, merges);
    }

    /// <summary>
    /// Tokenize a text string. Returns 77 token ids.
    /// Padded with EosToken (49407) as required by CLIP/SD1.5.
    /// </summary>
    public int[] Tokenize(string text)
    {
        var ids = new List<int> { BosToken };

        var words = SplitWords(text.ToLowerInvariant());
        foreach (var word in words)
        {
            var wordTokens = BpeEncode(word);
            foreach (var tok in wordTokens)
            {
                if (ids.Count >= MaxLen - 1) break;
                if (_vocab.TryGetValue(tok, out int id)) ids.Add(id);
            }
            if (ids.Count >= MaxLen - 1) break;
        }

        ids.Add(EosToken);

        // Pad to MaxLen with EosToken (49407)
        while (ids.Count < MaxLen) ids.Add(EosToken);
        return ids.ToArray();
    }

    private List<string> BpeEncode(string word)
    {
        if (word.Length == 0) return new List<string>();

        var pairs = new List<string>(word.Length);
        for (int i = 0; i < word.Length - 1; i++)
            pairs.Add(word[i].ToString());
        pairs.Add(word[^1] + "</w>");

        var mergeRank = new Dictionary<(string, string), int>(_merges.Count);
        for (int i = 0; i < _merges.Count; i++)
            mergeRank[_merges[i]] = i;

        while (pairs.Count > 1)
        {
            int bestRank = int.MaxValue;
            int bestIdx  = -1;

            for (int i = 0; i < pairs.Count - 1; i++)
            {
                var pair = (pairs[i], pairs[i + 1]);
                if (mergeRank.TryGetValue(pair, out int rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIdx  = i;
                }
            }

            if (bestIdx < 0) break;

            string merged = pairs[bestIdx] + pairs[bestIdx + 1];
            pairs[bestIdx] = merged;
            pairs.RemoveAt(bestIdx + 1);
        }

        return pairs;
    }

    private static List<string> SplitWords(string text)
    {
        var result = new List<string>();
        var sb = new StringBuilder();

        foreach (char c in text)
        {
            if (char.IsWhiteSpace(c) || char.IsPunctuation(c))
            {
                if (sb.Length > 0) { result.Add(sb.ToString()); sb.Clear(); }
                if (char.IsPunctuation(c)) result.Add(c.ToString());
            }
            else
            {
                sb.Append(c);
            }
        }
        if (sb.Length > 0) result.Add(sb.ToString());
        return result;
    }
}
