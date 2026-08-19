using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

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
    public static ClipTokenizer FromFile(string? path = null)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            string[] candidates =
            {
                path ?? "",
                @"C:\Git-Public\OpenTail.Stingray\models\clip_tokenizer.json",
                Path.Combine(AppContext.BaseDirectory, "models", "clip_tokenizer.json"),
                Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "models", "clip_tokenizer.json"),
                Path.Combine(Directory.GetCurrentDirectory(), "models", "clip_tokenizer.json")
            };
            foreach (var c in candidates)
            {
                if (!string.IsNullOrWhiteSpace(c) && File.Exists(c))
                {
                    path = c;
                    break;
                }
            }
        }

        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            throw new FileNotFoundException($"CLIP tokenizer JSON file not found: {path}");

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
            var bpeTokens = Bpe(word);
            foreach (var tok in bpeTokens)
            {
                if (ids.Count >= MaxLen - 1) break; // leave room for EOS
                if (_vocab.TryGetValue(tok, out var id))
                    ids.Add(id);
            }
            if (ids.Count >= MaxLen - 1) break;
        }

        while (ids.Count < MaxLen)
            ids.Add(EosToken);

        return ids.ToArray();
    }

    private static List<string> SplitWords(string text)
    {
        var result = new List<string>();
        // Match word characters (with apostrophes) or punctuation
        foreach (Match m in Regex.Matches(text, @"\w+|[^\w\s]"))
            result.Add(m.Value + "</w>");
        return result;
    }

    private List<string> Bpe(string token)
    {
        // Start with individual characters
        var word = new List<string>();
        for (int i = 0; i < token.Length; i++)
        {
            if (token.Substring(i).StartsWith("</w>"))
            {
                word.Add("</w>");
                i += 3;
            }
            else
            {
                word.Add(token[i].ToString());
            }
        }

        while (word.Count > 1)
        {
            // Find the highest-priority merge in _merges that exists in word
            int bestMergeIdx = int.MaxValue;
            int bestPos = -1;

            for (int i = 0; i < word.Count - 1; i++)
            {
                var pair = (word[i], word[i + 1]);
                int idx = _merges.IndexOf(pair);
                if (idx >= 0 && idx < bestMergeIdx)
                {
                    bestMergeIdx = idx;
                    bestPos = i;
                }
            }

            if (bestPos < 0) break; // No more merges possible

            var merged = word[bestPos] + word[bestPos + 1];
            word[bestPos] = merged;
            word.RemoveAt(bestPos + 1);
        }

        return word;
    }
}
