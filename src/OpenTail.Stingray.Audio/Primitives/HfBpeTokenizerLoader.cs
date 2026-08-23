using System.Text.Json;

namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// Shared real HF byte-level BPE vocab/merges loading, extracted (DRY pass) after the same
/// logic was independently written twice this session: `QwenASR/QwenAsrWeights.cs`'s
/// `BuildTokenizerFromHfFiles` (Qwen3-ASR's Safetensors tokenizer) and
/// `CosyVoice/CosyVoiceLlmGeneration.cs`'s `BuildTokenizer` (CosyVoice2's real downloaded
/// Qwen2 tokenizer) -- both hit the exact same real HF convention: `vocab.json` only holds the
/// base byte-level BPE vocab, and every "added"/special token (im_start/im_end/audio_pad/etc.)
/// lives separately in `tokenizer_config.json`'s `added_tokens_decoder` (id -&gt; {content, ...}).
/// Treating `vocab.json` as complete leaves those token slots as an empty string, which
/// corrupts BPE encoding (a real bug this session found via an `OutOfMemoryException`, not by
/// inspection -- see the QwenASR entry in docs/audio-review-progress.md).
///
/// Callers still build their own `TokenizerSource` on top of this (Bos/Eos/Pad ids and
/// `AdditionalSpecialTokens` differ per checkpoint), so this only extracts the shared file-
/// parsing half, not the whole tokenizer construction.
/// </summary>
public static class HfBpeTokenizerLoader
{
    /// <summary>
    /// Reads `{dir}/vocab.json` (id-complete token array, `vocabSize` sized to cover every id
    /// referenced by either file) and `{dir}/merges.txt`. `AddedTokensByContent` maps each real
    /// `added_tokens_decoder` entry's content string to its real id, for callers that need to
    /// look specific special tokens up by name (e.g. CosyVoice2's `&lt;|im_end|&gt;`).
    /// </summary>
    public static (string[] Tokens, string[] Merges, Dictionary<string, int> AddedTokensByContent) Load(string dir)
    {
        using var vocabDoc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "vocab.json")));
        var vocabProps = vocabDoc.RootElement;

        using var tokConfigDoc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "tokenizer_config.json")));
        var addedTokens = tokConfigDoc.RootElement.TryGetProperty("added_tokens_decoder", out var atd) ? atd : default;

        int vocabSize = 0;
        foreach (var p in vocabProps.EnumerateObject()) vocabSize = Math.Max(vocabSize, p.Value.GetInt32() + 1);
        if (addedTokens.ValueKind == JsonValueKind.Object)
            foreach (var p in addedTokens.EnumerateObject()) vocabSize = Math.Max(vocabSize, int.Parse(p.Name) + 1);

        var tokens = new string[vocabSize];
        foreach (var p in vocabProps.EnumerateObject())
        {
            int id = p.Value.GetInt32();
            if ((uint)id < (uint)tokens.Length) tokens[id] = p.Name;
        }

        var addedByContent = new Dictionary<string, int>(StringComparer.Ordinal);
        if (addedTokens.ValueKind == JsonValueKind.Object)
        {
            foreach (var p in addedTokens.EnumerateObject())
            {
                int id = int.Parse(p.Name);
                string content = p.Value.GetProperty("content").GetString()!;
                if ((uint)id < (uint)tokens.Length) tokens[id] = content;
                addedByContent[content] = id;
            }
        }
        for (int i = 0; i < tokens.Length; i++) tokens[i] ??= string.Empty;

        var mergesLines = File.ReadAllLines(Path.Combine(dir, "merges.txt"));
        var merges = new List<string>(mergesLines.Length);
        foreach (var line in mergesLines)
        {
            if (line.Length == 0 || line.StartsWith('#')) continue;
            merges.Add(line);
        }

        return (tokens, [.. merges], addedByContent);
    }

    /// <summary>Grows `vocabSize` to cover every id given (helper for callers that need extra explicit ids beyond what vocab.json/added_tokens_decoder cover on their own, e.g. QwenASR's audio_start/end/pad ids).</summary>
    public static string[] EnsureCovers(string[] tokens, params int[] ids)
    {
        int max = tokens.Length;
        foreach (var id in ids) max = Math.Max(max, id + 1);
        if (max == tokens.Length) return tokens;

        var grown = new string[max];
        Array.Copy(tokens, grown, tokens.Length);
        for (int i = tokens.Length; i < grown.Length; i++) grown[i] = string.Empty;
        return grown;
    }
}
