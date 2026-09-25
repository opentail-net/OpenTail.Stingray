using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Core;

/// <summary>
/// OpenAI CLIP tokenizer from a HF <c>tokenizer.json</c>, following that file's own pipeline: normalizer NFC → collapse
/// whitespace runs to one space → lowercase; pre-tokenizer Split on
/// <c>&lt;|startoftext|&gt;|&lt;|endoftext|&gt;|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+</c> (so digits are
/// one piece each) then ByteLevel (GPT-2 byte→unicode map); BPE with <c>end_of_word_suffix</c> "&lt;/w&gt;" fused onto the
/// last character; RobertaProcessing <c>&lt;|startoftext|&gt; … &lt;|endoftext|&gt;</c>.
/// <para>Limitation: under this repo's <c>InvariantGlobalization</c>, <c>string.Normalize(FormC)</c> does not compose
/// non-ASCII sequences, so pre-composed input is assumed (true for ordinary text).</para>
/// </summary>
public sealed partial class ClipBpeTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<(string, string), int> _ranks;
    private readonly Dictionary<string, int[]> _cache = new(StringComparer.Ordinal);
    private static readonly char[] s_byteToChar = BuildByteMap();

    public int StartId { get; }
    public int EndId { get; }

    private ClipBpeTokenizer(Dictionary<string, int> vocab, Dictionary<(string, string), int> ranks, int start, int end)
    {
        (_vocab, _ranks, StartId, EndId) = (vocab, ranks, start, end);
    }

    [GeneratedRegex(@"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+")]
    private static partial Regex PreTokenize();

    [GeneratedRegex(@"\s+")]
    private static partial Regex Whitespace();

    public static ClipBpeTokenizer FromTokenizerJson(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var model = doc.RootElement.GetProperty("model");
        if (model.GetProperty("type").GetString() != "BPE" ||
            (model.TryGetProperty("end_of_word_suffix", out var eow) ? eow.GetString() : null) != "</w>")
            throw new NotSupportedException($"{path}: not a CLIP BPE tokenizer (BPE with end_of_word_suffix \"</w>\").");
        var vocab = model.GetProperty("vocab").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetInt32(), StringComparer.Ordinal);
        var ranks = new Dictionary<(string, string), int>();
        int rank = 0;
        foreach (var m in model.GetProperty("merges").EnumerateArray())
        {
            string a, b;
            if (m.ValueKind == JsonValueKind.Array) (a, b) = (m[0].GetString()!, m[1].GetString()!);
            else
            {
                var parts = m.GetString()!.Split(' ', 2);
                (a, b) = (parts[0], parts[1]);
            }
            ranks.TryAdd((a, b), rank++);
        }
        return new ClipBpeTokenizer(vocab, ranks, vocab["<|startoftext|>"], vocab["<|endoftext|>"]);
    }

    /// <summary>Token ids with start/end tokens, truncated to <paramref name="maxLength"/> (the end token is kept).</summary>
    public int[] Encode(string text, int maxLength = 77)
    {
        string norm = Whitespace().Replace(text.Normalize(NormalizationForm.FormC), " ").ToLowerInvariant();
        var ids = new List<int> { StartId };
        foreach (Match m in PreTokenize().Matches(norm))
        {
            if (m.Value is "<|startoftext|>" or "<|endoftext|>") { ids.Add(_vocab[m.Value]); continue; }
            var bytes = Encoding.UTF8.GetBytes(m.Value);
            var sb = new StringBuilder(bytes.Length);
            foreach (byte b in bytes) sb.Append(s_byteToChar[b]);
            ids.AddRange(Bpe(sb.ToString()));
        }
        if (ids.Count > maxLength - 1) ids.RemoveRange(maxLength - 1, ids.Count - (maxLength - 1));
        ids.Add(EndId);
        return ids.ToArray();
    }

    private int[] Bpe(string piece)
    {
        if (_cache.TryGetValue(piece, out var cached)) return cached;
        var word = new List<string>(piece.Length);
        for (int i = 0; i < piece.Length; i++) word.Add(piece[i].ToString());
        word[^1] += "</w>";
        while (word.Count > 1)
        {
            int best = int.MaxValue, pos = -1;
            for (int i = 0; i < word.Count - 1; i++)
                if (_ranks.TryGetValue((word[i], word[i + 1]), out int r) && r < best) (best, pos) = (r, i);
            if (pos < 0) break;
            // Merge every occurrence of the best pair, left to right (as HF BPE does).
            string a = word[pos], b = word[pos + 1];
            var merged = new List<string>(word.Count);
            for (int i = 0; i < word.Count; i++)
            {
                if (i < word.Count - 1 && word[i] == a && word[i + 1] == b) { merged.Add(a + b); i++; }
                else merged.Add(word[i]);
            }
            word = merged;
        }
        var ids = word.Select(t => _vocab.TryGetValue(t, out int id) ? id : EndId).ToArray(); // unk_token = <|endoftext|>
        _cache[piece] = ids;
        return ids;
    }

    /// <summary>GPT-2 <c>bytes_to_unicode</c>: printable bytes map to themselves, the rest to U+0100 onward.</summary>
    private static char[] BuildByteMap()
    {
        var map = new char[256];
        var direct = new List<int>();
        for (int b = '!'; b <= '~'; b++) direct.Add(b);
        for (int b = 0xA1; b <= 0xAC; b++) direct.Add(b);
        for (int b = 0xAE; b <= 0xFF; b++) direct.Add(b);
        int n = 0;
        for (int b = 0; b < 256; b++) map[b] = direct.Contains(b) ? (char)b : (char)(256 + n++);
        return map;
    }
}
