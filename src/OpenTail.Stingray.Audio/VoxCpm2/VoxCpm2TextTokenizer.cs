using System.Text;
using System.Text.Json;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.VoxCpm2;

/// <summary>
/// Native port of VoxCPM2's real text tokenizer (`tokenizer_text.cpp`'s
/// `VoxCPM2TextTokenizer`, not guessed). Its `tokenizer.json` is a standard Hugging Face BPE
/// export with a SentencePiece-style space-to-'▁' `Replace` normalizer, so the base
/// vocab/merges/normalization is reused directly through the existing
/// <see cref="HuggingFaceTokenizerSource"/>/<see cref="GgufTokenizer"/> pipeline (same real
/// space&lt;-&gt;metaspace substitution path already used for Gemma/Llama SPM-style exports) --
/// no need to reimplement BPE merging from scratch a second time this session (MOSS-TTS-Nano
/// already required a bespoke port because ITS tokenizer is a raw SentencePiece protobuf, not
/// an HF `tokenizer.json`).
///
/// <para>Two genuinely bespoke pieces the generic loader does not cover, ported directly from
/// the real reference: (1) the four reserved audio-boundary token ids (`&lt;|audio_start|&gt;`,
/// `&lt;|audio_end|&gt;`, `&lt;|audio_prompt_start|&gt;`, `&lt;|audio_prompt_end|&gt;`) that the
/// generation loop splices around text/reference-audio spans; (2) `build_cjk_split_map`'s real
/// post-merge expansion -- any vocabulary token whose text (after stripping the SentencePiece
/// `▁` marker) is composed of two or more CJK codepoints, each of which ALSO exists as its own
/// single-codepoint vocab entry, gets expanded back into those individual codepoint ids after
/// BPE merging. This is real, deliberate reference behavior (not a bug) -- it keeps VoxCPM2's
/// downstream per-character-ish CJK prosody modeling from seeing multi-character merged tokens
/// it wasn't trained to expect in this position, even though the BPE merge table would happily
/// produce them.</para>
/// </summary>
public sealed class VoxCpm2TextTokenizer
{
    private readonly GgufTokenizer _tokenizer;
    private readonly Dictionary<int, int[]> _cjkSplitMap;

    public int AudioStartTokenId { get; }
    public int AudioEndTokenId { get; }
    public int ReferenceAudioStartTokenId { get; }
    public int ReferenceAudioEndTokenId { get; }

    private VoxCpm2TextTokenizer(GgufTokenizer tokenizer, Dictionary<int, int[]> cjkSplitMap,
        int audioStart, int audioEnd, int refAudioStart, int refAudioEnd)
    {
        _tokenizer = tokenizer;
        _cjkSplitMap = cjkSplitMap;
        AudioStartTokenId = audioStart;
        AudioEndTokenId = audioEnd;
        ReferenceAudioStartTokenId = refAudioStart;
        ReferenceAudioEndTokenId = refAudioEnd;
    }

    /// <summary>Extracts `tokenizer.json`/`tokenizer_config.json`/`special_tokens_map.json` from
    /// a packed GGUF's `audiocpp.embedded_files.*` metadata into a scratch directory, then loads
    /// from there. The scratch directory is left on disk (small text files, reused across
    /// calls in the same process by content-hash naming) rather than cleaned up per call.</summary>
    public static VoxCpm2TextTokenizer LoadFromPackedGguf(GgufModel model)
    {
        if (!model.Metadata.TryGetValue("audiocpp.embedded_files.names", out var namesObj) || namesObj is not object[] names)
            throw new InvalidOperationException("VoxCPM2 packed GGUF has no embedded_files metadata.");
        var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
        var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
        var bytes = data.Select(o => (byte)Convert.ToInt64(o)).ToArray();

        string dir = Path.Combine(Path.GetTempPath(), "stingray-voxcpm2-tokenizer");
        Directory.CreateDirectory(dir);

        string[] wanted = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"];
        for (int i = 0; i < names.Length; i++)
        {
            string name = (string)names[i];
            if (Array.IndexOf(wanted, name) < 0) continue;
            long start = Convert.ToInt64(offsets[i]);
            long end = i + 1 < offsets.Length ? Convert.ToInt64(offsets[i + 1]) : bytes.Length;
            File.WriteAllBytes(Path.Combine(dir, name), bytes[(int)start..(int)end]);
        }

        return Load(dir);
    }

    /// <summary>Loads from an extracted package directory containing `tokenizer.json` (required)
    /// and `tokenizer_config.json`/`special_tokens_map.json` (optional sidecars).</summary>
    public static VoxCpm2TextTokenizer Load(string packageRoot)
    {
        var result = HuggingFaceTokenizerSource.Load(packageRoot);
        if (result.Source is null)
            throw new InvalidOperationException(
                $"VoxCPM2 tokenizer load failed: {string.Join("; ", result.Rejections)}");

        var source = result.Source;
        var tokenizer = GgufTokenizer.FromSource(source);

        var byToken = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < source.Tokens.Length; i++) byToken.TryAdd(source.Tokens[i], i);

        // The real reference (`load_special_tokens` in `tokenizer_text.cpp`) reads these four
        // reserved ids from tokenizer_config.json's `added_tokens_decoder` directly -- they are
        // NOT present in tokenizer.json's own `added_tokens` array for this checkpoint, so
        // HuggingFaceTokenizerSource's generic loader (which only reads that array) never sees
        // them. Parsed here the same way, then merged into the lookup table.
        string configPath = Path.Combine(packageRoot, "tokenizer_config.json");
        if (File.Exists(configPath))
        {
            using var document = JsonDocument.Parse(File.ReadAllBytes(configPath));
            if (document.RootElement.TryGetProperty("added_tokens_decoder", out var added)
                && added.ValueKind == JsonValueKind.Object)
            {
                foreach (var entry in added.EnumerateObject())
                {
                    if (!entry.Value.TryGetProperty("content", out var contentEl)
                        || contentEl.ValueKind != JsonValueKind.String) continue;
                    string? content = contentEl.GetString();
                    if (content is null || !int.TryParse(entry.Name, out int id)) continue;
                    byToken[content] = id;
                }
            }
        }

        int RequireId(string token) => byToken.TryGetValue(token, out int id)
            ? id
            : throw new InvalidOperationException($"VoxCPM2 tokenizer missing token: {token}");

        var cjkSplitMap = BuildCjkSplitMap(source.Tokens, byToken);

        return new VoxCpm2TextTokenizer(tokenizer, cjkSplitMap,
            RequireId("<|audio_start|>"), RequireId("<|audio_end|>"),
            RequireId("<|audio_prompt_start|>"), RequireId("<|audio_prompt_end|>"));
    }

    /// <summary>Real `VoxCPM2TextTokenizer::encode` -- base BPE encode, then real CJK
    /// multi-char-token expansion applied to every resulting id.</summary>
    public List<int> Encode(string text)
    {
        var baseIds = _tokenizer.Encode(text);
        var expanded = new List<int>(baseIds.Count);
        foreach (int id in baseIds)
        {
            if (_cjkSplitMap.TryGetValue(id, out var split)) expanded.AddRange(split);
            else expanded.Add(id);
        }
        return expanded;
    }

    private static bool IsCjkCodepoint(int codepoint) =>
        (codepoint >= 0x4E00 && codepoint <= 0x9FFF) ||
        (codepoint >= 0x3400 && codepoint <= 0x4DBF) ||
        (codepoint >= 0xF900 && codepoint <= 0xFAFF) ||
        (codepoint >= 0x20000 && codepoint <= 0x2A6DF);

    private static bool IsPureMultiCharCjk(string text, out List<string> codepoints)
    {
        codepoints = [];
        var enumerator = System.Globalization.StringInfo.GetTextElementEnumerator(text);
        int offset = 0;
        while (offset < text.Length)
        {
            int cp = char.ConvertToUtf32(text, offset);
            if (!IsCjkCodepoint(cp)) { codepoints = []; return false; }
            string s = char.ConvertFromUtf32(cp);
            codepoints.Add(s);
            offset += s.Length;
        }
        return codepoints.Count >= 2;
    }

    private static Dictionary<int, int[]> BuildCjkSplitMap(string[] tokens, Dictionary<string, int> byToken)
    {
        var map = new Dictionary<int, int[]>();
        const string metaspace = "▁";
        for (int id = 0; id < tokens.Length; id++)
        {
            string token = tokens[id];
            if (token is null) continue;
            string clean = token.Replace(metaspace, "", StringComparison.Ordinal);
            if (!IsPureMultiCharCjk(clean, out var codepoints)) continue;

            var charIds = new int[codepoints.Count];
            bool allFound = true;
            for (int i = 0; i < codepoints.Count; i++)
            {
                if (!byToken.TryGetValue(codepoints[i], out int charId)) { allFound = false; break; }
                charIds[i] = charId;
            }
            if (allFound) map[id] = charIds;
        }
        return map;
    }
}
