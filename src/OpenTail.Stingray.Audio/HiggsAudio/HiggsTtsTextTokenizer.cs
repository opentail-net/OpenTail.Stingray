using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.HiggsAudio;

/// <summary>One built prompt: the ordered token ids for Higgs TTS's generation prefill, plus
/// the count of reserved audio-placeholder ids inserted for the reference-audio span (to be
/// replaced by the caller's real reference-audio embeddings before prefill).</summary>
public readonly struct HiggsTtsPromptEncoding(int[] tokenIds, int delayedReferenceTokens)
{
    public int[] TokenIds { get; } = tokenIds;
    public int DelayedReferenceTokens { get; } = delayedReferenceTokens;
}

/// <summary>
/// Real text tokenizer + prompt builder for Higgs Audio TTS, ported from `tokenizer_text.cpp`
/// (not guessed): a plain Qwen2-family byte-level BPE tokenizer (real reference:
/// `LlamaBpePreTokenizer::Qwen2`, standard GPT-2-style byte-level pre-tokenization -- NOT the
/// SentencePiece-metaspace-style path VibeVoice/VoxCPM2 needed) over the checkpoint's real
/// `tokenizer.json`/`tokenizer_config.json`, reusing this codebase's existing
/// <see cref="GgufTokenizer"/>/<see cref="HuggingFaceTokenizerSource"/> BPE machinery directly.
///
/// <para>Real prompt layout (`encode_prompt`'s exact real structure, not simplified):
/// `[&lt;|tts|&gt;] [&lt;|ref_text|&gt; ref_text_ids...]? [&lt;|ref_audio|&gt;
/// audio_placeholder_id * delayedReferenceTokens]? [&lt;|text|&gt; text_ids...]
/// [&lt;|audio|&gt;]`. The reference-text span is included ONLY when both
/// `referenceText` is non-empty AND `delayedReferenceTokens > 0`; the reference-audio
/// placeholder span is inserted independently whenever `delayedReferenceTokens > 0` (a
/// zero-shot generation with no reference text but a reference-audio embedding is a real,
/// valid combination). `audio_placeholder_id` comes from the checkpoint's own
/// `config.audio_token_id` (real reference casts it directly, not a special-token lookup) --
/// callers must splice real reference-audio-codec-derived embeddings into these placeholder
/// positions before prefill, mirroring every other bridge class's "extra vocab rows" /
/// "precomputed embedding" convention this session.</para>
/// </summary>
public sealed class HiggsTtsTextTokenizer
{
    private readonly GgufTokenizer _tokenizer;
    private readonly int _ttsId;
    private readonly int _refAudioId;
    private readonly int _refTextId;
    private readonly int _textId;
    private readonly int _audioId;
    private readonly int _audioPlaceholderId;

    public HiggsTtsTextTokenizer(TokenizerSource source, int audioTokenId)
    {
        _tokenizer = GgufTokenizer.FromSource(source);
        _ttsId = RequireTokenId(source, "<|tts|>");
        _refAudioId = RequireTokenId(source, "<|ref_audio|>");
        _refTextId = RequireTokenId(source, "<|ref_text|>");
        _textId = RequireTokenId(source, "<|text|>");
        _audioId = RequireTokenId(source, "<|audio|>");
        _audioPlaceholderId = audioTokenId;
    }

    /// <summary>Extracts `tokenizer.json`/`tokenizer_config.json`/`special_tokens_map.json` from
    /// a packed GGUF's `audiocpp.embedded_files.*` metadata into a scratch directory, then loads
    /// from there (same technique as `VoxCpm2TextTokenizer.LoadFromPackedGguf`).</summary>
    public static HiggsTtsTextTokenizer LoadFromPackedGguf(GgufModel model, int audioTokenId)
    {
        if (!model.Metadata.TryGetValue("audiocpp.embedded_files.names", out var namesObj) || namesObj is not object[] names)
            throw new InvalidOperationException("Higgs TTS packed GGUF has no embedded_files metadata.");
        var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
        var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
        var bytes = data.Select(o => (byte)Convert.ToInt64(o)).ToArray();

        string dir = Path.Combine(Path.GetTempPath(), "stingray-higgs-tts-tokenizer");
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

        var result = HuggingFaceTokenizerSource.Load(dir);
        if (result.Source is null)
            throw new InvalidOperationException(
                $"Higgs TTS tokenizer load failed: {string.Join("; ", result.Rejections)}");
        return new HiggsTtsTextTokenizer(result.Source, audioTokenId);
    }

    private static int RequireTokenId(TokenizerSource source, string token)
    {
        if (source.AdditionalSpecialTokens.TryGetValue(token, out int id)) return id;
        throw new InvalidDataException($"Higgs TTS tokenizer missing token: {token}");
    }

    public int[] Encode(string text) => [.. _tokenizer.Encode(text)];

    /// <summary>Real `HiggsTextTokenizer::encode_prompt`.</summary>
    public HiggsTtsPromptEncoding EncodePrompt(string text, string referenceText, int delayedReferenceTokens)
    {
        if (delayedReferenceTokens < 0) throw new ArgumentOutOfRangeException(nameof(delayedReferenceTokens));

        var textIds = Encode(text);
        int[] referenceTextIds = !string.IsNullOrEmpty(referenceText) && delayedReferenceTokens > 0
            ? Encode(referenceText) : [];

        var ids = new List<int> { _ttsId };
        if (referenceTextIds.Length > 0)
        {
            ids.Add(_refTextId);
            ids.AddRange(referenceTextIds);
        }
        if (delayedReferenceTokens > 0)
        {
            ids.Add(_refAudioId);
            for (int i = 0; i < delayedReferenceTokens; i++) ids.Add(_audioPlaceholderId);
        }
        ids.Add(_textId);
        ids.AddRange(textIds);
        ids.Add(_audioId);

        return new HiggsTtsPromptEncoding([.. ids], delayedReferenceTokens);
    }
}
