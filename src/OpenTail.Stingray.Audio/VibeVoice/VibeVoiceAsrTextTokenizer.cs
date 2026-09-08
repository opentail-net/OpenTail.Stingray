namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>One built prompt: token ids plus the row indices of the real
/// `&lt;|box_start|&gt;` speech placeholder tokens (to be remapped to
/// <see cref="VibeVoiceLlmTensorSource.SpeechTokenIdOffset"/>-based synthetic ids by the caller).</summary>
public readonly struct VibeVoiceAsrPrompt(int[] inputIds, int[] speechPositions)
{
    public int[] InputIds { get; } = inputIds;
    public int[] SpeechPositions { get; } = speechPositions;
}

/// <summary>
/// Real text tokenizer + prompt builder for VibeVoice ASR, ported from `tokenizer_text.cpp` (not
/// guessed): reuses this codebase's existing <see cref="GgufTokenizer"/>/
/// <see cref="HuggingFaceTokenizerSource"/> BPE machinery over the checkpoint's real embedded
/// `tokenizer.json`/`vocab.json`/`merges.txt` (extracted to a temp directory once at load time --
/// no bespoke tokenizer port needed, unlike MOSS-TTS-Nano's SentencePiece case). Real chat
/// template: `&lt;|im_start|&gt;role\n{content}&lt;|im_end|&gt;\n`, system message fixed
/// ("You are a helpful assistant that transcribes audio input into text output in JSON format."),
/// user message wraps `speechTokens` real `&lt;|box_start|&gt;` placeholders between
/// `&lt;|object_ref_start|&gt;`/`&lt;|object_ref_end|&gt;`, followed by a real instruction asking
/// for `Start time`/`End time`/`Speaker ID`/`Content` JSON keys.
/// </summary>
public sealed class VibeVoiceAsrTextTokenizer
{
    private readonly GgufTokenizer _tokenizer;
    private readonly int _speechPad;

    private const string SystemPrompt = "You are a helpful assistant that transcribes audio input into text output in JSON format.";

    public VibeVoiceAsrTextTokenizer(TokenizerSource source)
    {
        _tokenizer = GgufTokenizer.FromSource(source);
        _speechPad = RequireTokenId(source, "<|box_start|>");
    }

    /// <summary>Real `&lt;|endoftext|&gt;` id (`tokenizer_text.cpp`'s `eos_id()`, resolved via
    /// `require_token_id(tokenizer, "&lt;|endoftext|&gt;")`) -- `GgufTokenizer.EosTokenId` already
    /// resolves the checkpoint's real EOS from its `tokenizer_config.json`, no bespoke lookup
    /// needed.</summary>
    public int EosTokenId => _tokenizer.EosTokenId;

    public string Decode(IEnumerable<int> tokenIds) => _tokenizer.Decode(tokenIds);

    private static int RequireTokenId(TokenizerSource source, string token)
    {
        if (source.AdditionalSpecialTokens.TryGetValue(token, out int id)) return id;
        throw new InvalidDataException($"VibeVoice-ASR tokenizer missing token: {token}");
    }

    public int[] Encode(string text) => [.. _tokenizer.Encode(text)];

    private static string ChatMessage(string role, string content, bool addGenerationPrompt)
    {
        string result = $"<|im_start|>{role}\n{content}<|im_end|>\n";
        if (addGenerationPrompt) result += "<|im_start|>assistant\n";
        return result;
    }

    /// <summary>Builds the real prompt for a given audio duration and speech-token count (== the
    /// combined acoustic+semantic feature stream's frame count).</summary>
    public VibeVoiceAsrPrompt BuildPrompt(double audioSeconds, int speechTokens, string context = "")
    {
        if (speechTokens <= 0) throw new ArgumentOutOfRangeException(nameof(speechTokens));

        string suffix = context.Length > 0
            ? $"This is a {audioSeconds:F2} seconds audio, with extra info: {context}\n\nPlease transcribe it with these keys: Start time, End time, Speaker ID, Content"
            : $"This is a {audioSeconds:F2} seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content";

        var userContent = new StringBuilder("<|object_ref_start|>");
        for (int i = 0; i < speechTokens; i++) userContent.Append("<|box_start|>");
        userContent.Append("<|object_ref_end|>\n").Append(suffix);

        var ids = new List<int>();
        ids.AddRange(Encode(ChatMessage("system", SystemPrompt, false)));
        ids.AddRange(Encode(ChatMessage("user", userContent.ToString(), false)));

        var speechPositions = new List<int>();
        for (int i = 0; i < ids.Count; i++)
            if (ids[i] == _speechPad) speechPositions.Add(i);

        if (speechPositions.Count != speechTokens)
            throw new InvalidDataException("VibeVoice-ASR prompt speech token count mismatch.");

        return new VibeVoiceAsrPrompt([.. ids], [.. speechPositions]);
    }
}
