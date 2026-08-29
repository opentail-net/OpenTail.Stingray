
namespace OpenTail.Stingray.Audio.QwenASR;

/// <summary>
/// Qwen3-ASR ChatML prompt formatting and text decoding, built on top of the real BPE
/// tokenizer embedded in the checkpoint (<see cref="GgufTokenizer"/>, via
/// <see cref="QwenAsrWeights.Tokenizer"/>). This class previously hand-rolled its own fake
/// character-level vocabulary and a dedicated timestamp-token range that turned out not to
/// exist anywhere in the real checkpoint's vocabulary (~1500 fictional
/// <c>&lt;|timestamp_X.XX|&gt;</c> tokens) -- see docs/audio-review-progress.md's QwenASR
/// section for how that was found and corrected. Real segment-level ASR timestamps are not
/// confirmed to exist as a native model output for this checkpoint (the plan doc's own
/// section 20 distinguishes "ASR timestamp output" from "Forced alignment" -- forced
/// alignment is a genuinely separate model, <c>Qwen3-ForcedAligner-0.6B</c>, not bundled in
/// this GGUF); until that's independently verified, this class produces a single best-effort
/// segment spanning the whole decoded output rather than fabricating sub-segment timing.
/// </summary>
public sealed class QwenAsrTokenizer
{
    private readonly GgufTokenizer? _real;
    private readonly int _audioStartTokenId;
    private readonly int _audioEndTokenId;
    private readonly int _audioPadTokenId;
    private readonly int _eosTokenId;

    /// <summary>Real special-token ids from the checkpoint, or the verified real defaults when constructed without weights (structural-only use).</summary>
    public int AudioStartTokenId => _audioStartTokenId;
    public int AudioEndTokenId => _audioEndTokenId;
    public int AudioPadTokenId => _audioPadTokenId;
    public int EosTokenId => _eosTokenId;

    public int VocabSize => _real?.VocabSize ?? 151936;

    /// <summary>Structural-only constructor (no real tokenizer) -- Encode/Decode throw. Prefer <see cref="QwenAsrTokenizer(QwenAsrWeights)"/>.</summary>
    public QwenAsrTokenizer()
    {
        _audioStartTokenId = 151669;
        _audioEndTokenId = 151670;
        _audioPadTokenId = 151676;
        _eosTokenId = 151645;
    }

    public QwenAsrTokenizer(QwenAsrWeights weights)
    {
        _real = weights.Tokenizer;
        _audioStartTokenId = weights.AudioStartTokenId;
        _audioEndTokenId = weights.AudioEndTokenId;
        _audioPadTokenId = weights.AudioPadTokenId;
        _eosTokenId = weights.EosTokenId;
    }

    /// <summary>
    /// Formats the ChatML multimodal prompt: system + user turn with
    /// <c>&lt;|audio_start|&gt;&lt;|audio_pad|&gt;...&lt;|audio_pad|&gt;&lt;|audio_end|&gt;</c>
    /// (one <c>audio_pad</c> token per AuT-encoder output frame -- the LLM's embedding for each
    /// of those positions gets replaced by the encoder's projected audio features before decode,
    /// per the plan doc's "Phase 13: Multimodal Audio Injection", not yet wired up) + task text.
    /// </summary>
    public string FormatPrompt(int numAudioTokens, string? language = null, string? taskInstruction = null)
    {
        var sb = new StringBuilder();
        sb.Append("<|im_start|>system\nYou are a helpful speech-to-text assistant.<|im_end|>\n");
        sb.Append("<|im_start|>user\n<|audio_start|>");
        for (int i = 0; i < numAudioTokens; i++) sb.Append("<|audio_pad|>");
        sb.Append("<|audio_end|>\n");

        if (!string.IsNullOrEmpty(language))
            sb.Append($"Language: {language}\n");

        sb.Append(taskInstruction ?? "Transcribe the audio speech into text.");
        sb.Append("<|im_end|>\n<|im_start|>assistant\n");
        return sb.ToString();
    }

    /// <summary>Encodes a formatted prompt string into real BPE token ids.</summary>
    public int[] Encode(string text)
    {
        if (_real is null)
            throw new InvalidOperationException("QwenAsrTokenizer constructed without real weights -- use the QwenAsrWeights constructor for real encoding.");
        if (string.IsNullOrEmpty(text)) return [];
        return [.. _real.Encode(text)];
    }

    /// <summary>Decodes generated token ids into text.</summary>
    public string Decode(ReadOnlySpan<int> tokens)
    {
        if (_real is null)
            throw new InvalidOperationException("QwenAsrTokenizer constructed without real weights -- use the QwenAsrWeights constructor for real decoding.");
        return _real.Decode(tokens.ToArray());
    }

    /// <summary>
    /// Decodes generated tokens into text, dropping control/audio special tokens, and returns
    /// a single best-effort segment spanning the whole output (see this class's doc comment --
    /// no verified native sub-segment timestamp mechanism exists for this checkpoint).
    /// </summary>
    public (string FullText, List<SpeechSegment> Segments) DecodeWithTimestamps(
        ReadOnlySpan<int> tokens, TimeSpan timeOffset, TimeSpan duration)
    {
        var filtered = new List<int>(tokens.Length);
        foreach (int tid in tokens)
        {
            if (tid == _eosTokenId || tid == _audioStartTokenId || tid == _audioEndTokenId || tid == _audioPadTokenId)
                continue;
            filtered.Add(tid);
        }

        string text = filtered.Count > 0 ? Decode(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(filtered)).Trim() : string.Empty;
        var segments = new List<SpeechSegment>();
        if (text.Length > 0)
        {
            segments.Add(new SpeechSegment
            {
                Id = 0,
                Start = timeOffset,
                End = timeOffset + duration,
                Text = text,
                Tokens = filtered.ToArray(),
                Probability = 1.0f,
            });
        }
        return (text, segments);
    }
}
