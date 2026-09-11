
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
    /// Formats the ChatML multimodal prompt, matching the REAL reference template found in the
    /// vendored `examples/audio.cpp/src/models/qwen3_asr/tokenizer_text.cpp`'s
    /// `default_chat_prompt`/`build_prompt` (2026-09-12 fix -- the previous version here was a
    /// plausible-looking but unverified invention, per this method's own prior doc comment
    /// admitting the task-text placement was never checked against a real reference).
    ///
    /// The real template is structurally different from what you might guess from a generic
    /// ChatML/VLM convention:
    /// - System turn's content ("context") is EMPTY by default in the real pipeline (only
    ///   populated from an optional user-supplied hint/hotword string, `taskInstruction` here) --
    ///   NOT a fixed "You are a helpful..." system prompt.
    /// - The user turn contains ONLY the audio block
    ///   (<c>&lt;|audio_start|&gt;&lt;|audio_pad|&gt;...&lt;|audio_end|&gt;</c>) with NO
    ///   trailing instruction text at all -- unlike a typical VLM prompt, the real reference
    ///   never appends "Transcribe the audio..." inside the user turn.
    /// - <c>language</c>, when set, is NOT prose ("Language: en") appended before the user
    ///   turn's &lt;|im_end|&gt; -- it's the literal string "language {lang}&lt;asr_text&gt;"
    ///   seeded as a forced PREFIX of the assistant's own turn (appended directly after
    ///   &lt;|im_start|&gt;assistant\n, with no &lt;|im_end|&gt; in between), i.e. part of
    ///   the PROMPT tokens fed into prefill, not something the model generates. &lt;asr_text&gt;
    ///   is a real special token in this checkpoint's own vocab (confirmed via
    ///   `tokenizer_config.json`) that appears to act as a trigger telling the model "now emit
    ///   the actual transcript" -- omitting it left the model with no signal for when its real
    ///   completion should start, producing a short generic non-transcript reply then immediate
    ///   EOS (confirmed real bug, see docs/00-current-work.md's 2026-09-12 entry: exactly one
    ///   token generated then EOS, regardless of real 14s speech input).
    /// </summary>
    public string FormatPrompt(int numAudioTokens, string? language = null, string? taskInstruction = null)
    {
        var sb = new StringBuilder();
        sb.Append("<|im_start|>system\n").Append(taskInstruction ?? "").Append("<|im_end|>\n");
        sb.Append("<|im_start|>user\n<|audio_start|>");
        for (int i = 0; i < numAudioTokens; i++) sb.Append("<|audio_pad|>");
        sb.Append("<|audio_end|><|im_end|>\n<|im_start|>assistant\n");

        if (!string.IsNullOrEmpty(language) && !string.Equals(language, "Auto", StringComparison.OrdinalIgnoreCase))
            sb.Append($"language {language}<asr_text>");

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
