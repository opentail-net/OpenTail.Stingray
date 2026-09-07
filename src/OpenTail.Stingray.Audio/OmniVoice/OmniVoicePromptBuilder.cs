namespace OpenTail.Stingray.Audio.OmniVoice;

/// <summary>
/// Real prompt-template builder for OmniVoice, ported from `prompt_builder.cpp`'s real
/// `OmniVoicePromptBuilder::build` (not guessed): `style_text =
/// "&lt;|lang_start|&gt;{lang}&lt;|lang_end|&gt;&lt;|instruct_start|&gt;{instruct}&lt;|instruct_end|&gt;"`
/// (both default to the literal string `"None"` when unset) and `wrapped_text =
/// "&lt;|text_start|&gt;{text}&lt;|text_end|&gt;"`, each tokenized independently via the
/// checkpoint's real tokenizer (`encode_with_nonverbal_tags` degenerates to plain `encode` for
/// text with no `[laughter]`-style bracketed nonverbal tags, confirmed from the reference's own
/// real fallback -- this port only implements that plain-text case).
///
/// <para><b>Real, deliberate scope limit</b>: `target_audio_tokens` real estimation
/// (`estimate_target_tokens`/`RuleDurationEstimator`, a real text-length/language-rate heuristic)
/// is NOT ported -- callers must supply a real target frame count themselves (e.g. via the
/// reference's own real `duration_seconds` override path, `round(durationSeconds * frameRate)`,
/// `frameRate = audioTokenizer.sampleRate / hopLength` = 24000/960 = 25 for this checkpoint).
/// Voice-cloning (real reference-audio-conditioned) prompts are also not yet built here -- only
/// the zero-shot (`AutoVoice`) case.</para>
/// </summary>
public static class OmniVoicePromptBuilder
{
    public readonly struct Prompt(int[] styleTokenIds, int[] textTokenIds)
    {
        public int[] StyleTokenIds { get; } = styleTokenIds;
        public int[] TextTokenIds { get; } = textTokenIds;
    }

    /// <summary>Builds the real zero-shot (`AutoVoice`, no reference audio, no instruct) prompt.
    /// `tokenize` is the caller's real BPE/Unigram encoder (e.g. `GgufTokenizer.Encode`).</summary>
    public static Prompt BuildZeroShot(Func<string, int[]> tokenize, string text, string? language = null)
    {
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("OmniVoice prompt requires non-empty text.", nameof(text));

        string styleText = $"<|lang_start|>{(string.IsNullOrEmpty(language) ? "None" : language)}<|lang_end|><|instruct_start|>None<|instruct_end|>";
        string wrappedText = $"<|text_start|>{text.Trim()}<|text_end|>";

        return new Prompt(tokenize(styleText), tokenize(wrappedText));
    }

    /// <summary>Real `frame_rate = audioTokenizer.sampleRate / hopLength`, confirmed 25 for this
    /// checkpoint's real config (sampleRate=24000, hopLength=960).</summary>
    public const int FrameRate = 25;

    /// <summary>Real `duration_seconds` override path: `target_audio_tokens =
    /// round(durationSeconds * frameRate)`, at least 1.</summary>
    public static int TargetFramesForDuration(float durationSeconds) =>
        Math.Max(1, (int)MathF.Round(durationSeconds * FrameRate));
}
