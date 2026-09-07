namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>
/// Real greedy-decode sampling utilities for VibeVoice ASR, ported from `session.cpp`'s anonymous
/// namespace (`argmax_token`/`apply_repetition_penalty`, not guessed). Beam search's
/// `top_log_probs` is a real, separate, lower-priority piece not yet ported (see
/// docs/audio-review-progress.md).
/// </summary>
public static class VibeVoiceSampling
{
    /// <summary>Real BF16-rounded argmax: the reference compares logits after a real
    /// fp32-&gt;bf16-&gt;fp32 round-trip when `compareBf16` is set (matches the checkpoint's real
    /// compute precision so tie-breaks land the same way as the reference), otherwise plain fp32
    /// comparison.</summary>
    public static int ArgmaxToken(ReadOnlySpan<float> logits, bool compareBf16)
    {
        if (logits.IsEmpty) throw new ArgumentException("VibeVoice-ASR decoder returned empty logits.", nameof(logits));

        int best = 0;
        float bestValue = compareBf16 ? Bf16Round(logits[0]) : logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            float value = compareBf16 ? Bf16Round(logits[i]) : logits[i];
            if (value > bestValue)
            {
                best = i;
                bestValue = value;
            }
        }
        return best;
    }

    /// <summary>Real repetition penalty: for every token id seen in either the prompt or the
    /// generated sequence so far (deduplicated -- each token id visited once), scales its logit
    /// by `penalty` if positive or divides by `penalty` if negative -- matches the reference's
    /// real branch exactly (NOT a uniform division), a common but easy-to-get-backwards HF
    /// `repetition_penalty` convention.</summary>
    public static void ApplyRepetitionPenalty(Span<float> logits, ReadOnlySpan<int> promptIds, ReadOnlySpan<int> generated, float penalty)
    {
        if (penalty == 1.0f) return;
        if (!(penalty > 0.0f) || !float.IsFinite(penalty))
            throw new ArgumentOutOfRangeException(nameof(penalty), "VibeVoice-ASR repetition_penalty must be finite and positive.");

        var seen = new bool[logits.Length];
        foreach (int token in promptIds) VisitToken(logits, seen, token, penalty);
        foreach (int token in generated) VisitToken(logits, seen, token, penalty);
    }

    private static void VisitToken(Span<float> logits, bool[] seen, int token, float penalty)
    {
        if ((uint)token >= (uint)logits.Length) return;
        if (seen[token]) return;
        seen[token] = true;
        logits[token] = logits[token] < 0.0f ? logits[token] * penalty : logits[token] / penalty;
    }

    private static float Bf16Round(float value)
    {
        // Real BF16 round-trip: truncate the mantissa to 7 bits (round-to-nearest-even via the
        // top bit of the discarded mantissa), matching ggml_fp32_to_bf16/ggml_bf16_to_fp32.
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if (float.IsNaN(value)) return value;
        uint roundingBias = 0x7FFFu + ((bits >> 16) & 1u);
        uint rounded = (bits + roundingBias) & 0xFFFF0000u;
        return BitConverter.UInt32BitsToSingle(rounded);
    }
}
