namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>Real generation request options (`session.cpp`'s `request.generation.*` fields this
/// port implements -- greedy/argmax path only, matching this session's "beam search's
/// top_log_probs not yet ported" scope note).</summary>
public sealed class VibeVoiceAsrGenerationOptions
{
    public long MaxNewTokens { get; init; } = 32768;
    public float RepetitionPenalty { get; init; } = 1.0f;
}

/// <summary>
/// Real greedy-decode generation loop for VibeVoice ASR, ported from `session.cpp`'s
/// `generate_greedy_or_sample` (the `temperature &lt;= 0` / plain-argmax branch -- real sampling
/// with temperature/top-p/top-k is a separate, lower-priority piece not yet ported, matching this
/// session's existing sampling-utilities scope note). Real per-step algorithm: apply the real
/// repetition penalty to the current logits, argmax, stop on EOS or `MaxNewTokens`, otherwise embed
/// the sampled token and advance the KV cache by one incremental `Forward` step.
/// </summary>
public static class VibeVoiceAsrGenerator
{
    public static string GenerateTranscript(
        IForwardPass fwd,
        VibeVoiceAsrTextTokenizer tokenizer,
        VibeVoiceAsrPrompt prompt,
        VibeVoiceAsrGenerationOptions options)
    {
        var promptIds = prompt.InputIds;
        var logits = fwd.Prefill(promptIds, startPos: 0).ToArray();

        var generated = new List<int>();
        int position = promptIds.Length;
        for (long step = 0; step < options.MaxNewTokens; step++)
        {
            VibeVoiceSampling.ApplyRepetitionPenalty(logits, promptIds, [.. generated], options.RepetitionPenalty);
            int next = VibeVoiceSampling.ArgmaxToken(logits, compareBf16: false);

            if (next == tokenizer.EosTokenId) break;
            generated.Add(next);

            logits = fwd.Forward(next, position).ToArray();
            position++;
        }

        string rawText = tokenizer.Decode(generated);
        return VibeVoicePostprocessor.Decode(rawText).Text;
    }
}
