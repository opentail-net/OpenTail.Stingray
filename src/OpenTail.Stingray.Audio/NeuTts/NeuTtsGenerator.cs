using System.Buffers;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Audio.NeuTts;

/// <summary>Real generation options for NeuTTS's AR speech-token loop.</summary>
public sealed class NeuTtsGenerationOptions
{
    public int MaxNewTokens { get; init; } = 2000;
    public int MinTokens { get; init; } = 50;
    public SamplingParams? Sampling { get; init; }
    public Random? Rng { get; init; }
}

/// <summary>
/// Real autoregressive speech-token generation loop for NeuTTS, driven by the standard
/// `IForwardPass` prefill/decode contract (this checkpoint needs no bespoke per-step wiring --
/// see `NeuTtsPromptBuilder`'s doc comment for why: speech tokens are plain vocabulary ids here).
/// Prefills the full prompt, then greedily (or, with real sampling params, via the shared
/// `Sampler`) decodes one token at a time until `SpeechGenerationEnd` or `maxNewTokens`. Returns
/// the real codec codes (each generated token minus `SpeechTokenStart`) for the downstream FSQ
/// audio-codec decoder (not yet ported this session).
/// </summary>
public static class NeuTtsGenerator
{
    public static int[] GenerateSpeechCodes(IForwardPass fwd, NeuTtsPrompt prompt, NeuTtsGenerationOptions options)
    {
        var codes = new List<int>();
        var logits = fwd.Prefill(prompt.TokenIds, startPos: 0);
        int pos = prompt.TokenIds.Length;

        float[]? rentedLogits = null;
        try
        {
            for (int step = 0; step < options.MaxNewTokens; step++)
            {
                int next;
                if (options.Sampling is null)
                {
                    int maskedIndex = (step < options.MinTokens && prompt.SpeechGenerationEnd < logits.Length)
                        ? prompt.SpeechGenerationEnd
                        : -1;
                    next = Argmax(logits, maskedIndex);
                }
                else
                {
                    ReadOnlySpan<float> activeLogits = logits;
                    if (step < options.MinTokens && prompt.SpeechGenerationEnd < logits.Length)
                    {
                        rentedLogits ??= ArrayPool<float>.Shared.Rent(logits.Length);
                        logits.CopyTo(rentedLogits.AsSpan(0, logits.Length));
                        rentedLogits[prompt.SpeechGenerationEnd] = float.NegativeInfinity;
                        activeLogits = rentedLogits.AsSpan(0, logits.Length);
                    }
                    next = Sampler.Sample(activeLogits, options.Sampling, options.Rng);
                }

                if (next == prompt.SpeechGenerationEnd && step >= options.MinTokens) break;
                if (next < prompt.SpeechTokenStart || next > prompt.SpeechTokenEnd)
                    throw new InvalidOperationException($"NeuTTS generated an out-of-range token {next} (expected a speech token or the stop token).");

                codes.Add(next - prompt.SpeechTokenStart);
                logits = fwd.Forward(next, pos);
                pos++;
            }
        }
        finally
        {
            if (rentedLogits is not null)
                ArrayPool<float>.Shared.Return(rentedLogits);
        }

        return [.. codes];
    }

    private static int Argmax(ReadOnlySpan<float> logits, int maskedIndex = -1)
    {
        int best = (maskedIndex == 0 && logits.Length > 1) ? 1 : 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (i == maskedIndex) continue;
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }
}
