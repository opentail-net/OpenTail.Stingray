using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Audio.NeuTts;

/// <summary>Real generation options for NeuTTS's AR speech-token loop.</summary>
public sealed class NeuTtsGenerationOptions
{
    public int MaxNewTokens { get; init; } = 2000;
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

        for (int step = 0; step < options.MaxNewTokens; step++)
        {
            int next = options.Sampling is null
                ? Argmax(logits)
                : Sampler.Sample(logits, options.Sampling, options.Rng);

            if (next == prompt.SpeechGenerationEnd) break;
            if (next < prompt.SpeechTokenStart || next > prompt.SpeechTokenEnd)
                throw new InvalidOperationException($"NeuTTS generated an out-of-range token {next} (expected a speech token or the stop token).");

            codes.Add(next - prompt.SpeechTokenStart);
            logits = fwd.Forward(next, pos);
            pos++;
        }

        return [.. codes];
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }
}
