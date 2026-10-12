namespace OpenTail.Stingray.Engine;

/// <summary>
/// Per-request measurements the engine already computes (encode, prefill, first token, decode,
/// token counts), allocated once per request and referenced only from the final
/// <see cref="GenerateChunkKind.Stop"/> chunk. Counts and durations only; never content.
/// A later request-level metric adds a field here rather than a new plumbing path.
/// </summary>
/// <param name="PromptTokens">Encoded prompt length, including reused prefix tokens.</param>
/// <param name="ReusedTokens">Prompt tokens served from the prefix cache (not prefilled).</param>
/// <param name="OutputTokens">Tokens generated (the stop token is not counted).</param>
/// <param name="EncodeMs">Tokenisation time, from request start.</param>
/// <param name="PrefillMs">Prompt processing time for the new (not reused) tokens.</param>
/// <param name="FirstTokenMs">Request start to the first generated token, or -1 if none was produced.</param>
/// <param name="TotalMs">Request start to this snapshot.</param>
public sealed record GenerationStats(
    int PromptTokens,
    int ReusedTokens,
    int OutputTokens,
    long EncodeMs,
    long PrefillMs,
    long FirstTokenMs,
    long TotalMs)
{
    /// <summary>Prompt tokens that were actually prefilled.</summary>
    public int NewPromptTokens => PromptTokens - ReusedTokens;

    /// <summary>Milliseconds from the first generated token to this snapshot, or 0 if no token was produced.</summary>
    public long DecodeMs => FirstTokenMs >= 0 ? Math.Max(0, TotalMs - FirstTokenMs) : 0;
}
