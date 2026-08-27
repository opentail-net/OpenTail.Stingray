using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>
/// Merges per-request HTTP sampling fields with the host-level defaults configured via
/// <see cref="OpenTailStingrayServerOptions"/>. Request-supplied values always win; the
/// host defaults are used only when the request omits a field. Speculative-decoding
/// knobs (<c>SpecType</c>, <c>SpecDraftNMax/NMin/PMin</c>) come from options only —
/// the API doesn't yet expose them per-request.
/// </summary>
internal static class SamplingParamsBuilder
{
    public static SamplingParams Build(
        OpenTailStingrayServerOptions options,
        float? temperature  = null,
        float? topP         = null,
        int?   topK         = null,
        float? minP         = null,
        float? repPenalty   = null,
        float? presencePenalty = null,
        float? frequencyPenalty = null,
        int?   maxTokens    = null,
        int?   maxThinking  = null,
        IReadOnlyDictionary<int, float>? logitBias = null,
        bool   thinkingDisabled = false,
        IReadOnlyList<string>? allowedChoices = null)
    {
        var d = options.Sampling;

        return new SamplingParams
        {
            Temperature       = temperature ?? d.Temperature,
            TopP              = topP        ?? d.TopP,
            TopK              = topK        ?? d.TopK,
            MinP              = minP        ?? d.MinP,
            RepetitionPenalty = repPenalty  ?? d.RepetitionPenalty,
            PresencePenalty   = presencePenalty ?? 0f,
            FrequencyPenalty  = frequencyPenalty ?? 0f,
            MaxNewTokens      = maxTokens   ?? d.MaxNewTokens,
            MaxThinkingTokens = maxThinking ?? d.MaxThinkingTokens,
            LogitBias         = logitBias,
            AllowedChoices    = allowedChoices,
            // The chat template for this request rendered enable_thinking=false —
            // lets the engine's greedy no-thinking fast paths (MTP, DSpark) engage
            // on thinking-capable models (CLI --no-thinking parity).
            ThinkingDisabled  = thinkingDisabled,

            SpecType          = MapSpecType(options.SpecType),
            SpecDraftNMax     = options.SpecDraftNMax,
            SpecDraftNMin     = options.SpecDraftNMin,
            SpecDraftPMin     = options.SpecDraftPMin,
        };
    }

    private static SpecType MapSpecType(ServerSpecType s) => s switch
    {
        ServerSpecType.Auto => SpecType.Auto,
        ServerSpecType.None => SpecType.None,
        ServerSpecType.Mtp  => SpecType.Mtp,
        _                   => SpecType.Auto,
    };
}
