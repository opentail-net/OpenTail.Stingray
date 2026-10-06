#nullable enable

using OpenTail.Stingray.Core;
using OpenTail.Stingray.Core.Grammar;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray;

/// <summary>
/// Contract for per-generation execution parameters.
/// Inspired by LLamaSharp's <c>IInferenceParams</c>, extended with Stingray's reasoning, stopping, and speculative decoding controls.
/// </summary>
public interface IInferenceParams
{
    /// <summary>Maximum number of new tokens to generate.</summary>
    int MaxTokens { get; }

    /// <summary>Optional RNG seed for deterministic sampling across generations.</summary>
    int? Seed { get; }

    /// <summary>
    /// Optional chat-template render of the message history with add_generation_prompt=false.
    /// Used for canonical-history prefix caching across multi-turn conversations.
    /// </summary>
    string? CanonicalHistoryPrefix { get; }

    /// <summary>Sampling temperature. 0.0 specifies deterministic greedy decoding.</summary>
    float Temperature { get; }

    /// <summary>Top-K sampling limit. 0 disables top-k truncation.</summary>
    int TopK { get; }

    /// <summary>Top-P (nucleus) sampling threshold. 1.0 disables nucleus filtering.</summary>
    float TopP { get; }

    /// <summary>Min-P sampling threshold relative to the most likely token.</summary>
    float MinP { get; }

    /// <summary>Penalty multiplier for repeated tokens. 1.0 disables penalty.</summary>
    float RepetitionPenalty { get; }

    /// <summary>Number of recent tokens considered for repetition penalty.</summary>
    int RepeatLastTokensCount { get; }

    /// <summary>Presence penalty applied additively to any token that has appeared at least once.</summary>
    float PresencePenalty { get; }

    /// <summary>Frequency penalty applied additively proportionally to token occurrence count.</summary>
    float FrequencyPenalty { get; }

    /// <summary>Optional additive logit bias dictionary mapping token IDs to bias values [-100, 100].</summary>
    IReadOnlyDictionary<int, float>? LogitBias { get; }

    /// <summary>Stop sequences (text strings) that halt generation when encountered.</summary>
    IReadOnlyList<string>? StopSequences { get; }

    /// <summary>Token IDs that halt generation, replacing the default end-of-generation set.</summary>
    IReadOnlyList<int>? StopTokens { get; }

    /// <summary>Additional token IDs that halt generation, unioned with the default end-of-generation set.</summary>
    IReadOnlyList<int>? AdditionalStopTokens { get; }

    /// <summary>Optional list of exact text choices to constrain generation.</summary>
    IReadOnlyList<string>? AllowedChoices { get; }

    /// <summary>
    /// Reasoning control: <c>null</c> follows model default, <c>true</c> enables thinking stream, <c>false</c> disables it.
    /// </summary>
    bool? EnableThinking { get; }

    /// <summary>Maximum thinking/reasoning tokens before closing the thinking block (0 = unlimited).</summary>
    int ThinkingBudget { get; }

    /// <summary>Speculative decoding strategy (Auto, None, Mtp).</summary>
    SpecType SpecType { get; }

    /// <summary>Maximum draft tokens per speculative step (0 = engine default).</summary>
    int SpecDraftNMax { get; }

    /// <summary>Optional structured output constraint (e.g. JSON schema or grammar masker).</summary>
    ITokenConstraint? Constraint { get; }
}

/// <summary>
/// Execution parameters controlling token generation, stopping criteria, sampling, and reasoning behavior.
/// </summary>
public record InferenceParams : IInferenceParams
{
    /// <summary>Maximum number of new tokens to generate. Default is 512.</summary>
    public int MaxTokens { get; init; } = 512;

    /// <summary>Optional RNG seed for deterministic sampling across generations.</summary>
    public int? Seed { get; init; }

    /// <summary>
    /// Optional chat-template render of the message history with add_generation_prompt=false.
    /// Used for canonical-history prefix caching across multi-turn conversations.
    /// </summary>
    public string? CanonicalHistoryPrefix { get; init; }

    /// <summary>Sampling temperature. 0.0 specifies deterministic greedy decoding. Default is 0.7.</summary>
    public float Temperature { get; init; } = 0.7f;

    /// <summary>Top-K sampling limit. 0 disables top-k truncation. Default is 40.</summary>
    public int TopK { get; init; } = 40;

    /// <summary>Top-P (nucleus) sampling threshold. 1.0 disables nucleus filtering. Default is 0.9.</summary>
    public float TopP { get; init; } = 0.9f;

    /// <summary>Min-P sampling threshold relative to the most likely token. Default is 0.0.</summary>
    public float MinP { get; init; } = 0.0f;

    /// <summary>Penalty multiplier for repeated tokens. 1.0 disables penalty. Default is 1.0.</summary>
    public float RepetitionPenalty { get; init; } = 1.0f;

    /// <summary>Number of recent tokens considered for repetition penalty. Default is 64.</summary>
    public int RepeatLastTokensCount { get; init; } = 64;

    /// <summary>Presence penalty applied additively to any token that has appeared at least once. Default is 0.0.</summary>
    public float PresencePenalty { get; init; } = 0.0f;

    /// <summary>Frequency penalty applied additively proportionally to token occurrence count. Default is 0.0.</summary>
    public float FrequencyPenalty { get; init; } = 0.0f;

    /// <summary>Optional additive logit bias dictionary mapping token IDs to bias values [-100, 100].</summary>
    public IReadOnlyDictionary<int, float>? LogitBias { get; init; }

    /// <summary>Stop sequences (text strings) that halt generation when encountered.</summary>
    public IReadOnlyList<string>? StopSequences { get; init; }

    /// <summary>Token IDs that halt generation, replacing the default end-of-generation set.</summary>
    public IReadOnlyList<int>? StopTokens { get; init; }

    /// <summary>Additional token IDs that halt generation, unioned with the default end-of-generation set.</summary>
    public IReadOnlyList<int>? AdditionalStopTokens { get; init; }

    /// <summary>Optional list of exact text choices to constrain generation.</summary>
    public IReadOnlyList<string>? AllowedChoices { get; init; }

    /// <summary>
    /// Reasoning control: <c>null</c> follows model default, <c>true</c> enables thinking stream, <c>false</c> disables it.
    /// </summary>
    public bool? EnableThinking { get; init; }

    /// <summary>Maximum thinking/reasoning tokens before closing the thinking block (0 = unlimited). Default is 0.</summary>
    public int ThinkingBudget { get; init; } = 0;

    /// <summary>Speculative decoding strategy (Auto, None, Mtp). Default is Auto.</summary>
    public SpecType SpecType { get; init; } = SpecType.Auto;

    /// <summary>Maximum draft tokens per speculative step (0 = engine default). Default is 0.</summary>
    public int SpecDraftNMax { get; init; } = 0;

    /// <summary>Optional structured output constraint (e.g. JSON schema or grammar masker).</summary>
    public ITokenConstraint? Constraint { get; init; }

    /// <summary>
    /// Converts this <see cref="InferenceParams"/> to an engine-internal <see cref="SamplingParams"/>.
    /// </summary>
    public SamplingParams ToSamplingParams()
    {
        return new SamplingParams
        {
            MaxNewTokens = MaxTokens,
            Seed = Seed,
            Temperature = Temperature,
            TopK = TopK,
            TopP = TopP,
            MinP = MinP,
            RepetitionPenalty = RepetitionPenalty,
            RepeatLastN = RepeatLastTokensCount,
            PresencePenalty = PresencePenalty,
            FrequencyPenalty = FrequencyPenalty,
            LogitBias = LogitBias,
            StopTokenIds = StopTokens is not null ? [.. StopTokens] : null,
            AdditionalStopTokenIds = AdditionalStopTokens is not null ? [.. AdditionalStopTokens] : null,
            AllowedChoices = AllowedChoices,
            ThinkingDisabled = EnableThinking == false,
            MaxThinkingTokens = ThinkingBudget,
            SpecType = SpecType,
            SpecDraftNMax = SpecDraftNMax,
            Constraint = Constraint,
        };
    }
}
