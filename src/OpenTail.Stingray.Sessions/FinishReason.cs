namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Outcome condition for a model generation operation.
/// </summary>
public enum FinishReason
{
    /// <summary>Model generated output up to a natural end-of-sequence / stop token.</summary>
    Completed,

    /// <summary>Generation reached the configured <c>SamplingParams.MaxNewTokens</c> limit.</summary>
    MaxTokens,

    /// <summary>Generation halted because the model emitted one or more structured tool calls.</summary>
    ToolCall,

    /// <summary>Generation stopped because the session's maximum context length was reached.</summary>
    ContextLimit,

    /// <summary>Generation was prematurely cancelled via a <see cref="System.Threading.CancellationToken"/>.</summary>
    Cancelled
}
