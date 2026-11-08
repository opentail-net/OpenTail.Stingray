namespace OpenTail.Stingray.Engine;

/// <summary>
/// Thrown when a request arrives while the engine's waiting queue is at its limit (see the
/// <c>maxQueuedRequests</c> parameter of <see cref="ContinuousBatchingEngine"/>). The request is refused, not held.
/// Mapping this to a server status (for example 429 or 503) is left to the host.
/// </summary>
public sealed class EngineQueueFullException(string modelId, int limit)
    : InvalidOperationException($"The request queue for model '{modelId}' is full (limit {limit} waiting requests).")
{
    /// <summary>Model whose queue refused the request.</summary>
    public string ModelId { get; } = modelId;

    /// <summary>The configured limit that was reached.</summary>
    public int Limit { get; } = limit;
}
