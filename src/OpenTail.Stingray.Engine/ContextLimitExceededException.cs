
namespace OpenTail.Stingray.Engine;

/// <summary>
/// Exception thrown when an append or generation operation would exceed the configured maximum sequence context limit (<see cref="MaxContextTokens"/>).
/// </summary>
public sealed class ContextLimitExceededException : InvalidOperationException
{
    public ContextLimitExceededException(int currentTokens, int requestedTokens, int maxContextTokens)
        : base($"Session context limit exceeded: current token count ({currentTokens}) + requested tokens ({requestedTokens}) exceeds maximum context limit ({maxContextTokens}).")
    {
        CurrentTokens = currentTokens;
        RequestedTokens = requestedTokens;
        MaxContextTokens = maxContextTokens;
    }

    /// <summary>Current token count of the session before the attempted operation.</summary>
    public int CurrentTokens { get; }

    /// <summary>Number of additional tokens requested by the operation.</summary>
    public int RequestedTokens { get; }

    /// <summary>Configured maximum context token limit for the session.</summary>
    public int MaxContextTokens { get; }
}
