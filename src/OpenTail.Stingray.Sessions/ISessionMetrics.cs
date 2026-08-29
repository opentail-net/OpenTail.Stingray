
namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Read-only metrics surface exposing per-session inference and physical KV page usage statistics.
/// </summary>
public interface ISessionMetrics
{
    /// <summary>Total successfully committed prompt/prefill tokens (including tool result appends).</summary>
    long PromptTokens { get; }

    /// <summary>Total committed model-generated output tokens (discarded speculative tokens are excluded).</summary>
    long GeneratedTokens { get; }

    /// <summary>Cumulative wall-clock time spent performing prompt/prefill operations.</summary>
    TimeSpan TotalPrefillTime { get; }

    /// <summary>Cumulative wall-clock time spent performing token generation operations.</summary>
    TimeSpan TotalGenerationTime { get; }

    /// <summary>Cumulative output throughput rate in generated tokens per second. Returns 0.0 if no generation has occurred.</summary>
    double TokensPerSecond { get; }

    /// <summary>Number of physical KV pages currently retained by this session. Returns 0 when suspended.</summary>
    int KvPagesHeld { get; }
}
