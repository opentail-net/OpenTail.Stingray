using System;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Immutable point-in-time snapshot of session inference and physical KV page usage metrics.
/// </summary>
public sealed record SessionMetricsSnapshot(
    long PromptTokens,
    long GeneratedTokens,
    TimeSpan TotalPrefillTime,
    TimeSpan TotalGenerationTime,
    double TokensPerSecond,
    int KvPagesHeld)
{
    /// <summary>
    /// Captures a static snapshot of the provided live <see cref="ISessionMetrics"/> instance.
    /// </summary>
    public static SessionMetricsSnapshot Capture(ISessionMetrics metrics)
    {
        ArgumentNullException.ThrowIfNull(metrics);
        return new SessionMetricsSnapshot(
            PromptTokens: metrics.PromptTokens,
            GeneratedTokens: metrics.GeneratedTokens,
            TotalPrefillTime: metrics.TotalPrefillTime,
            TotalGenerationTime: metrics.TotalGenerationTime,
            TokensPerSecond: metrics.TokensPerSecond,
            KvPagesHeld: metrics.KvPagesHeld);
    }
}
