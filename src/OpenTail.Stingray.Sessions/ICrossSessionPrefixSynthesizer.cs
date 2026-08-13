using System;
using System.Threading;
using System.Threading.Tasks;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Options configuring background cross-session prefix synthesis.
/// </summary>
public sealed class SynthesizerOptions
{
    /// <summary>Scan interval for background periodic discovery (default: 500 ms).</summary>
    public TimeSpan ScanInterval { get; init; } = TimeSpan.FromMilliseconds(500);

    /// <summary>Minimum token count for a synthesizable prefix (default: 16 tokens / 1 page).</summary>
    public int MinSynthesizedTokens { get; init; } = 16;

    /// <summary>Maximum sessions processed per background scan pass (default: 64).</summary>
    public int MaxSessionsPerScan { get; init; } = 64;

    /// <summary>Whether background auto-synthesis is enabled (default: true).</summary>
    public bool Enabled { get; init; } = true;
}

/// <summary>
/// Telemetry metrics snapshot for background prefix synthesis.
/// </summary>
public readonly record struct SynthesizerMetrics(
    long ScansCompleted,
    long SessionsScanned,
    long CandidatePrefixesDiscovered,
    long PublishedPrefixes,
    long PublishedPages,
    long SkippedUnstableSessions,
    long SkippedPartialPages);

/// <summary>
/// Abstraction for the background cross-session prefix synthesizer service.
/// Automatically discovers identical prompt prefixes across active sessions and publishes
/// their physical KV pages into IPrefixCacheIndex.
/// </summary>
public interface ICrossSessionPrefixSynthesizer : IAsyncDisposable
{
    /// <summary>Starts the background synthesis task.</summary>
    Task StartAsync(CancellationToken cancellationToken = default);

    /// <summary>Stops the background synthesis task.</summary>
    Task StopAsync(CancellationToken cancellationToken = default);

    /// <summary>Performs one synchronous/immediate synthesis scan pass over active sessions.</summary>
    int SynthesizeOnce();

    /// <summary>Current metrics snapshot.</summary>
    SynthesizerMetrics Metrics { get; }
}
