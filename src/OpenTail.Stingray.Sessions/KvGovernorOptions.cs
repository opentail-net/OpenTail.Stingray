using System;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Configuration options for the adaptive physical KV memory governor (<see cref="KvMemoryGovernor"/>).
/// <para>
/// Plan 016 implements <b>transient in-memory physical-KV reclamation</b>: under high memory pressure,
/// idle sessions release their physical KV cache pages back to the pool while retaining logical token history
/// in memory. Interacting with a suspended session transparently reconstructs KV state on demand.
/// Durable session persistence to disk remains strictly managed by <c>FileSessionStore</c>.
/// </para>
/// </summary>
public sealed record KvGovernorOptions
{
    /// <summary>Whether the governor is enabled. Defaults to true.</summary>
    public bool Enabled { get; init; } = true;

    /// <summary>KV utilization ratio above which idle session reclamation begins. Defaults to 85% (0.85).</summary>
    public double PressureThreshold { get; init; } = 0.85;

    /// <summary>Target KV utilization ratio to stop reclaiming when under pressure (hysteresis). Defaults to 70% (0.70).</summary>
    public double RecoveryThreshold { get; init; } = 0.70;

    /// <summary>Minimum age an idle session must reach to be eligible for suspension under pressure. Defaults to 30 seconds.</summary>
    public TimeSpan MinimumIdleDuration { get; init; } = TimeSpan.FromSeconds(30);

    /// <summary>Polling interval for background pressure monitoring. Defaults to 1 second.</summary>
    public TimeSpan PollInterval { get; init; } = TimeSpan.FromSeconds(1);

    /// <summary>Maximum number of idle sessions suspended per governor cycle. Defaults to 4.</summary>
    public int MaxSessionsSuspendedPerCycle { get; init; } = 4;

    /// <summary>Whether cold storage eviction (persisting suspended sessions to disk and clearing RAM token history) is enabled.</summary>
    public bool EnableColdTiering { get; init; } = true;

    /// <summary>KV utilization ratio above which suspended sessions are evicted to cold disk storage. Defaults to 85% (0.85).</summary>
    public double ColdPressureThreshold { get; init; } = 0.85;

    /// <summary>Persistent store used for cold-tiering. If null, a default FileSessionStore will be instantiated when cold tiering is needed.</summary>
    public ISessionStore? SessionStore { get; init; }
}
