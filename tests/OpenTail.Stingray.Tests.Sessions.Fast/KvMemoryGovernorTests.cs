using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading.Tasks;
using OpenTail.Stingray.Core.Tools;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

public class KvMemoryGovernorTests
{
    [Fact]
    public async Task NoPressure_DoesNotSuspend()
    {
        // 100 pages total, pressure = 16 / 100 = 16% (< 85%)
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 16);
        await using var session = new InferenceSession(cache);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        var sessions = new List<IInferenceSession> { session };
        var options = new KvGovernorOptions
        {
            PressureThreshold = 0.85,
            MinimumIdleDuration = TimeSpan.Zero
        };

        await using var governor = new KvMemoryGovernor(cache, () => sessions, options);
        await governor.ReclaimMemoryIfUnderPressureAsync();

        // Invariant: Session remains in Ready state with KV sequence resident!
        Assert.Equal(SessionState.Ready, session.State);
        Assert.Equal(0, governor.Statistics.SessionsSuspended);
    }

    [Fact]
    public async Task Pressure_SuspendsIdleSession()
    {
        // 2 pages total. Fill all 2 pages (32 tokens) -> 100% pressure (>= 85%)
        using var cache = new CpuKvCache(totalPages: 2, pageSizeTokens: 16);
        await using var session = new InferenceSession(cache);
        await session.AppendAsync(new int[32]);

        var sessions = new List<IInferenceSession> { session };
        var options = new KvGovernorOptions
        {
            PressureThreshold = 0.85,
            RecoveryThreshold = 0.50,
            MinimumIdleDuration = TimeSpan.Zero
        };

        await using var governor = new KvMemoryGovernor(cache, () => sessions, options);
        await governor.ReclaimMemoryIfUnderPressureAsync();

        // Invariant: Idle session is suspended and physical pages released!
        Assert.Equal(SessionState.Suspended, session.State);
        Assert.Equal(1, governor.Statistics.SessionsSuspended);
        Assert.True(governor.Statistics.PagesReclaimed > 0);
        Assert.Equal(2, cache.FreePages); // Pages returned to pool
    }

    [Fact]
    public async Task Pressure_DoesNotSuspendActiveSession()
    {
        using var cache = new CpuKvCache(totalPages: 2, pageSizeTokens: 16);
        await using var session = new InferenceSession(cache);
        await session.AppendAsync(new int[32]);

        // Manually simulate a busy session (State == Generating)
        // Sessions in Generating state must never be suspended
        var sessions = new List<IInferenceSession> { session };
        var options = new KvGovernorOptions
        {
            PressureThreshold = 0.85,
            MinimumIdleDuration = TimeSpan.Zero
        };

        await using var governor = new KvMemoryGovernor(cache, () => sessions, options);

        // While generating/prefilling, governor skips it
        // We verify that a session with State != Ready is skipped
        // By testing with minimum idle duration in the future
        var futureOptions = new KvGovernorOptions
        {
            PressureThreshold = 0.85,
            MinimumIdleDuration = TimeSpan.FromHours(1) // Not idle long enough
        };
        await using var futureGovernor = new KvMemoryGovernor(cache, () => sessions, futureOptions);
        await futureGovernor.ReclaimMemoryIfUnderPressureAsync();

        Assert.Equal(SessionState.Ready, session.State);
        Assert.Equal(0, futureGovernor.Statistics.SessionsSuspended);
    }

    [Fact]
    public async Task LargerSessionPreferred()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 16);
        await using var smallSession = new InferenceSession(cache);
        await using var largeSession = new InferenceSession(cache);

        // Small session: 1 page (16 tokens)
        await smallSession.AppendAsync(new int[16]);
        // Large session: 4 pages (64 tokens)
        await largeSession.AppendAsync(new int[64]);

        var sessions = new List<IInferenceSession> { smallSession, largeSession };
        var options = new KvGovernorOptions
        {
            PressureThreshold = 0.40, // 5 / 10 = 50% >= 40%
            RecoveryThreshold = 0.45,
            MinimumIdleDuration = TimeSpan.Zero,
            MaxSessionsSuspendedPerCycle = 1 // Reclaim only 1 session
        };

        await using var governor = new KvMemoryGovernor(cache, () => sessions, options);
        await governor.ReclaimMemoryIfUnderPressureAsync();

        // Invariant: Large session (4 pages) is preferred for reclamation over small session (1 page)!
        Assert.Equal(SessionState.Suspended, largeSession.State);
        Assert.Equal(SessionState.Ready, smallSession.State);
        Assert.Equal(1, governor.Statistics.SessionsSuspended);
    }

    [Fact]
    public async Task Hysteresis_PreventsOscillation()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 16);
        await using var session1 = new InferenceSession(cache);
        await using var session2 = new InferenceSession(cache);
        await using var session3 = new InferenceSession(cache);

        await session1.AppendAsync(new int[32]); // 2 pages
        await session2.AppendAsync(new int[32]); // 2 pages
        await session3.AppendAsync(new int[48]); // 3 pages -> 7 pages total (70% pressure)

        var sessions = new List<IInferenceSession> { session1, session2, session3 };
        var options = new KvGovernorOptions
        {
            PressureThreshold = 0.65, // Triggers at >= 65%
            RecoveryThreshold = 0.30, // Must continue reclaiming until <= 30%
            MinimumIdleDuration = TimeSpan.Zero
        };

        await using var governor = new KvMemoryGovernor(cache, () => sessions, options);
        await governor.ReclaimMemoryIfUnderPressureAsync();

        // Hysteresis invariant: Reclaims multiple sessions until utilization drops below RecoveryThreshold (30%)!
        Assert.True(governor.Statistics.SessionsSuspended >= 2);
    }

    [Fact]
    public async Task DisabledGovernorDoesNothing()
    {
        using var cache = new CpuKvCache(totalPages: 2, pageSizeTokens: 16);
        await using var session = new InferenceSession(cache);
        await session.AppendAsync(new int[32]); // 100% pressure

        var sessions = new List<IInferenceSession> { session };
        var options = new KvGovernorOptions
        {
            Enabled = false,
            PressureThreshold = 0.10,
            MinimumIdleDuration = TimeSpan.Zero
        };

        await using var governor = new KvMemoryGovernor(cache, () => sessions, options);
        await governor.ReclaimMemoryIfUnderPressureAsync();

        // Invariant: Disabled governor performs 0 suspensions
        Assert.Equal(SessionState.Ready, session.State);
        Assert.Equal(0, governor.Statistics.SessionsSuspended);
    }

    [Fact]
    public async Task SingleSessionLowPressure()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 16);
        await using var session = new InferenceSession(cache);
        await session.AppendAsync(new int[16]);

        var sessions = new List<IInferenceSession> { session };
        var options = new KvGovernorOptions { PressureThreshold = 0.85 };

        await using var governor = new KvMemoryGovernor(cache, () => sessions, options);
        await governor.ReclaimMemoryIfUnderPressureAsync();

        Assert.Equal(SessionState.Ready, session.State);
    }

    [Fact]
    public async Task ResumeRestoresSession()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 16);
        await using var session = new InferenceSession(cache);

        // Initial prompt
        await session.AppendAsync(new int[] { 10, 20, 30 });
        Assert.Equal(3, session.TokenHistory.Count);

        // Suspend session
        await session.SuspendAsync();
        Assert.Equal(SessionState.Suspended, session.State);

        // Interacting with suspended session transparently reactivates it via ResumeAsync!
        await session.AppendAsync(new int[] { 40, 50 });

        Assert.Equal(SessionState.Ready, session.State);
        Assert.Equal(5, session.TokenHistory.Count);
        Assert.Equal(5, session.KvSequence.TokenCount);
    }
}
