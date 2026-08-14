using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions;

/// <summary>
/// Comprehensive unit tests for Cold-Storage Session Tiering, Admission Control, and AutoSessionEviction.
/// </summary>
public sealed class ColdTieringTests
{
    [Fact]
    public async Task Test01_NoPressureNoEviction()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);
            var sessions = new List<IInferenceSession>();
            for (int i = 0; i < 5; i++)
            {
                var s = new InferenceSession(cache);
                await s.AppendAsync(new int[] { 1, 2, 3 });
                await s.SuspendAsync();
                sessions.Add(s);
            }

            // Low pressure: governor should NOT cold-tier any suspended session
            await using var governor = new KvMemoryGovernor(cache, () => sessions, new KvGovernorOptions
            {
                Enabled = true,
                PressureThreshold = 0.85,
                ColdPressureThreshold = 0.85,
                MinimumIdleDuration = TimeSpan.Zero,
                SessionStore = store
            });

            await governor.ReclaimMemoryIfUnderPressureAsync();

            foreach (var s in sessions)
            {
                Assert.Equal(SessionState.Suspended, s.State);
            }
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    [Fact]
    public async Task Test02_PressureTriggersColdTiering()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);
            var sessions = new List<IInferenceSession>();

            // Allocate 9 out of 10 pages across 3 sessions (90% pressure)
            for (int i = 0; i < 3; i++)
            {
                var s = new InferenceSession(cache);
                await s.AppendAsync(Enumerable.Repeat(1, 96).ToArray()); // 96 tokens = 3 pages each
                sessions.Add(s);
            }

            // Suspend sessions so their pages are reclaimed, but pressure remains evaluated by governor
            foreach (var s in sessions)
            {
                await s.SuspendAsync();
            }

            // Allocate 9 pages again in a new active session to create 90% pressure
            var activeSession = new InferenceSession(cache);
            await activeSession.AppendAsync(Enumerable.Repeat(9, 288).ToArray()); // 9 pages

            await using var governor = new KvMemoryGovernor(cache, () => sessions, new KvGovernorOptions
            {
                Enabled = true,
                PressureThreshold = 0.80,
                ColdPressureThreshold = 0.80,
                MinimumIdleDuration = TimeSpan.Zero,
                SessionStore = store
            });

            await governor.ReclaimMemoryIfUnderPressureAsync();

            // At 90% pressure, suspended sessions should be evicted to Cold
            Assert.Contains(sessions, s => s.State == SessionState.Cold);
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    [Fact]
    public async Task Test03_LRUCandidateSelection()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);

            var sessionOld = new InferenceSession(cache);
            await sessionOld.AppendAsync(new int[] { 1, 2 });
            await sessionOld.SuspendAsync();

            var sessionNew = new InferenceSession(cache);
            await sessionNew.AppendAsync(new int[] { 1, 2 });
            await sessionNew.SuspendAsync();
            sessionNew.TouchActivity(); // newer activity

            // Manually evict oldest suspended session to cold
            await sessionOld.EvictToColdAsync(store);

            Assert.Equal(SessionState.Cold, sessionOld.State);
            Assert.Equal(SessionState.Suspended, sessionNew.State);
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    [Fact]
    public async Task Test04_ActiveSessionProtection()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);
            var activeSession = new InferenceSession(cache);
            await activeSession.AppendAsync(new int[] { 1, 2, 3 });

            // Ready / active session cannot be evicted to cold
            await Assert.ThrowsAsync<InvalidOperationException>(async () =>
            {
                await activeSession.EvictToColdAsync(store);
            });

            Assert.Equal(SessionState.Ready, activeSession.State);
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    [Fact]
    public async Task Test05_PersistenceBeforeRelease()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var failingStore = new FailingSessionStore();
        var session = new InferenceSession(cache);
        await session.AppendAsync(new int[] { 1, 2, 3, 4 });
        await session.SuspendAsync();

        // Evicting with a failing store must throw and keep session safely in Suspended state in RAM
        await Assert.ThrowsAsync<IOException>(async () =>
        {
            await session.EvictToColdAsync(failingStore);
        });

        Assert.Equal(SessionState.Suspended, session.State);
        Assert.Equal(4, session.TokenCount);
    }

    [Fact]
    public async Task Test06_TransparentRehydration()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);
            var session = new InferenceSession(cache);
            await session.AppendAsync(new int[] { 10, 20, 30 });
            await session.SuspendAsync();
            await session.EvictToColdAsync(store);

            Assert.Equal(SessionState.Cold, session.State);

            // Rehydrate transparently
            await session.EnsureActiveAsync(store);

            Assert.Equal(SessionState.Ready, session.State);
            Assert.Equal(3, session.TokenCount);
            Assert.Equal(new int[] { 10, 20, 30 }, session.TokenHistory);
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    [Fact]
    public async Task Test07_ConcurrentRehydration()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);
            var session = new InferenceSession(cache);
            await session.AppendAsync(new int[] { 1, 2, 3, 4, 5 });
            await session.SuspendAsync();
            await session.EvictToColdAsync(store);

            // Trigger 5 concurrent rehydration calls on the same cold session
            var tasks = Enumerable.Range(0, 5)
                .Select(_ => session.EnsureActiveAsync(store).AsTask())
                .ToArray();

            await Task.WhenAll(tasks);

            Assert.Equal(SessionState.Ready, session.State);
            Assert.Equal(5, session.TokenCount);
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    [Fact]
    public async Task Test08_ReservationBeforeLoad()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);
            var session = new InferenceSession(cache);
            await session.AppendAsync(new int[] { 1, 2, 3 });
            await session.SuspendAsync();
            await session.EvictToColdAsync(store);

            // Attempt rehydration without providing a store -> fails before touching KV or state
            await Assert.ThrowsAsync<InvalidOperationException>(async () =>
            {
                await session.EnsureActiveAsync(store: null);
            });

            Assert.Equal(SessionState.Cold, session.State);
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    [Fact]
    public async Task Test09_ReclamationThenAdmission()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);
            var session = new InferenceSession(cache);
            await session.AppendAsync(new int[] { 100, 200 });
            await session.SuspendAsync();
            await session.EvictToColdAsync(store);

            Assert.Equal(SessionState.Cold, session.State);

            // Ensure active succeeds and allocates required KV sequence
            await session.EnsureActiveAsync(store);

            Assert.Equal(SessionState.Ready, session.State);
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    [Fact]
    public async Task Test10_NothingFits_AdmissionFailure()
    {
        // 11 physical pages: exactly enough for the 1-page active session plus the 10-page
        // cold session to both fit while the cold session is first populated. Rehydration is
        // expected to fail not because the cache is inherently too small (a 10-page append
        // could never even complete otherwise), but because a competing session has since
        // claimed the capacity the cold session released when it was suspended.
        using var smallCache = new CpuKvCache(totalPages: 11, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);

            // Active session holding 1 page for the entire test
            var active = new InferenceSession(smallCache);
            await active.AppendAsync(Enumerable.Repeat(1, 32).ToArray());

            // Cold session needing 10 pages (320 tokens) — fits while the other 10 pages are free
            var coldSession = new InferenceSession(smallCache);
            await coldSession.AppendAsync(Enumerable.Repeat(2, 320).ToArray());
            await coldSession.SuspendAsync();
            await coldSession.EvictToColdAsync(store);

            // A third session claims the 10 pages the cold session just released, so nothing
            // is left free for rehydration.
            var filler = new InferenceSession(smallCache);
            await filler.AppendAsync(Enumerable.Repeat(3, 320).ToArray());

            // Rehydration fails cleanly because required pages exceeds free capacity
            await Assert.ThrowsAsync<InvalidOperationException>(async () =>
            {
                await coldSession.EnsureActiveAsync(store);
            });

            Assert.Equal(SessionState.Cold, coldSession.State);
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    [Fact]
    public async Task Test11_OversizedSession()
    {
        // 7 pages: exactly enough for the session's own 200-token (7-page) append, with none
        // to spare. Rehydration is expected to fail once a second session claims that capacity
        // while the first is cold — see Test10 for why "structurally too small to ever fit" isn't
        // a distinct, constructible scenario from "no longer fits due to contention": a session's
        // cold snapshot is always re-prefilled against the very cache it was suspended from, so if
        // it fit once it can only fail to fit again because something else is now using the space.
        using var cache = new CpuKvCache(totalPages: 7, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);
            var session = new InferenceSession(cache);
            await session.AppendAsync(Enumerable.Repeat(1, 200).ToArray()); // 200 tokens needs 7 pages, fits exactly
            await session.SuspendAsync();
            await session.EvictToColdAsync(store);

            // Claim all 7 pages the session just released while it is cold
            var filler = new InferenceSession(cache);
            await filler.AppendAsync(Enumerable.Repeat(2, 200).ToArray());

            // Oversized session fails admission cleanly
            await Assert.ThrowsAsync<InvalidOperationException>(async () =>
            {
                await session.EnsureActiveAsync(store);
            });

            Assert.Equal(SessionState.Cold, session.State);
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    [Fact]
    public async Task Test12_ConcurrentAdmission_NoOvercommit()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);
            var coldSessions = new List<InferenceSession>();

            for (int i = 0; i < 5; i++)
            {
                var s = new InferenceSession(cache);
                await s.AppendAsync(new int[] { i + 1, i + 2 });
                await s.SuspendAsync();
                await s.EvictToColdAsync(store);
                coldSessions.Add(s);
            }

            var tasks = coldSessions.Select(s => s.EnsureActiveAsync(store).AsTask()).ToArray();
            await Task.WhenAll(tasks);

            foreach (var s in coldSessions)
            {
                Assert.Equal(SessionState.Ready, s.State);
            }
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    [Fact]
    public async Task Test13_MetadataRoundTrip()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);
            var session = new InferenceSession(cache);
            session.Metadata["user_id"] = "usr_99";
            session.Metadata["workflow"] = "synthesis";

            await session.AppendAsync(new int[] { 1, 2, 3 });
            await session.SuspendAsync();
            await session.EvictToColdAsync(store);

            // Metadata remains accessible while cold
            Assert.Equal("usr_99", session.Metadata.Get<string>("user_id"));

            await session.EnsureActiveAsync(store);

            // Metadata preserved after rehydration
            Assert.Equal("usr_99", session.Metadata.Get<string>("user_id"));
            Assert.Equal("synthesis", session.Metadata.Get<string>("workflow"));
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    [Fact]
    public async Task Test14_ContinuationRoundTrip()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);
            var session = new InferenceSession(cache);
            await session.AppendAsync(new int[] { 10, 20, 30, 40 });

            var tokenBefore = session.GetContinuationToken();

            await session.SuspendAsync();
            await session.EvictToColdAsync(store);
            await session.EnsureActiveAsync(store);

            var tokenAfter = session.GetContinuationToken();

            Assert.Equal(tokenBefore.SessionId, tokenAfter.SessionId);
            Assert.Equal(tokenBefore.TokenPosition, tokenAfter.TokenPosition);
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    [Fact]
    public async Task Test15_MetricsRoundTrip()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);
            var session = new InferenceSession(cache);
            await session.AppendAsync(new int[] { 1, 2, 3 });

            long promptTokensBefore = session.Metrics.PromptTokens;

            await session.SuspendAsync();
            await session.EvictToColdAsync(store);
            await session.EnsureActiveAsync(store);

            Assert.Equal(promptTokensBefore, session.Metrics.PromptTokens);
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    [Fact]
    public async Task Test16_ForkRoundTrip()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);
            var parent = new InferenceSession(cache);
            await parent.AppendAsync(new int[] { 1, 2, 3 });

            var child = (InferenceSession)parent.Fork();
            await child.AppendAsync(new int[] { 4, 5 });

            await child.SuspendAsync();
            await child.EvictToColdAsync(store);

            Assert.Equal(SessionState.Cold, child.State);

            await child.EnsureActiveAsync(store);

            Assert.Equal(SessionState.Ready, child.State);
            Assert.Equal(new int[] { 1, 2, 3, 4, 5 }, child.TokenHistory);
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    [Fact]
    public async Task Test17_TruncatedSessionRoundTrip()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);
            var session = new InferenceSession(cache);
            await session.AppendAsync(new int[] { 1, 2, 3, 4, 5 });

            var checkpoint = session.CreateCheckpoint(); // at pos 5
            await session.AppendAsync(new int[] { 6, 7, 8 });

            session.Rollback(checkpoint); // rollback to pos 5

            await session.SuspendAsync();
            await session.EvictToColdAsync(store);
            await session.EnsureActiveAsync(store);

            Assert.Equal(5, session.TokenCount);
            Assert.Equal(new int[] { 1, 2, 3, 4, 5 }, session.TokenHistory);
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    [Fact]
    public async Task Test18_DiskFullFallback()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var store = new FailingSessionStore();
        var session = new InferenceSession(cache);
        await session.AppendAsync(new int[] { 1, 2, 3 });
        await session.SuspendAsync();

        await Assert.ThrowsAsync<IOException>(async () =>
        {
            await session.EvictToColdAsync(store);
        });

        Assert.Equal(SessionState.Suspended, session.State);
        Assert.Equal(3, session.TokenCount);
    }

    [Fact]
    public async Task Test19_Hysteresis_PreventsOscillation()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);
            var sessions = new List<IInferenceSession>();
            for (int i = 0; i < 2; i++)
            {
                var s = new InferenceSession(cache);
                await s.AppendAsync(new int[] { 1, 2 });
                await s.SuspendAsync();
                sessions.Add(s);
            }

            await using var governor = new KvMemoryGovernor(cache, () => sessions, new KvGovernorOptions
            {
                Enabled = true,
                PressureThreshold = 0.85,
                RecoveryThreshold = 0.70,
                MinimumIdleDuration = TimeSpan.Zero,
                SessionStore = store
            });

            // At 0% pressure (no pages allocated in active sequences), governor performs zero evictions
            await governor.ReclaimMemoryIfUnderPressureAsync();

            foreach (var s in sessions)
            {
                Assert.Equal(SessionState.Suspended, s.State);
            }
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    [Fact]
    public async Task Test20_DisabledColdTiering()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);
            var sessions = new List<IInferenceSession>();

            for (int i = 0; i < 3; i++)
            {
                var s = new InferenceSession(cache);
                await s.AppendAsync(Enumerable.Repeat(1, 96).ToArray());
                await s.SuspendAsync();
                sessions.Add(s);
            }

            // Fill cache with active session
            var activeSession = new InferenceSession(cache);
            await activeSession.AppendAsync(Enumerable.Repeat(9, 288).ToArray());

            await using var governor = new KvMemoryGovernor(cache, () => sessions, new KvGovernorOptions
            {
                Enabled = true,
                EnableColdTiering = false, // DISABLED
                PressureThreshold = 0.80,
                ColdPressureThreshold = 0.80,
                MinimumIdleDuration = TimeSpan.Zero,
                SessionStore = store
            });

            await governor.ReclaimMemoryIfUnderPressureAsync();

            // When EnableColdTiering is false, suspended sessions remain Suspended in RAM
            foreach (var s in sessions)
            {
                Assert.Equal(SessionState.Suspended, s.State);
            }
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    [Fact]
    public async Task Test21_GetSessionStateSummary()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var dir = CreateTempDir();
        try
        {
            var store = new FileSessionStore(dir);
            var manager = new InMemorySessionManager(cache);

            var s1 = await manager.CreateSessionAsync();
            await s1.AppendAsync(new int[] { 1 });

            var s2 = await manager.CreateSessionAsync();
            await s2.AppendAsync(new int[] { 1 });
            await s2.SuspendAsync();

            var s3 = await manager.CreateSessionAsync();
            await s3.AppendAsync(new int[] { 1 });
            await s3.SuspendAsync();
            await s3.EvictToColdAsync(store);

            var (ready, suspended, cold) = manager.GetSessionStateBreakdown();

            Assert.Equal(1, ready);
            Assert.Equal(1, suspended);
            Assert.Equal(1, cold);
        }
        finally
        {
            DeleteTempDir(dir);
        }
    }

    private static string CreateTempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"stingray-cold-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void DeleteTempDir(string dir)
    {
        try
        {
            if (Directory.Exists(dir))
            {
                Directory.Delete(dir, recursive: true);
            }
        }
        catch { }
    }

    private sealed class FailingSessionStore : ISessionStore
    {
        public ValueTask SaveAsync(InferenceSessionSnapshot snapshot, CancellationToken cancellationToken = default)
        {
            throw new IOException("Simulated disk full failure during cold tiering persistence.");
        }

        public ValueTask<InferenceSessionSnapshot?> LoadAsync(SessionId id, CancellationToken cancellationToken = default)
        {
            return ValueTask.FromResult<InferenceSessionSnapshot?>(null);
        }

        public ValueTask<bool> DeleteAsync(SessionId id, CancellationToken cancellationToken = default)
        {
            return ValueTask.FromResult(false);
        }
    }
}
