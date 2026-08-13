using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Adaptive, pressure-aware physical KV memory governor.
/// Automatically reclaims physical KV cache memory by suspending eligible idle sessions when memory is contested.
/// <para>
/// <b>Transient In-Memory Suspension:</b> Reclaims physical KV cache pages while preserving logical token history
/// in memory. Subsequent interaction with a suspended session transparently reconstructs KV state on demand.
/// Durable session persistence to disk remains strictly managed by <c>FileSessionStore</c>.
/// </para>
/// </summary>
public sealed class KvMemoryGovernor : IKvMemoryGovernor
{
    private readonly IKvCache _kvCache;
    private readonly Func<IReadOnlyList<IInferenceSession>> _sessionProvider;
    private readonly KvGovernorOptions _options;
    private readonly CancellationTokenSource _cts = new();
    private readonly Task _governorLoopTask;

    private long _governorCycles;
    private long _pressureEvents;
    private long _sessionsSuspended;
    private long _pagesReclaimed;
    private long _suspensionFailures;
    private long _skippedBusySessions;
    private bool _disposed;

    private readonly ISessionStore _sessionStore;

    public KvMemoryGovernor(
        IKvCache kvCache,
        Func<IReadOnlyList<IInferenceSession>> sessionProvider,
        KvGovernorOptions? options = null)
    {
        _kvCache = kvCache ?? throw new ArgumentNullException(nameof(kvCache));
        _sessionProvider = sessionProvider ?? throw new ArgumentNullException(nameof(sessionProvider));
        _options = options ?? new KvGovernorOptions();
        _sessionStore = _options.SessionStore ?? new FileSessionStore();

        if (_options.Enabled)
        {
            _governorLoopTask = Task.Run(() => GovernorLoopAsync(_cts.Token));
        }
        else
        {
            _governorLoopTask = Task.CompletedTask;
        }
    }

    public KvGovernorOptions Options => _options;

    public KvGovernorStatistics Statistics => new(
        GovernorCycles: Volatile.Read(ref _governorCycles),
        PressureEvents: Volatile.Read(ref _pressureEvents),
        SessionsSuspended: Volatile.Read(ref _sessionsSuspended),
        PagesReclaimed: Volatile.Read(ref _pagesReclaimed),
        SuspensionFailures: Volatile.Read(ref _suspensionFailures),
        SkippedBusySessions: Volatile.Read(ref _skippedBusySessions));

    public async Task ReclaimMemoryIfUnderPressureAsync(CancellationToken cancellationToken = default)
    {
        if (!_options.Enabled) return;

        Interlocked.Increment(ref _governorCycles);

        int totalPages = _kvCache.TotalPages;
        if (totalPages <= 0) return;

        int usedPages = _kvCache.UsedPages;
        double currentPressure = (double)usedPages / totalPages;

        var sessions = _sessionProvider();
        if (sessions == null || sessions.Count == 0) return;

        double minThreshold = _options.EnableColdTiering
            ? Math.Min(_options.PressureThreshold, _options.ColdPressureThreshold)
            : _options.PressureThreshold;

        bool hasSuspendedCandidates = _options.EnableColdTiering && sessions.Any(s => s.State == SessionState.Suspended);

        if (currentPressure < minThreshold && !hasSuspendedCandidates)
        {
            return; // Zero / low pressure: leave idle sessions resident
        }

        // Phase 1: Reclaim physical KV pages for idle Ready sessions (Ready -> Suspended)
        if (currentPressure >= _options.PressureThreshold)
        {
            Interlocked.Increment(ref _pressureEvents);

            var now = DateTimeOffset.UtcNow;
            var eligibleCandidates = new List<(IInferenceSession Session, double Score, int Pages)>();

            foreach (var s in sessions)
            {
                if (s.State != SessionState.Ready)
                {
                    if (s.State == SessionState.Generating)
                    {
                        Interlocked.Increment(ref _skippedBusySessions);
                    }
                    continue;
                }

                var idleDuration = now - s.LastActivityUtc;
                if (idleDuration < _options.MinimumIdleDuration)
                {
                    continue; // Not eligible yet
                }

                int pageCount = s.KvSequence?.PageCount ?? 0;
                double score = idleDuration.TotalSeconds * Math.Max(1, pageCount);
                eligibleCandidates.Add((s, score, pageCount));
            }

            if (eligibleCandidates.Count > 0)
            {
                // Rank by highest score (longest idle duration x largest physical memory footprint)
                eligibleCandidates.Sort((a, b) => b.Score.CompareTo(a.Score));

                int reclaimedThisCycle = 0;
                foreach (var candidate in eligibleCandidates)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    if (reclaimedThisCycle >= _options.MaxSessionsSuspendedPerCycle) break;

                    // Re-check pressure to enforce hysteresis
                    double activePressure = (double)_kvCache.UsedPages / _kvCache.TotalPages;
                    if (activePressure <= _options.RecoveryThreshold) break;

                    // Race guard: verify session is still in Ready state before suspending
                    if (candidate.Session.State != SessionState.Ready)
                    {
                        Interlocked.Increment(ref _skippedBusySessions);
                        continue;
                    }

                    try
                    {
                        int pagesBefore = candidate.Session.KvSequence?.PageCount ?? 0;
                        await candidate.Session.SuspendAsync(cancellationToken).ConfigureAwait(false);

                        Interlocked.Increment(ref _sessionsSuspended);
                        Interlocked.Add(ref _pagesReclaimed, pagesBefore);
                        reclaimedThisCycle++;
                    }
                    catch
                    {
                        Interlocked.Increment(ref _suspensionFailures);
                        // Non-fatal: governor continues evaluating next candidate
                    }
                }
            }
        }

        // Phase 2: Cold-tier idle Suspended sessions to disk (Suspended -> Cold) under sustained / high pressure
        double activePressureAfterPhase1 = (double)_kvCache.UsedPages / _kvCache.TotalPages;
        if (_options.EnableColdTiering && activePressureAfterPhase1 >= _options.ColdPressureThreshold)
        {
            var store = _sessionStore;
            var now = DateTimeOffset.UtcNow;
            var suspendedCandidates = new List<(IInferenceSession Session, TimeSpan Idle)>();

            foreach (var s in sessions)
            {
                if (s.State == SessionState.Suspended)
                {
                    var idleDuration = now - s.LastActivityUtc;
                    if (idleDuration >= _options.MinimumIdleDuration)
                    {
                        suspendedCandidates.Add((s, idleDuration));
                    }
                }
            }

            if (suspendedCandidates.Count > 0)
            {
                suspendedCandidates.Sort((a, b) => b.Idle.CompareTo(a.Idle)); // Longest idle first

                foreach (var candidate in suspendedCandidates)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    if (candidate.Session.State != SessionState.Suspended)
                    {
                        continue;
                    }

                    try
                    {
                        await candidate.Session.EvictToColdAsync(store, cancellationToken).ConfigureAwait(false);
                    }
                    catch
                    {
                        // Disk failure during cold eviction: non-fatal, session remains safely suspended in RAM
                    }
                }
            }
        }
    }

    private async Task GovernorLoopAsync(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(_options.PollInterval, cancellationToken).ConfigureAwait(false);
                await ReclaimMemoryIfUnderPressureAsync(cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch
            {
                // Non-fatal loop exception protection
            }
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;

        _cts.Cancel();
        try
        {
            await _governorLoopTask.ConfigureAwait(false);
        }
        catch
        {
            // Ignore cancellation on dispose
        }
        _cts.Dispose();
    }
}
