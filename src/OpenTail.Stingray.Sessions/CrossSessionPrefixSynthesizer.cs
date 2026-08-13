using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Background service that automatically synthesizes common committed prompt prefixes
/// across active sessions into IPrefixCacheIndex / RadixPrefixTree.
/// </summary>
public sealed class CrossSessionPrefixSynthesizer : ICrossSessionPrefixSynthesizer
{
    private readonly IPrefixCacheIndex _index;
    private readonly IActiveSessionRegistry _registry;
    private readonly SynthesizerOptions _options;

    private CancellationTokenSource? _cts;
    private Task? _backgroundTask;
    private bool _disposed;

    private long _scansCompleted;
    private long _sessionsScanned;
    private long _candidatePrefixesDiscovered;
    private long _publishedPrefixes;
    private long _publishedPages;
    private long _skippedUnstableSessions;
    private long _skippedPartialPages;

    public CrossSessionPrefixSynthesizer(
        IPrefixCacheIndex index,
        IActiveSessionRegistry registry,
        SynthesizerOptions? options = null)
    {
        _index = index ?? throw new ArgumentNullException(nameof(index));
        _registry = registry ?? throw new ArgumentNullException(nameof(registry));
        _options = options ?? new SynthesizerOptions();
    }

    public SynthesizerMetrics Metrics => new(
        _scansCompleted,
        _sessionsScanned,
        _candidatePrefixesDiscovered,
        _publishedPrefixes,
        _publishedPages,
        _skippedUnstableSessions,
        _skippedPartialPages);

    public Task StartAsync(CancellationToken cancellationToken = default)
    {
        if (_disposed || !_options.Enabled) return Task.CompletedTask;
        if (_backgroundTask is not null) return Task.CompletedTask;

        _cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        _backgroundTask = Task.Run(() => RunLoopAsync(_cts.Token), CancellationToken.None);
        return Task.CompletedTask;
    }

    public async Task StopAsync(CancellationToken cancellationToken = default)
    {
        if (_cts is null || _backgroundTask is null) return;

        _cts.Cancel();
        try
        {
            await _backgroundTask.ConfigureAwait(false);
        }
        catch (OperationCanceledException) { }
        finally
        {
            _cts.Dispose();
            _cts = null;
            _backgroundTask = null;
        }
    }

    public int SynthesizeOnce()
    {
        if (_disposed) return 0;

        Interlocked.Increment(ref _scansCompleted);
        var activeSessions = _registry.GetActiveSessionsSnapshot();
        if (activeSessions.Count == 0) return 0;

        int publishedThisScan = 0;
        int limit = Math.Min(activeSessions.Count, _options.MaxSessionsPerScan);

        for (int i = 0; i < limit; i++)
        {
            var session = activeSessions[i];
            Interlocked.Increment(ref _sessionsScanned);

            if (session.State == SessionState.Disposed || session.State == SessionState.Faulted)
            {
                Interlocked.Increment(ref _skippedUnstableSessions);
                continue;
            }

            try
            {
                var history = session.TokenHistory;
                var sequence = session.KvSequence;
                if (history is null || sequence is null)
                {
                    Interlocked.Increment(ref _skippedUnstableSessions);
                    continue;
                }

                int pageSize = sequence.PageSize;
                if (pageSize <= 0)
                {
                    Interlocked.Increment(ref _skippedUnstableSessions);
                    continue;
                }

                int totalTokens = history.Count;
                int completePageCount = totalTokens / pageSize;
                int pageAlignedCount = completePageCount * pageSize;

                if (pageAlignedCount < _options.MinSynthesizedTokens)
                {
                    Interlocked.Increment(ref _skippedPartialPages);
                    continue;
                }

                // Namespace by the session's actual model (and a cheap proxy for KV physical
                // layout, page size) instead of a shared constant -- a hardcoded namespace let
                // sessions running different models be treated as prefix-compatible and share
                // physical KV pages through Publish/MatchPrefix, exactly what this namespace type
                // exists to prevent.
                var ns = new PrefixCacheNamespace(session.ModelId, $"page{pageSize}");
                int[] prefixArray = new int[pageAlignedCount];
                for (int t = 0; t < pageAlignedCount; t++)
                {
                    prefixArray[t] = history[t];
                }

                Interlocked.Increment(ref _candidatePrefixesDiscovered);
                _index.Publish(ns, prefixArray, sequence, pageAlignedCount);

                Interlocked.Increment(ref _publishedPrefixes);
                Interlocked.Add(ref _publishedPages, completePageCount);
                publishedThisScan++;
            }
            catch
            {
                Interlocked.Increment(ref _skippedUnstableSessions);
            }
        }

        return publishedThisScan;
    }

    private async Task RunLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                SynthesizeOnce();
                await Task.Delay(_options.ScanInterval, ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch
            {
                // Background scan resilience — continue loop
            }
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;
        await StopAsync().ConfigureAwait(false);
    }
}
