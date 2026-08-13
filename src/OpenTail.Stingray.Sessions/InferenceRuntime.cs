using System;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

using OpenTail.Stingray.Core.Capabilities;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Concrete implementation of <see cref="IInferenceRuntime"/> managing the physical
/// <see cref="IKvCache"/> allocator and the active <see cref="InMemorySessionManager"/> registry.
/// </summary>
public sealed class InferenceRuntime : IInferenceRuntime
{
    private readonly bool _ownsKvCache;
    private bool _disposed;

    public InferenceRuntime(int totalPages = 1024, int pageSizeTokens = 32, IForwardPass? forwardPass = null, InferenceCapabilities? capabilityOverride = null, KvGovernorOptions? governorOptions = null)
    {
        var kvCache = new CpuKvCache(totalPages, pageSizeTokens);
        KvCache = kvCache;
        PrefixIndex = new RadixPrefixTree(kvCache);
        SessionManager = new InMemorySessionManager(kvCache);
        Capabilities = capabilityOverride ?? InferenceCapabilityBuilder.Build(hasKvCache: true, forwardPass: forwardPass);
        var effectiveOptions = governorOptions ?? new KvGovernorOptions { Enabled = false };
        MemoryGovernor = new KvMemoryGovernor(kvCache, () => SessionManager.GetActiveSessionsSnapshot(), effectiveOptions);
        if (PrefixIndex is not null)
        {
            Synthesizer = new CrossSessionPrefixSynthesizer(PrefixIndex, SessionManager);
        }
        _ownsKvCache = true;
    }

    public InferenceRuntime(IKvCache kvCache, IForwardPass? forwardPass = null, InferenceCapabilities? capabilityOverride = null, KvGovernorOptions? governorOptions = null)
    {
        KvCache = kvCache ?? throw new ArgumentNullException(nameof(kvCache));
        PrefixIndex = kvCache is CpuKvCache cpuCache ? new RadixPrefixTree(cpuCache) : null;
        SessionManager = new InMemorySessionManager(KvCache);
        Capabilities = capabilityOverride ?? InferenceCapabilityBuilder.Build(hasKvCache: true, forwardPass: forwardPass);
        var effectiveOptions = governorOptions ?? new KvGovernorOptions { Enabled = false };
        MemoryGovernor = new KvMemoryGovernor(KvCache, () => SessionManager.GetActiveSessionsSnapshot(), effectiveOptions);
        if (PrefixIndex is not null)
        {
            Synthesizer = new CrossSessionPrefixSynthesizer(PrefixIndex, SessionManager);
        }
        _ownsKvCache = false;
    }

    public IKvCache KvCache { get; }
    public InMemorySessionManager SessionManager { get; }
    public InferenceCapabilities Capabilities { get; }
    public ModelCapabilities ModelCapabilities => Capabilities.Model;
    public IPrefixCacheIndex? PrefixIndex { get; }
    public IKvMemoryGovernor MemoryGovernor { get; }
    public ICrossSessionPrefixSynthesizer? Synthesizer { get; }

    public async ValueTask<IInferenceSession> CreateSessionAsync(
        KvSequenceOptions? options = null,
        IForwardPass? forwardPass = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        if (KvCache.TotalPages > 0 && ((double)KvCache.UsedPages / KvCache.TotalPages) >= MemoryGovernor.Options.PressureThreshold)
        {
            await MemoryGovernor.ReclaimMemoryIfUnderPressureAsync(cancellationToken).ConfigureAwait(false);
        }

        return await SessionManager.CreateSessionAsync(options, forwardPass, cancellationToken).ConfigureAwait(false);
    }

    public IInferenceSession? GetSession(SessionId id)
    {
        ThrowIfDisposed();
        return SessionManager.GetSession(id);
    }

    public ValueTask<bool> RemoveSessionAsync(SessionId id)
    {
        ThrowIfDisposed();
        return SessionManager.RemoveSessionAsync(id);
    }

    public async ValueTask<int> PruneBranchTreeAsync(SessionId rootId, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        var session = GetSession(rootId);
        if (session is null)
        {
            return 0;
        }

        var descendants = new List<IInferenceSession>();
        var queue = new Queue<IInferenceSession>();
        foreach (var childId in session.Tree.Children)
        {
            var child = GetSession(childId);
            if (child is not null) queue.Enqueue(child);
        }

        var visited = new HashSet<SessionId> { rootId };

        while (queue.Count > 0)
        {
            var child = queue.Dequeue();
            if (visited.Add(child.Id))
            {
                descendants.Add(child);
                foreach (var gId in child.Tree.Children)
                {
                    if (!visited.Contains(gId))
                    {
                        var g = GetSession(gId);
                        if (g is not null) queue.Enqueue(g);
                    }
                }
            }
        }

        int count = 0;
        foreach (var desc in descendants)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await RemoveSessionAsync(desc.Id).ConfigureAwait(false);
            await desc.DisposeAsync().ConfigureAwait(false);
            count++;
        }

        return count;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(InferenceRuntime));
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;

        // Stop the background governor loop first so it cannot access sessions or KV cache
        // after they are disposed below. KvMemoryGovernor.DisposeAsync() cancels its internal
        // CancellationTokenSource and awaits the background task before returning.
        if (MemoryGovernor is IAsyncDisposable asyncGovernor)
        {
            try { await asyncGovernor.DisposeAsync().ConfigureAwait(false); } catch { }
        }

        if (Synthesizer is not null)
        {
            await Synthesizer.DisposeAsync().ConfigureAwait(false);
        }

        await SessionManager.DisposeAsync().ConfigureAwait(false);

        PrefixIndex?.Dispose();

        if (_ownsKvCache && KvCache is IDisposable disposableCache)
        {
            disposableCache.Dispose();
        }
    }
}
