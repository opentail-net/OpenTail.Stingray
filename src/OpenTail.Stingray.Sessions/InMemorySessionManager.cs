using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// In-memory session manager registry for managing lifecycle of active <see cref="IInferenceSession"/> instances.
/// </summary>
public sealed class InMemorySessionManager : IAsyncDisposable, IActiveSessionRegistry
{
    private readonly IKvCache _kvCache;
    private readonly ConcurrentDictionary<SessionId, IInferenceSession> _sessions = new();
    private bool _disposed;

    public InMemorySessionManager(IKvCache kvCache)
    {
        _kvCache = kvCache ?? throw new ArgumentNullException(nameof(kvCache));
    }

    public int ActiveSessionCount => _sessions.Count;
    public IReadOnlyCollection<SessionId> ActiveSessionIds => _sessions.Keys.ToArray();
    public IReadOnlyCollection<IInferenceSession> GetActiveSessions() => _sessions.Values.ToArray();
    public IReadOnlyList<IInferenceSession> GetActiveSessionsSnapshot() => _sessions.Values.ToArray();

    public (int ReadyCount, int SuspendedCount, int ColdCount) GetSessionStateBreakdown()
    {
        int ready = 0, suspended = 0, cold = 0;
        foreach (var s in _sessions.Values)
        {
            switch (s.State)
            {
                case SessionState.Ready:
                case SessionState.Generating:
                    ready++;
                    break;
                case SessionState.Suspended:
                    suspended++;
                    break;
                case SessionState.Cold:
                    cold++;
                    break;
            }
        }
        return (ready, suspended, cold);
    }

    public ValueTask<IInferenceSession> CreateSessionAsync(
        KvSequenceOptions? options = null,
        OpenTail.Stingray.Core.IForwardPass? forwardPass = null,
        CancellationToken ct = default)
    {
        ThrowIfDisposed();
        var session = new InferenceSession(_kvCache, options: options, forwardPass: forwardPass);
        RegisterSession(session);
        return ValueTask.FromResult<IInferenceSession>(session);
    }

    public void RegisterSession(IInferenceSession session)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(session);
        _sessions[session.Id] = session;
    }

    public IInferenceSession? GetSession(SessionId id)
    {
        ThrowIfDisposed();
        _sessions.TryGetValue(id, out var session);
        return session;
    }

    public async ValueTask<bool> RemoveSessionAsync(SessionId id)
    {
        ThrowIfDisposed();
        if (_sessions.TryRemove(id, out var session))
        {
            await session.DisposeAsync().ConfigureAwait(false);
            return true;
        }
        return false;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(InMemorySessionManager));
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var pair in _sessions)
        {
            await pair.Value.DisposeAsync().ConfigureAwait(false);
        }
        _sessions.Clear();
    }
}
