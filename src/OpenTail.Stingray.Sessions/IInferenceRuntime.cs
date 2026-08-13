using System;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

using OpenTail.Stingray.Core.Capabilities;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Top-level execution runtime orchestrating shared model resources, global physical
/// <see cref="IKvCache"/> page allocators, and active session registries.
/// </summary>
public interface IInferenceRuntime : IAsyncDisposable
{
    IKvCache KvCache { get; }
    InMemorySessionManager SessionManager { get; }
    InferenceCapabilities Capabilities { get; }
    ModelCapabilities ModelCapabilities => Capabilities.Model;
    IPrefixCacheIndex? PrefixIndex { get; }
    IKvMemoryGovernor MemoryGovernor { get; }

    ValueTask<IInferenceSession> CreateSessionAsync(
        KvSequenceOptions? options = null,
        IForwardPass? forwardPass = null,
        CancellationToken cancellationToken = default);

    IInferenceSession? GetSession(SessionId id);
    ValueTask<bool> RemoveSessionAsync(SessionId id);

    /// <summary>
    /// Disposes all active descendant branches in the lineage tree of <paramref name="rootId"/> without disposing the root session itself.
    /// Returns the total count of pruned descendant sessions.
    /// </summary>
    ValueTask<int> PruneBranchTreeAsync(SessionId rootId, CancellationToken cancellationToken = default);
}
