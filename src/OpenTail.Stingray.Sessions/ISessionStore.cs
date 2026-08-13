using System.Threading;
using System.Threading.Tasks;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Persistent storage provider for saving, loading, and deleting <see cref="InferenceSessionSnapshot"/> instances.
/// </summary>
public interface ISessionStore
{
    ValueTask SaveAsync(InferenceSessionSnapshot snapshot, CancellationToken cancellationToken = default);
    ValueTask SaveDeltaAsync(SessionDelta delta, CancellationToken cancellationToken = default) => ValueTask.CompletedTask;
    ValueTask<InferenceSessionSnapshot?> LoadAsync(SessionId id, CancellationToken cancellationToken = default);
    ValueTask<bool> DeleteAsync(SessionId id, CancellationToken cancellationToken = default);
}
