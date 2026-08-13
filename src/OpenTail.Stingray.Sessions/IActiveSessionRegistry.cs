using System.Collections.Generic;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Read-only registry interface for enumerating active inference sessions safely
/// without holding long-lived runtime locks.
/// </summary>
public interface IActiveSessionRegistry
{
    /// <summary>
    /// Returns a point-in-time snapshot of active session references.
    /// </summary>
    IReadOnlyList<IInferenceSession> GetActiveSessionsSnapshot();
}
