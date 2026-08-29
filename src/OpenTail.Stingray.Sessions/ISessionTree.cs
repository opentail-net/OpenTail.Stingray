
namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Read-only topology interface exposing session lineage, parent-child branch relationships, and aggregated tree metrics.
/// </summary>
public interface ISessionTree
{
    /// <summary>Unique ID of the root session at the base of this lineage tree.</summary>
    SessionId RootId { get; }

    /// <summary>Unique ID of the direct parent session if this session was spawned via Fork(), or null if root.</summary>
    SessionId? ParentId { get; }

    /// <summary>Snapshot of currently active child branch session IDs directly descended from this session.</summary>
    IReadOnlyList<SessionId> Children { get; }

    /// <summary>Aggregated snapshot metrics across this session and all currently active descendant branches in its subtree.</summary>
    ISessionMetrics CumulativeTreeMetrics { get; }
}
