using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace OpenTail.Stingray.Sessions;

internal sealed class SessionTree : ISessionTree
{
    private readonly IInferenceSession _owner;
    private readonly SessionTree? _parentTree;
    private readonly ConcurrentDictionary<SessionId, IInferenceSession> _activeChildren = new();

    public SessionId RootId { get; }
    public SessionId? ParentId { get; }
    public SessionId OwnerId => _owner.Id;

    public SessionTree(IInferenceSession owner, SessionTree? parentTree)
    {
        _owner = owner ?? throw new ArgumentNullException(nameof(owner));
        _parentTree = parentTree;

        if (parentTree is null)
        {
            ParentId = null;
            RootId = _owner.Id;
        }
        else
        {
            ParentId = parentTree.OwnerId;
            RootId = parentTree.RootId;
            parentTree.AddChild(_owner);
        }
    }

    public IReadOnlyList<SessionId> Children => _activeChildren.Keys.ToList();

    internal IReadOnlyList<IInferenceSession> ActiveChildSessions => _activeChildren.Values.ToList();

    public ISessionMetrics CumulativeTreeMetrics => CalculateCumulativeMetrics();

    public void AddChild(IInferenceSession child)
    {
        if (child is not null)
        {
            _activeChildren[child.Id] = child;
        }
    }

    public void RemoveChild(SessionId childId)
    {
        _activeChildren.TryRemove(childId, out _);
    }

    public void OnDisposed()
    {
        _parentTree?.RemoveChild(_owner.Id);
    }

    private ISessionMetrics CalculateCumulativeMetrics()
    {
        long promptTokens = _owner.Metrics.PromptTokens;
        long generatedTokens = _owner.Metrics.GeneratedTokens;
        TimeSpan totalPrefillTime = _owner.Metrics.TotalPrefillTime;
        TimeSpan totalGenerationTime = _owner.Metrics.TotalGenerationTime;
        int kvPages = _owner.Metrics.KvPagesHeld;

        var queue = new Queue<IInferenceSession>(_activeChildren.Values);
        var visited = new HashSet<SessionId> { _owner.Id };

        while (queue.Count > 0)
        {
            var child = queue.Dequeue();
            if (visited.Add(child.Id))
            {
                promptTokens += child.Metrics.PromptTokens;
                generatedTokens += child.Metrics.GeneratedTokens;
                totalPrefillTime += child.Metrics.TotalPrefillTime;
                totalGenerationTime += child.Metrics.TotalGenerationTime;
                kvPages += child.Metrics.KvPagesHeld;

                if (child.Tree is SessionTree childTree)
                {
                    foreach (var grandchild in childTree.ActiveChildSessions)
                    {
                        if (!visited.Contains(grandchild.Id))
                        {
                            queue.Enqueue(grandchild);
                        }
                    }
                }
            }
        }

        var snapshot = new SessionMetrics(() => kvPages);
        snapshot.AddPromptTokens(promptTokens, totalPrefillTime);
        snapshot.AddGeneratedTokens(generatedTokens, totalGenerationTime);
        return snapshot;
    }
}
