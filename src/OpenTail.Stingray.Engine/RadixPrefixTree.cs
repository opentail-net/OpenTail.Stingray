
namespace OpenTail.Stingray.Engine;

/// <summary>
/// Thread-safe, page-aligned Radix Tree Prefix Cache implementing <see cref="IPrefixCacheIndex"/>.
/// Maps token ID paths within isolated model/KV namespaces to physical <see cref="KvPageId"/> pages.
/// Supports node splitting, multi-depth branch traversal, and atomic ref-count matching.
/// </summary>
public sealed class RadixPrefixTree : IPrefixCacheIndex
{
    private readonly CpuKvCache _kvCache;
    private readonly int _pageSizeTokens;
    private readonly ReaderWriterLockSlim _lock = new(LockRecursionPolicy.NoRecursion);
    private readonly Dictionary<PrefixCacheNamespace, RadixTreeNode> _namespaces = new();
    private bool _disposed;

    public RadixPrefixTree(CpuKvCache kvCache)
    {
        _kvCache = kvCache ?? throw new ArgumentNullException(nameof(kvCache));
        _pageSizeTokens = kvCache.PageSizeTokens;
    }

    public PrefixMatchResult MatchPrefix(PrefixCacheNamespace ns, ReadOnlySpan<int> tokens)
    {
        ThrowIfDisposed();
        if (tokens.Length < _pageSizeTokens)
        {
            return new PrefixMatchResult(0, 0, ReadOnlyMemory<KvPageId>.Empty);
        }

        _lock.EnterUpgradeableReadLock();
        try
        {
            if (!_namespaces.TryGetValue(ns, out var root))
            {
                return new PrefixMatchResult(0, 0, ReadOnlyMemory<KvPageId>.Empty);
            }

            var matchedPages = new List<KvPageId>();
            int currentTokenIndex = 0;
            var currentNode = root;

            while (currentTokenIndex < tokens.Length)
            {
                int remainingTokens = tokens.Length - currentTokenIndex;
                if (remainingTokens < _pageSizeTokens) break;

                int firstToken = tokens[currentTokenIndex];
                if (!currentNode.Children.TryGetValue(firstToken, out var child))
                {
                    break;
                }

                var childTokens = child.Tokens.Span;
                if (remainingTokens < childTokens.Length) break;

                bool match = true;
                for (int i = 0; i < childTokens.Length; i++)
                {
                    if (tokens[currentTokenIndex + i] != childTokens[i])
                    {
                        match = false;
                        break;
                    }
                }

                if (!match) break;

                // Match confirmed for child node segment
                currentTokenIndex += childTokens.Length;
                matchedPages.AddRange(child.Pages.Span.ToArray());
                child.LastAccessedAt = DateTimeOffset.UtcNow;
                currentNode = child;
            }

            if (matchedPages.Count == 0)
            {
                return new PrefixMatchResult(0, 0, ReadOnlyMemory<KvPageId>.Empty);
            }

            // ATOMIC RETENTION: Call RetainPage inside tree lock before returning
            foreach (var page in matchedPages)
            {
                _kvCache.RetainPage(page);
            }

            int matchedTokenCount = matchedPages.Count * _pageSizeTokens;
            return new PrefixMatchResult(matchedTokenCount, matchedPages.Count, matchedPages.ToArray());
        }
        finally
        {
            _lock.ExitUpgradeableReadLock();
        }
    }

    public void Publish(PrefixCacheNamespace ns, ReadOnlySpan<int> tokens, IKvSequence sequence, int committedTokenCount)
    {
        ThrowIfDisposed();
        if (sequence == null || committedTokenCount < _pageSizeTokens) return;

        int fullPageCount = committedTokenCount / _pageSizeTokens;
        if (fullPageCount <= 0 || sequence.Pages.Length < fullPageCount) return;

        int fullTokenCount = fullPageCount * _pageSizeTokens;
        if (tokens.Length < fullTokenCount) return;

        var fullPages = sequence.Pages.Slice(0, fullPageCount);
        var pageTokens = tokens.Slice(0, fullTokenCount).ToArray();

        _lock.EnterWriteLock();
        try
        {
            if (!_namespaces.TryGetValue(ns, out var root))
            {
                root = new RadixTreeNode(ReadOnlyMemory<int>.Empty, ReadOnlyMemory<KvPageId>.Empty);
                _namespaces[ns] = root;
            }

            int tokenOffset = 0;
            int pageOffset = 0;
            var currentNode = root;

            while (tokenOffset < fullTokenCount)
            {
                int firstToken = pageTokens[tokenOffset];

                if (!currentNode.Children.TryGetValue(firstToken, out var child))
                {
                    // Attach remaining unsplit segment
                    int remainingTokenCount = fullTokenCount - tokenOffset;
                    int remainingPageCount = remainingTokenCount / _pageSizeTokens;

                    var nodeTokens = pageTokens.AsSpan(tokenOffset, remainingTokenCount).ToArray();
                    var nodePages = fullPages.Slice(pageOffset, remainingPageCount).ToArray();

                    foreach (var page in nodePages)
                    {
                        _kvCache.RetainPage(page);
                    }

                    var newNode = new RadixTreeNode(nodeTokens, nodePages);
                    currentNode.Children[firstToken] = newNode;
                    break;
                }

                // Node exists: calculate common prefix matching length
                var childTokens = child.Tokens.Span;
                int commonLength = 0;
                int maxCompare = Math.Min(childTokens.Length, fullTokenCount - tokenOffset);

                while (commonLength < maxCompare && childTokens[commonLength] == pageTokens[tokenOffset + commonLength])
                {
                    commonLength++;
                }

                if (commonLength < childTokens.Length)
                {
                    // Physical pages can only be shared at page boundaries.  Splitting a
                    // compressed token segment in the middle of a page would attach that
                    // page to two different token prefixes, so retain the existing entry and
                    // decline to index this conflicting prefix instead.
                    int alignedCommonLength = commonLength / _pageSizeTokens * _pageSizeTokens;
                    if (alignedCommonLength == 0)
                    {
                        return;
                    }

                    int commonPages = alignedCommonLength / _pageSizeTokens;

                    var commonChildTokens = child.Tokens.Slice(0, alignedCommonLength);
                    var commonChildPages = child.Pages.Slice(0, commonPages);

                    var splitChildTokens = child.Tokens.Slice(alignedCommonLength);
                    var splitChildPages = child.Pages.Slice(commonPages);

                    // Create split child holding tail tokens/pages
                    var splitChild = new RadixTreeNode(splitChildTokens, splitChildPages);
                    foreach (var childKvp in child.Children)
                    {
                        splitChild.Children[childKvp.Key] = childKvp.Value;
                    }
                    splitChild.LastAccessedAt = child.LastAccessedAt;

                    // Reconfigure existing child node to hold common prefix segment
                    child.SetSegment(commonChildTokens, commonChildPages);
                    child.Children.Clear();
                    child.Children[splitChildTokens.Span[0]] = splitChild;

                    // Continue from the page-aligned common segment. The next iteration
                    // inserts the divergent suffix with its own physical pages.
                    commonLength = alignedCommonLength;
                }

                tokenOffset += commonLength;
                pageOffset += commonLength / _pageSizeTokens;
                child.LastAccessedAt = DateTimeOffset.UtcNow;
                currentNode = child;
            }
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    public void Remove(PrefixCacheNamespace ns, ReadOnlySpan<int> tokens)
    {
        ThrowIfDisposed();
        _lock.EnterWriteLock();
        try
        {
            if (!_namespaces.TryGetValue(ns, out var root) || tokens.Length == 0) return;

            RadixTreeNode parent = root;
            int offset = 0;
            while (offset < tokens.Length)
            {
                int key = tokens[offset];
                if (!parent.Children.TryGetValue(key, out var child)) return;

                var childTokens = child.Tokens.Span;
                if (tokens.Length - offset < childTokens.Length) return;
                for (int i = 0; i < childTokens.Length; i++)
                {
                    if (tokens[offset + i] != childTokens[i]) return;
                }

                offset += childTokens.Length;
                if (offset == tokens.Length)
                {
                    if (parent.Children.Remove(key, out var removed)) ReleaseSubtreePages(removed);
                    return;
                }
                parent = child;
            }
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    public int EvictLruEntries(int targetPagesToFree)
    {
        ThrowIfDisposed();
        if (targetPagesToFree <= 0) return 0;

        _lock.EnterWriteLock();
        try
        {
            int freedPages = 0;
            var candidates = new List<(PrefixCacheNamespace Ns, int Key, RadixTreeNode Parent, RadixTreeNode Child)>();

            foreach (var kvp in _namespaces)
            {
                CollectEvictionCandidates(kvp.Key, kvp.Value, candidates);
            }

            candidates.Sort((a, b) => a.Child.LastAccessedAt.CompareTo(b.Child.LastAccessedAt));

            foreach (var candidate in candidates)
            {
                if (freedPages >= targetPagesToFree) break;

                if (candidate.Parent.Children.Remove(candidate.Key, out var removedNode))
                {
                    freedPages += ReleaseSubtreePages(removedNode);
                }
            }

            return freedPages;
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    private void CollectEvictionCandidates(
        PrefixCacheNamespace ns,
        RadixTreeNode parent,
        List<(PrefixCacheNamespace Ns, int Key, RadixTreeNode Parent, RadixTreeNode Child)> candidates)
    {
        foreach (var kvp in parent.Children)
        {
            var child = kvp.Value;
            candidates.Add((ns, kvp.Key, parent, child));
            CollectEvictionCandidates(ns, child, candidates);
        }
    }

    private int ReleaseSubtreePages(RadixTreeNode node)
    {
        int releasedCount = 0;
        foreach (var page in node.Pages.Span)
        {
            _kvCache.ReleasePage(page);
            releasedCount++;
        }

        foreach (var childKvp in node.Children.Values)
        {
            releasedCount += ReleaseSubtreePages(childKvp);
        }
        node.Children.Clear();
        return releasedCount;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(RadixPrefixTree));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _lock.EnterWriteLock();
        try
        {
            foreach (var nsKvp in _namespaces)
            {
                foreach (var childKvp in nsKvp.Value.Children.Values)
                {
                    ReleaseSubtreePages(childKvp);
                }
                nsKvp.Value.Children.Clear();
            }
            _namespaces.Clear();
        }
        finally
        {
            _lock.ExitWriteLock();
            _lock.Dispose();
        }
    }

    private sealed class RadixTreeNode
    {
        public ReadOnlyMemory<int> Tokens { get; private set; }
        public ReadOnlyMemory<KvPageId> Pages { get; private set; }
        public Dictionary<int, RadixTreeNode> Children { get; } = new();
        public DateTimeOffset LastAccessedAt { get; set; } = DateTimeOffset.UtcNow;

        public RadixTreeNode(ReadOnlyMemory<int> tokens, ReadOnlyMemory<KvPageId> pages)
        {
            Tokens = tokens;
            Pages = pages;
        }

        public void SetSegment(ReadOnlyMemory<int> tokens, ReadOnlyMemory<KvPageId> pages)
        {
            Tokens = tokens;
            Pages = pages;
        }
    }
}
