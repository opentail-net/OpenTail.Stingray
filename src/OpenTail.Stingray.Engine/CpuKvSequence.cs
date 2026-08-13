using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Managed CPU sequence implementation of <see cref="IKvSequence"/> maintaining a sequence's logical page table.
/// </summary>
public sealed class CpuKvSequence : IKvSequence
{
    private readonly CpuKvCache _owner;
    private readonly long _sequenceId;
    private readonly int _pageSizeTokens;
    private readonly List<KvPageId> _pages = new();
    private int _tokenCount;
    private bool _disposed;

    internal CpuKvSequence(CpuKvCache owner, long sequenceId, int pageSizeTokens)
    {
        _owner = owner;
        _sequenceId = sequenceId;
        _pageSizeTokens = pageSizeTokens;
    }

    public long SequenceId => _sequenceId;
    public int TokenCount => _tokenCount;
    public int CapacityTokens => _pages.Count * _pageSizeTokens;
    public int PageSize => _pageSizeTokens;
    public int PageCount => _pages.Count;
    public ReadOnlySpan<KvPageId> Pages => CollectionsMarshal.AsSpan(_pages);

    public void Append(int tokenCount)
    {
        if (tokenCount < 0) throw new ArgumentOutOfRangeException(nameof(tokenCount));
        if (tokenCount == 0) return;

        KvPageId originalTailPage = KvPageId.Invalid;
        KvPageId cowNewPage = KvPageId.Invalid;
        int originalTailIndex = -1;

        var newlyAllocated = new List<KvPageId>();

        try
        {
            // Copy-on-write check: if appending to a shared, unaligned page boundary, duplicate to private page
            if (_pages.Count > 0 && _tokenCount % _pageSizeTokens != 0)
            {
                var lastPage = _pages[^1];
                if (_owner.IsPageShared(lastPage))
                {
                    originalTailIndex = _pages.Count - 1;
                    originalTailPage = lastPage;
                    cowNewPage = _owner.PerformCopyOnWrite(lastPage);
                    _pages[originalTailIndex] = cowNewPage;
                }
            }

            int newTotalTokens = _tokenCount + tokenCount;
            int requiredPages = KvPageMath.GetRequiredPageCount(newTotalTokens, _pageSizeTokens);

                while (_pages.Count < requiredPages)
            {
                if (!_owner.TryAllocatePage(out KvPageId newPage))
                {
                    throw new InvalidOperationException($"Out of KV cache pages. Requested {requiredPages}, currently allocated {_pages.Count}.");
                }
                newlyAllocated.Add(newPage);
                _pages.Add(newPage);
            }

            _tokenCount = newTotalTokens;
        }
        catch
        {
            foreach (var page in newlyAllocated)
            {
                _pages.Remove(page);
                _owner.ReleasePage(page);
            }

            // Rollback COW page if it was performed in this Append transaction
            if (originalTailIndex >= 0 && originalTailPage.IsValid && cowNewPage.IsValid)
            {
                _owner.RetainPage(originalTailPage);
                _pages[originalTailIndex] = originalTailPage;
                _owner.ReleasePage(cowNewPage);
            }

            throw;
        }
    }

    public void TruncateTo(int targetTokenCount)
    {
        if (targetTokenCount < 0 || targetTokenCount > _tokenCount)
            throw new ArgumentOutOfRangeException(nameof(targetTokenCount));

        if (targetTokenCount == _tokenCount) return;

        int requiredPages = KvPageMath.GetRequiredPageCount(targetTokenCount, _pageSizeTokens);
        while (_pages.Count > requiredPages)
        {
            var pageToRelease = _pages[^1];
            _pages.RemoveAt(_pages.Count - 1);
            _owner.ReleasePage(pageToRelease);
        }

        _tokenCount = targetTokenCount;
    }

    public IKvSequence Fork() => ForkAt(_tokenCount);

    public IKvSequence ForkAt(int tokenCount)
    {
        if (tokenCount < 0 || tokenCount > _tokenCount)
            throw new ArgumentOutOfRangeException(nameof(tokenCount));

        var child = (CpuKvSequence)_owner.AllocateSequence();
        int sharedPageCount = KvPageMath.GetRequiredPageCount(tokenCount, _pageSizeTokens);

        try
        {
            for (int i = 0; i < sharedPageCount; i++)
            {
                KvPageId page = _pages[i];
                _owner.RetainPage(page);
                child._pages.Add(page);
            }
        }
        catch
        {
            child.Dispose();
            throw;
        }

        child._tokenCount = tokenCount;
        _owner.RecordFork();
        return child;
    }

    public void Clear()
    {
        foreach (var page in _pages)
        {
            _owner.ReleasePage(page);
        }
        _pages.Clear();
        _tokenCount = 0;
    }

    public void Release() => Dispose();

    internal void ReleaseInternal() => Clear();

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Clear();
    }
}
