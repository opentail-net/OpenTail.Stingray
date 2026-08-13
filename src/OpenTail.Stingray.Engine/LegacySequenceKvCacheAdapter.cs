using System.Threading;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Adapter wrapping a legacy <see cref="ISequenceKvCache"/> instance into the new <see cref="IKvSequence"/> contract.
/// Enables incremental migration of existing engine and forward pass components without immediate raw buffer rewrites.
/// </summary>
public sealed class LegacySequenceKvCacheAdapter : IKvSequence, ISequenceKvCache
{
    private static long s_sequenceIdCounter;

    private readonly ISequenceKvCache _innerCache;
    private readonly int _pageSize;
    private readonly long _sequenceId;
    private KvPageId[] _pages;
    private int _tokenCount;
    private bool _disposed;

    private readonly bool _ownsInnerCache;

    public LegacySequenceKvCacheAdapter(ISequenceKvCache innerCache, int pageSize = 32, bool ownsInnerCache = true)
    {
        _innerCache = innerCache ?? throw new ArgumentNullException(nameof(innerCache));
        _pageSize = pageSize > 0 ? pageSize : 32;
        _sequenceId = Interlocked.Increment(ref s_sequenceIdCounter);
        _ownsInnerCache = ownsInnerCache;
        _pages = Array.Empty<KvPageId>();
        _tokenCount = 0;
        UpdatePages();
    }

    public ISequenceKvCache InnerCache => _innerCache;

    public long SequenceId => _sequenceId;

    public int TokenCount => _tokenCount;

    public int CapacityTokens => int.MaxValue;

    public int PageSize => _pageSize;

    public int PageCount => _pages.Length;

    public ReadOnlySpan<KvPageId> Pages => _pages;

    public void Append(int tokenCount)
    {
        if (tokenCount < 0) throw new ArgumentOutOfRangeException(nameof(tokenCount));
        if (tokenCount == 0) return;

        _tokenCount += tokenCount;
        UpdatePages();
    }

    public void TruncateTo(int targetTokenCount)
    {
        if (targetTokenCount < 0 || targetTokenCount > _tokenCount)
            throw new ArgumentOutOfRangeException(nameof(targetTokenCount));
        _tokenCount = targetTokenCount;
        UpdatePages();
    }

    public IKvSequence Fork() => throw new NotSupportedException("LegacySequenceKvCacheAdapter does not support zero-copy forking. Use CpuKvCache for paged CoW forking.");

    public IKvSequence ForkAt(int tokenCount) => throw new NotSupportedException("LegacySequenceKvCacheAdapter does not support zero-copy forking. Use CpuKvCache for paged CoW forking.");

    public void Clear()
    {
        _tokenCount = 0;
        UpdatePages();
    }

    public void Release() => Dispose();

    private void UpdatePages()
    {
        int requiredPages = KvPageMath.GetRequiredPageCount(_tokenCount, _pageSize);
        if (_pages.Length != requiredPages)
        {
            var newPages = new KvPageId[requiredPages];
            for (int i = 0; i < requiredPages; i++)
            {
                newPages[i] = new KvPageId(i);
            }
            _pages = newPages;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_ownsInnerCache)
        {
            _innerCache.Dispose();
        }
    }
}
