using System.Collections.Concurrent;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Managed CPU implementation of <see cref="IKvCache"/> owning a pool of logical page IDs,
/// reference counting for page sharing/forking, copy-on-write mechanics, and admission reservations.
/// </summary>
public sealed class CpuKvCache : IKvCache
{
    // Diagnostic flag controlled by environment variable OPENTAIL_DEBUG_KV
    internal static readonly bool DiagEnabled = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("OPENTAIL_DEBUG_KV"));

    private static void Diag(string fmt, params object[] args)
    {
        if (!DiagEnabled) return;
        try { Console.Error.WriteLine("[CpuKvCache] " + string.Format(fmt, args)); } catch { }
    }

    private readonly int _pageSizeTokens;
    private readonly long _bytesPerToken;
    private readonly int _totalPages;
    private readonly ConcurrentQueue<int> _freePageIds = new();
    private readonly int[] _pageRefCounts;
    private readonly byte[][] _pageBuffers;

    private long _allocations;
    private long _releases;
    private long _forks;
    private long _cowCopies;
    private long _evictions;
    private long _reservedTokens;
    private long _reservedPages;
    private long _sequenceIdCounter;

    private bool _disposed;

    public CpuKvCache(int totalPages, int pageSizeTokens = 32, long bytesPerToken = 1024)
    {
        if (totalPages <= 0) throw new ArgumentOutOfRangeException(nameof(totalPages));
        if (pageSizeTokens <= 0) throw new ArgumentOutOfRangeException(nameof(pageSizeTokens));

        _totalPages = totalPages;
        _pageSizeTokens = pageSizeTokens;
        _bytesPerToken = bytesPerToken;
        _pageRefCounts = new int[totalPages];
        _pageBuffers = new byte[totalPages][];

        for (int i = 0; i < totalPages; i++)
        {
            _freePageIds.Enqueue(i);
        }
    }

    public long CapacityBytes => (long)_totalPages * _pageSizeTokens * _bytesPerToken;
    public long UsedBytes => (long)UsedPages * _pageSizeTokens * _bytesPerToken;
    public long FreeBytes => (long)FreePages * _pageSizeTokens * _bytesPerToken;
    public long BytesPerToken => _bytesPerToken;
    public int PageSizeTokens => _pageSizeTokens;
    public int TotalPages => _totalPages;
    public int FreePages => _freePageIds.Count;
    public int UsedPages => _totalPages - FreePages;
    public int SharedPages
    {
        get
        {
            int shared = 0;
            for (int i = 0; i < _totalPages; i++)
            {
                if (Volatile.Read(ref _pageRefCounts[i]) > 1) shared++;
            }
            return shared;
        }
    }

    public IKvSequence AllocateSequence(KvSequenceOptions? options = null)
    {
        long seqId = Interlocked.Increment(ref _sequenceIdCounter);
        Interlocked.Increment(ref _allocations);
        return new CpuKvSequence(this, seqId, _pageSizeTokens);
    }

    public IKvReservation? TryReserve(long sequenceId, int requiredTokens)
    {
        int requiredPages = KvPageMath.GetRequiredPageCount(requiredTokens, _pageSizeTokens);
        Diag("TryReserve: seq={0} requiredTokens={1} requiredPages={2} FreePages={3} ReservedPages={4}", sequenceId, requiredTokens, requiredPages, FreePages, Volatile.Read(ref _reservedPages));
        while (true)
        {
            long currentReservedPages = Volatile.Read(ref _reservedPages);
            int unreservedFreePages = FreePages - (int)currentReservedPages;

            if (unreservedFreePages < requiredPages)
            {
                return null; // Admission control backpressure
            }

            if (Interlocked.CompareExchange(ref _reservedPages, currentReservedPages + requiredPages, currentReservedPages) == currentReservedPages)
            {
                Interlocked.Add(ref _reservedTokens, requiredTokens);
                Diag("TryReserve: seq={0} success reservedPages={1} reservedTokens={2}", sequenceId, currentReservedPages + requiredPages, Volatile.Read(ref _reservedTokens));
                return new CpuKvReservation(this, sequenceId, requiredTokens, requiredPages);
            }
        }
    }

    /// <summary>
    /// Reserves physical KV cache page capacity accounting for per-sequence page rounding.
    /// Prevents page-overcommit admission when multiple sequence requests have unaligned token lengths.
    /// </summary>
    public IKvReservation? TryReserveSequences(long sequenceId, IReadOnlyList<int> sequenceTokenCounts)
    {
        if (sequenceTokenCounts is null || sequenceTokenCounts.Count == 0) return null;

        int totalTokens = 0;
        int requiredPages = 0;
        for (int i = 0; i < sequenceTokenCounts.Count; i++)
        {
            totalTokens += sequenceTokenCounts[i];
            requiredPages += KvPageMath.GetRequiredPageCount(sequenceTokenCounts[i], _pageSizeTokens);
        }

        while (true)
        {
            long currentReservedPages = Volatile.Read(ref _reservedPages);
            int unreservedFreePages = FreePages - (int)currentReservedPages;

            if (unreservedFreePages < requiredPages)
            {
                return null; // Admission control backpressure
            }

            if (Interlocked.CompareExchange(ref _reservedPages, currentReservedPages + requiredPages, currentReservedPages) == currentReservedPages)
            {
                Interlocked.Add(ref _reservedTokens, totalTokens);
                Diag("TryReserveSequences: seq={0} success reservedPages={1} reservedTokens={2}", sequenceId, currentReservedPages + requiredPages, Volatile.Read(ref _reservedTokens));
                return new CpuKvReservation(this, sequenceId, totalTokens, requiredPages);
            }
        }
    }

    public void ReleaseSequence(IKvSequence sequence)
    {
        if (sequence is CpuKvSequence cpuSeq)
        {
            cpuSeq.ReleaseInternal();
            Interlocked.Increment(ref _releases);
        }
    }

    public bool IsPageShared(KvPageId pageId)
    {
        if (!pageId.IsValid) return false;
        return Volatile.Read(ref _pageRefCounts[pageId.Value]) > 1;
    }

    internal bool TryAllocatePage(out KvPageId pageId)
    {
        if (_freePageIds.TryDequeue(out int id))
        {
            Volatile.Write(ref _pageRefCounts[id], 1);
            _pageBuffers[id] ??= new byte[(int)(_pageSizeTokens * _bytesPerToken)];
            Interlocked.Increment(ref _allocations);
            pageId = new KvPageId(id);
            Diag("TryAllocatePage: allocated id={0} FreePages={1} UsedPages={2}", id, FreePages, UsedPages);
            return true;
        }
        pageId = KvPageId.Invalid;
        Diag("TryAllocatePage: failed to allocate page; FreePages={0}", FreePages);
        return false;
    }

    internal void RetainPage(KvPageId pageId)
    {
        if (!pageId.IsValid) return;
        Interlocked.Increment(ref _pageRefCounts[pageId.Value]);
        Diag("RetainPage: id={0} refCount={1}", pageId.Value, Volatile.Read(ref _pageRefCounts[pageId.Value]));
    }

    internal void ReleasePage(KvPageId pageId)
    {
        if (!pageId.IsValid) return;
        int newRefCount = Interlocked.Decrement(ref _pageRefCounts[pageId.Value]);
        if (newRefCount == 0)
        {
            _freePageIds.Enqueue(pageId.Value);
            Diag("ReleasePage: id={0} freed FreePages={1}", pageId.Value, FreePages);
        }
        else if (newRefCount < 0)
        {
            throw new InvalidOperationException($"Double free or corrupt page refcount for page {pageId.Value}.");
        }
    }

    internal KvPageId PerformCopyOnWrite(KvPageId oldPageId)
    {
        if (!TryAllocatePage(out KvPageId newPageId))
        {
            Diag("PerformCopyOnWrite: allocation failed for oldPage={0}", oldPageId.Value);
            throw new InvalidOperationException("Failed to allocate new page for copy-on-write operation.");
        }

        // Copy physical tensor page memory from old page to new page before releasing old page handle
        if (oldPageId.IsValid && newPageId.IsValid && _pageBuffers[oldPageId.Value] != null && _pageBuffers[newPageId.Value] != null)
        {
            Buffer.BlockCopy(_pageBuffers[oldPageId.Value], 0, _pageBuffers[newPageId.Value], 0, _pageBuffers[oldPageId.Value].Length);
        }

        Interlocked.Increment(ref _cowCopies);
        ReleasePage(oldPageId);
        return newPageId;
    }

    public byte[]? GetPageBuffer(KvPageId pageId)
    {
        if (!pageId.IsValid || pageId.Value < 0 || pageId.Value >= _totalPages) return null;
        return _pageBuffers[pageId.Value];
    }

    internal void RecordFork() => Interlocked.Increment(ref _forks);

    internal void RecordEviction() => Interlocked.Increment(ref _evictions);

    internal void ReleaseReservation(int pages, int tokens)
    {
        Interlocked.Add(ref _reservedPages, -pages);
        Interlocked.Add(ref _reservedTokens, -tokens);
        Diag("ReleaseReservation: pages={0} tokens={1} FreePages={2} ReservedPages={3}", pages, tokens, FreePages, Volatile.Read(ref _reservedPages));
    }

    public KvCacheStatistics GetStatistics()
    {
        return new KvCacheStatistics(
            CapacityBytes: CapacityBytes,
            UsedBytes: UsedBytes,
            FreeBytes: FreeBytes,
            ReservedBytes: Volatile.Read(ref _reservedTokens) * _bytesPerToken,
            TotalPages: TotalPages,
            UsedPages: UsedPages,
            FreePages: FreePages,
            SharedPages: SharedPages,
            Allocations: Volatile.Read(ref _allocations),
            Releases: Volatile.Read(ref _releases),
            Forks: Volatile.Read(ref _forks),
            CopyOnWriteCopies: Volatile.Read(ref _cowCopies),
            Evictions: Volatile.Read(ref _evictions));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
    }

    private sealed class CpuKvReservation : IKvReservation
    {
        private readonly CpuKvCache _owner;
        private bool _disposed;

        public CpuKvReservation(CpuKvCache owner, long sequenceId, int reservedTokens, int reservedPages)
        {
            _owner = owner;
            SequenceId = sequenceId;
            ReservedTokens = reservedTokens;
            ReservedPages = reservedPages;
        }

        public long SequenceId { get; }
        public int ReservedTokens { get; private set; }
        public int ReservedPages { get; private set; }
        public long ReservedBytes => ReservedTokens * _owner.BytesPerToken;

        public bool TryGrow(int additionalTokens)
        {
            if (additionalTokens <= 0) return true;

            int currentSeqPages = KvPageMath.GetRequiredPageCount(ReservedTokens, _owner.PageSizeTokens);
            int newSeqPages = KvPageMath.GetRequiredPageCount(ReservedTokens + additionalTokens, _owner.PageSizeTokens);
            int additionalPagesNeeded = newSeqPages - currentSeqPages;

            if (additionalPagesNeeded == 0)
            {
                ReservedTokens += additionalTokens;
                Interlocked.Add(ref _owner._reservedTokens, additionalTokens);
                return true;
            }

            while (true)
            {
                long currentReservedPages = Volatile.Read(ref _owner._reservedPages);
                int unreservedFreePages = _owner.FreePages - (int)currentReservedPages;

                if (unreservedFreePages < additionalPagesNeeded) return false;

                if (Interlocked.CompareExchange(ref _owner._reservedPages, currentReservedPages + additionalPagesNeeded, currentReservedPages) == currentReservedPages)
                {
                    ReservedTokens += additionalTokens;
                    ReservedPages += additionalPagesNeeded;
                    Interlocked.Add(ref _owner._reservedTokens, additionalTokens);
                    return true;
                }
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _owner.ReleaseReservation(ReservedPages, ReservedTokens);
        }
    }
}
