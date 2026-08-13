namespace OpenTail.Stingray.Engine;

/// <summary>
/// Strong-typed identifier for a physical or logical KV page in the paged KV cache architecture.
/// </summary>
public readonly record struct KvPageId(int Value)
{
    public static KvPageId Invalid => new(-1);
    public bool IsValid => Value >= 0;

    public override string ToString() => IsValid ? $"Page#{Value}" : "Page#Invalid";
}

/// <summary>
/// Fixed token capacity per page (e.g. 32 tokens/page).
/// </summary>
public readonly record struct KvPageSize(int Tokens)
{
    public static KvPageSize Default => new(32);

    public override string ToString() => $"{Tokens} tokens/page";
}

/// <summary>
/// Centralized token-to-page indexing calculations.
/// </summary>
public static class KvPageMath
{
    public static int GetPageIndex(int tokenPosition, int pageSize)
    {
        if (pageSize <= 0) throw new ArgumentOutOfRangeException(nameof(pageSize), "Page size must be positive.");
        if (tokenPosition < 0) throw new ArgumentOutOfRangeException(nameof(tokenPosition), "Token position cannot be negative.");
        return tokenPosition / pageSize;
    }

    public static int GetPageOffset(int tokenPosition, int pageSize)
    {
        if (pageSize <= 0) throw new ArgumentOutOfRangeException(nameof(pageSize), "Page size must be positive.");
        if (tokenPosition < 0) throw new ArgumentOutOfRangeException(nameof(tokenPosition), "Token position cannot be negative.");
        return tokenPosition % pageSize;
    }

    public static int GetRequiredPageCount(int tokenCount, int pageSize)
    {
        if (pageSize <= 0) throw new ArgumentOutOfRangeException(nameof(pageSize), "Page size must be positive.");
        if (tokenCount <= 0) return 0;
        return (tokenCount + pageSize - 1) / pageSize;
    }
}

/// <summary>
/// Global options for constructing an <see cref="IKvCache"/>.
/// </summary>
public sealed record KvCacheOptions
{
    public KvPageSize PageSize { get; init; } = KvPageSize.Default;
    public long MaxCapacityBytes { get; init; }
    public int MaxPages { get; init; }
}

/// <summary>
/// Per-sequence creation options.
/// </summary>
public sealed record KvSequenceOptions
{
    public int InitialCapacityTokens { get; init; }
    /// <summary>Optional sequence-level maximum context limit in tokens. null or &lt;= 0 means unlimited.</summary>
    public int? MaxContextTokens { get; init; }
}

/// <summary>
/// Detailed observability counters and status snapshot for an <see cref="IKvCache"/>.
/// </summary>
public readonly record struct KvCacheStatistics(
    long CapacityBytes,
    long UsedBytes,
    long FreeBytes,
    long ReservedBytes,
    int TotalPages,
    int UsedPages,
    int FreePages,
    int SharedPages,
    long Allocations,
    long Releases,
    long Forks,
    long CopyOnWriteCopies,
    long Evictions);

/// <summary>
/// Admission-control reservation ticket protecting against continuous-batching memory overcommit.
/// </summary>
public interface IKvReservation : IDisposable
{
    long SequenceId { get; }
    int ReservedTokens { get; }
    long ReservedBytes { get; }
    bool TryGrow(int additionalTokens);
}

/// <summary>
/// Logical owner of a sequence's page table.
/// </summary>
public interface IKvSequence : IDisposable
{
    long SequenceId { get; }
    int TokenCount { get; }
    int CapacityTokens { get; }
    /// <summary>The physical KV page capacity in tokens. This is the authoritative source of page size; never infer it from TokenCount / PageCount.</summary>
    int PageSize { get; }
    int PageCount { get; }
    ReadOnlySpan<KvPageId> Pages { get; }

    void Append(int tokenCount);
    void TruncateTo(int tokenCount);
    IKvSequence Fork();
    IKvSequence ForkAt(int tokenCount);
    void Release();
    void Clear();
}

/// <summary>
/// Zero-allocation view over a sequence's logical pages passed into forward pass computation.
/// </summary>
public readonly ref struct KvSequenceView
{
    public int TokenCount { get; }
    public int PageSize { get; }
    public ReadOnlySpan<KvPageId> Pages { get; }

    public KvSequenceView(int tokenCount, int pageSize, ReadOnlySpan<KvPageId> pages)
    {
        TokenCount = tokenCount;
        PageSize = pageSize;
        Pages = pages;
    }
}

/// <summary>
/// Global manager of physical KV storage pages, sequence allocations, and capacity reservations.
/// Decouples physical memory residency from logical inference scheduling.
/// </summary>
public interface IKvCache : IDisposable
{
    long CapacityBytes { get; }
    long UsedBytes { get; }
    long FreeBytes { get; }
    long BytesPerToken { get; }
    int PageSizeTokens { get; }
    int TotalPages { get; }
    int FreePages { get; }
    int UsedPages { get; }

    IKvSequence AllocateSequence(KvSequenceOptions? options = null);
    IKvReservation? TryReserve(long sequenceId, int requiredTokens);
    IKvReservation? TryReserveSequences(long sequenceId, System.Collections.Generic.IReadOnlyList<int> sequenceTokenCounts);
    void ReleaseSequence(IKvSequence sequence);
    KvCacheStatistics GetStatistics();
}
