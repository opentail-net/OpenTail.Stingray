using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPass;

public class PagedKvCacheInvariantsTests
{
    [Fact]
    public void KvPageMath_IndexingAndPageCounts_AreCorrect()
    {
        int pageSize = 32;

        Assert.Equal(0, KvPageMath.GetPageIndex(0, pageSize));
        Assert.Equal(0, KvPageMath.GetPageOffset(0, pageSize));

        Assert.Equal(0, KvPageMath.GetPageIndex(31, pageSize));
        Assert.Equal(31, KvPageMath.GetPageOffset(31, pageSize));

        Assert.Equal(1, KvPageMath.GetPageIndex(32, pageSize));
        Assert.Equal(0, KvPageMath.GetPageOffset(32, pageSize));

        Assert.Equal(2, KvPageMath.GetPageIndex(64, pageSize));

        Assert.Equal(0, KvPageMath.GetRequiredPageCount(0, pageSize));
        Assert.Equal(1, KvPageMath.GetRequiredPageCount(1, pageSize));
        Assert.Equal(1, KvPageMath.GetRequiredPageCount(31, pageSize));
        Assert.Equal(1, KvPageMath.GetRequiredPageCount(32, pageSize));
        Assert.Equal(2, KvPageMath.GetRequiredPageCount(33, pageSize));
        Assert.Equal(2, KvPageMath.GetRequiredPageCount(64, pageSize));
        Assert.Equal(3, KvPageMath.GetRequiredPageCount(65, pageSize));
    }

    [Fact]
    public void KvPageMath_NegativeOrZeroArguments_Throw()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => KvPageMath.GetPageIndex(0, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => KvPageMath.GetPageIndex(-1, 32));
        Assert.Throws<ArgumentOutOfRangeException>(() => KvPageMath.GetPageOffset(0, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => KvPageMath.GetPageOffset(-1, 32));
        Assert.Throws<ArgumentOutOfRangeException>(() => KvPageMath.GetRequiredPageCount(1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => KvPageMath.GetRequiredPageCount(1, -1));
    }

    [Fact]
    public void CpuKvCache_Constructor_InvalidArguments_Throw()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new CpuKvCache(totalPages: 0, pageSizeTokens: 32));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CpuKvCache(totalPages: -1, pageSizeTokens: 32));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CpuKvCache(totalPages: 10, pageSizeTokens: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CpuKvCache(totalPages: 10, pageSizeTokens: -1));
    }

    [Fact]
    public void CpuKvCache_AllocationAndPageCount_MatchesTokenGrowth()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        using var seq = cache.AllocateSequence();

        Assert.Equal(0, seq.TokenCount);
        Assert.Equal(0, seq.PageCount);
        Assert.Equal(100, cache.FreePages);
        Assert.Equal(0, cache.UsedPages);

        seq.Append(15);
        Assert.Equal(15, seq.TokenCount);
        Assert.Equal(1, seq.PageCount);
        Assert.Equal(99, cache.FreePages);
        Assert.Equal(1, cache.UsedPages);

        seq.Append(17); // Total 32 tokens -> 1 page
        Assert.Equal(32, seq.TokenCount);
        Assert.Equal(1, seq.PageCount);

        seq.Append(1); // Total 33 tokens -> 2 pages
        Assert.Equal(33, seq.TokenCount);
        Assert.Equal(2, seq.PageCount);
        Assert.Equal(98, cache.FreePages);
        Assert.Equal(2, cache.UsedPages);
    }

    [Fact]
    public void CpuKvCache_ForkAndRefCounting_SharesPagesCorrectly()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        using var parent = cache.AllocateSequence();

        parent.Append(64); // 2 pages (0, 1)
        Assert.Equal(2, parent.PageCount);
        Assert.Equal(0, cache.SharedPages);

        using var child = parent.Fork();
        Assert.Equal(64, child.TokenCount);
        Assert.Equal(2, child.PageCount);
        Assert.Equal(2, cache.SharedPages);

        // Verify page IDs match between parent and child
        Assert.Equal(parent.Pages[0], child.Pages[0]);
        Assert.Equal(parent.Pages[1], child.Pages[1]);

        // Releasing child decreases shared pages
        child.Release();
        Assert.Equal(0, cache.SharedPages);
        Assert.Equal(2, cache.UsedPages);

        // Releasing parent frees physical pages back to pool
        parent.Release();
        Assert.Equal(0, cache.UsedPages);
        Assert.Equal(100, cache.FreePages);
    }

    [Fact]
    public void CpuKvCache_CopyOnWrite_DuplicatesSharedUnalignedPageOnAppend()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        using var parent = cache.AllocateSequence();

        parent.Append(15); // 1 page (unaligned)
        using var child = parent.ForkAt(15); // Shared page 0

        Assert.Equal(1, cache.SharedPages);
        Assert.Equal(parent.Pages[0], child.Pages[0]);

        // Appending to child on unaligned shared page triggers Copy-On-Write
        child.Append(10); // Total 25 tokens in child

        Assert.NotEqual(parent.Pages[0], child.Pages[0]); // Child acquired private copy
        Assert.Equal(1, cache.GetStatistics().CopyOnWriteCopies);
        Assert.Equal(0, cache.SharedPages);
    }

    [Fact]
    public void CpuKvCache_Reservation_EnforcesAdmissionControlBackpressure()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 32); // Max 320 tokens

        using var res1 = cache.TryReserve(sequenceId: 1, requiredTokens: 200);
        Assert.NotNull(res1);
        Assert.Equal(200, res1!.ReservedTokens);

        using var res2 = cache.TryReserve(sequenceId: 2, requiredTokens: 150);
        // Total requested 350 > capacity 320 -> should fail admission
        Assert.Null(res2);

        // After releasing res1, res2 can be reserved
        res1.Dispose();

        using var res3 = cache.TryReserve(sequenceId: 2, requiredTokens: 150);
        Assert.NotNull(res3);
    }

    [Fact]
    public void CpuKvSequence_TruncateTo_ReleasesPagesBackToPool()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        using var seq = cache.AllocateSequence();

        seq.Append(65); // 3 pages: 32 + 32 + 1
        Assert.Equal(3, seq.PageCount);
        Assert.Equal(97, cache.FreePages);

        seq.TruncateTo(32); // exactly 1 page remains
        Assert.Equal(32, seq.TokenCount);
        Assert.Equal(1, seq.PageCount);
        Assert.Equal(99, cache.FreePages);
    }

    [Fact]
    public void CpuKvSequence_TruncateTo_PartialPage_RetainsPageButReducesTokenCount()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        using var seq = cache.AllocateSequence();

        seq.Append(40); // 2 pages
        Assert.Equal(2, seq.PageCount);

        seq.TruncateTo(20); // still needs 1 page (ceil(20/32) == 1)
        Assert.Equal(20, seq.TokenCount);
        Assert.Equal(1, seq.PageCount);
        Assert.Equal(99, cache.FreePages);
    }

    [Fact]
    public void CpuKvSequence_TruncateTo_SameTokenCount_IsNoOp()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 32);
        using var seq = cache.AllocateSequence();
        seq.Append(10);
        var pageBefore = seq.Pages[0];

        seq.TruncateTo(10);

        Assert.Equal(10, seq.TokenCount);
        Assert.Equal(1, seq.PageCount);
        Assert.Equal(pageBefore, seq.Pages[0]);
    }

    [Fact]
    public void CpuKvSequence_TruncateTo_OutOfRange_Throws()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 32);
        using var seq = cache.AllocateSequence();
        seq.Append(10);

        Assert.Throws<ArgumentOutOfRangeException>(() => seq.TruncateTo(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => seq.TruncateTo(11));
    }

    [Fact]
    public void CpuKvSequence_Clear_ReleasesAllPagesAndResetsTokenCount()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 32);
        using var seq = cache.AllocateSequence();
        seq.Append(50); // 2 pages
        Assert.Equal(8, cache.FreePages);

        seq.Clear();

        Assert.Equal(0, seq.TokenCount);
        Assert.Equal(0, seq.PageCount);
        Assert.Equal(10, cache.FreePages);
    }

    [Fact]
    public void CpuKvSequence_ForkAt_OutOfRange_Throws()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 32);
        using var seq = cache.AllocateSequence();
        seq.Append(10);

        Assert.Throws<ArgumentOutOfRangeException>(() => seq.ForkAt(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => seq.ForkAt(11));
    }

    [Fact]
    public void CpuKvSequence_Append_NegativeTokenCount_Throws()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 32);
        using var seq = cache.AllocateSequence();

        Assert.Throws<ArgumentOutOfRangeException>(() => seq.Append(-1));
    }

    /// <summary>
    /// Every other Fork/COW test here only checks bookkeeping (counts, page ids). This checks the
    /// thing that actually matters: the physical page bytes. A shared page must show identical
    /// content through either handle, and copy-on-write must hand the child a genuine byte-for-byte
    /// copy -- not an aliased buffer and not a zeroed one.
    /// </summary>
    [Fact]
    public void CpuKvCache_ForkedPageData_SurvivesCopyOnWrite_WithoutAliasingParent()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 32);
        using var parent = cache.AllocateSequence();
        parent.Append(15); // 1 unaligned page

        var parentPageId = parent.Pages[0];
        var parentBuffer = cache.GetPageBuffer(parentPageId)!;
        parentBuffer[0] = 0xAB;
        parentBuffer[1] = 0xCD;

        using var child = parent.ForkAt(15); // shares the same physical page
        var childPageIdBeforeCow = child.Pages[0];
        Assert.Equal(parentPageId, childPageIdBeforeCow);

        var childBufferBeforeCow = cache.GetPageBuffer(childPageIdBeforeCow)!;
        Assert.Equal((byte)0xAB, childBufferBeforeCow[0]);
        Assert.Equal((byte)0xCD, childBufferBeforeCow[1]);

        // Appending to the child on the shared unaligned page must trigger COW: the child gets a
        // private page whose initial content is a byte-for-byte copy of the parent's page.
        child.Append(5);

        var childPageIdAfterCow = child.Pages[0];
        Assert.NotEqual(parentPageId, childPageIdAfterCow);

        var childBufferAfterCow = cache.GetPageBuffer(childPageIdAfterCow)!;
        Assert.Equal((byte)0xAB, childBufferAfterCow[0]);
        Assert.Equal((byte)0xCD, childBufferAfterCow[1]);

        // Mutating the child's private post-COW page must not affect the parent's original page.
        childBufferAfterCow[0] = 0xFF;
        Assert.Equal((byte)0xAB, parentBuffer[0]);
    }

    [Fact]
    public void CpuKvCache_TryReserveSequences_AggregatesPagesAcrossSequences_RespectsCapacity()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 32); // 320 tokens capacity

        // Two sequences of 100 tokens each need ceil(100/32) = 4 pages each = 8 pages total.
        var res = cache.TryReserveSequences(sequenceId: 1, new[] { 100, 100 });
        Assert.NotNull(res);
        Assert.Equal(200, res!.ReservedTokens);

        // A third sequence needing 3 more pages: 8 + 3 = 11 > 10 available -> rejected.
        using var res2 = cache.TryReserveSequences(sequenceId: 2, new[] { 65 }); // ceil(65/32) = 3 pages
        Assert.Null(res2);

        res.Dispose();

        using var res3 = cache.TryReserveSequences(sequenceId: 2, new[] { 65 });
        Assert.NotNull(res3);
    }

    [Fact]
    public void CpuKvCache_TryReserveSequences_EmptyOrNullList_ReturnsNull()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 32);

        Assert.Null(cache.TryReserveSequences(1, Array.Empty<int>()));
        Assert.Null(cache.TryReserveSequences(1, null!));
    }

    [Fact]
    public void CpuKvReservation_TryGrow_WithinSamePage_DoesNotConsumeAdditionalPages()
    {
        using var cache = new CpuKvCache(totalPages: 5, pageSizeTokens: 32);
        using var res = cache.TryReserve(sequenceId: 1, requiredTokens: 10); // 1 page
        Assert.NotNull(res);

        Assert.True(res!.TryGrow(5)); // 15 tokens still fits in 1 page
        Assert.Equal(15, res.ReservedTokens);

        // The other 4 pages must still be free for a second sequence.
        using var res2 = cache.TryReserve(sequenceId: 2, requiredTokens: 128); // 4 pages
        Assert.NotNull(res2);
    }

    [Fact]
    public void CpuKvReservation_TryGrow_AcrossPageBoundary_ReservesAdditionalPage()
    {
        using var cache = new CpuKvCache(totalPages: 2, pageSizeTokens: 32);
        using var res = cache.TryReserve(sequenceId: 1, requiredTokens: 20); // 1 page
        Assert.NotNull(res);

        Assert.True(res!.TryGrow(20)); // 40 tokens total -> needs a 2nd page
        Assert.Equal(40, res.ReservedTokens);

        // No pages left for another sequence.
        using var res2 = cache.TryReserve(sequenceId: 2, requiredTokens: 1);
        Assert.Null(res2);
    }

    [Fact]
    public void CpuKvReservation_TryGrow_ExceedsCapacity_ReturnsFalseAndLeavesReservationUnchanged()
    {
        using var cache = new CpuKvCache(totalPages: 1, pageSizeTokens: 32);
        using var res = cache.TryReserve(sequenceId: 1, requiredTokens: 20);
        Assert.NotNull(res);

        Assert.False(res!.TryGrow(20)); // would need a 2nd page; capacity is only 1 page total
        Assert.Equal(20, res.ReservedTokens); // unchanged on a failed grow
    }

    [Fact]
    public void CpuKvReservation_TryGrow_NonPositiveAmount_ReturnsTrueWithoutChange()
    {
        using var cache = new CpuKvCache(totalPages: 1, pageSizeTokens: 32);
        using var res = cache.TryReserve(sequenceId: 1, requiredTokens: 20);
        Assert.NotNull(res);

        Assert.True(res!.TryGrow(0));
        Assert.Equal(20, res.ReservedTokens);
    }

    [Fact]
    public void CpuKvCache_StressAllocationRelease_NoMemoryLeaks()
    {
        using var cache = new CpuKvCache(totalPages: 50, pageSizeTokens: 32);

        for (int i = 0; i < 1000; i++)
        {
            using var seq = cache.AllocateSequence();
            seq.Append(100);

            using var child = seq.ForkAt(64);
            child.Append(20);

            child.Release();
            seq.Release();
        }

        Assert.Equal(50, cache.FreePages);
        Assert.Equal(0, cache.UsedPages);
        Assert.Equal(0, cache.SharedPages);
    }

    [Fact]
    public void LegacySequenceKvCacheAdapter_WrapsAndTracksTokenCount()
    {
        using var fakeInner = new FakeLegacyCache();
        using var adapter = new LegacySequenceKvCacheAdapter(fakeInner, pageSize: 32);

        Assert.Equal(0, adapter.TokenCount);
        Assert.Equal(0, adapter.PageCount);

        adapter.Append(50);
        Assert.Equal(50, adapter.TokenCount);
        Assert.Equal(2, adapter.PageCount);
        Assert.Equal(0, adapter.Pages[0].Value);
        Assert.Equal(1, adapter.Pages[1].Value);
    }

    [Fact]
    public void LegacySequenceKvCacheAdapter_Constructor_NullInnerCache_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new LegacySequenceKvCacheAdapter(null!, pageSize: 32));
    }

    [Fact]
    public void LegacySequenceKvCacheAdapter_TruncateTo_UpdatesTokenCountAndPageCount()
    {
        using var fakeInner = new FakeLegacyCache();
        using var adapter = new LegacySequenceKvCacheAdapter(fakeInner, pageSize: 32);

        adapter.Append(50); // 2 pages
        Assert.Equal(2, adapter.PageCount);

        adapter.TruncateTo(10); // still 1 page
        Assert.Equal(10, adapter.TokenCount);
        Assert.Equal(1, adapter.PageCount);
    }

    [Fact]
    public void LegacySequenceKvCacheAdapter_TruncateTo_OutOfRange_Throws()
    {
        using var fakeInner = new FakeLegacyCache();
        using var adapter = new LegacySequenceKvCacheAdapter(fakeInner, pageSize: 32);
        adapter.Append(10);

        Assert.Throws<ArgumentOutOfRangeException>(() => adapter.TruncateTo(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => adapter.TruncateTo(11));
    }

    [Fact]
    public void LegacySequenceKvCacheAdapter_Clear_ResetsTokenCountAndPages()
    {
        using var fakeInner = new FakeLegacyCache();
        using var adapter = new LegacySequenceKvCacheAdapter(fakeInner, pageSize: 32);
        adapter.Append(50);

        adapter.Clear();

        Assert.Equal(0, adapter.TokenCount);
        Assert.Equal(0, adapter.PageCount);
    }

    [Fact]
    public void LegacySequenceKvCacheAdapter_Fork_ThrowsNotSupported()
    {
        using var fakeInner = new FakeLegacyCache();
        using var adapter = new LegacySequenceKvCacheAdapter(fakeInner, pageSize: 32);
        adapter.Append(10);

        Assert.Throws<NotSupportedException>(() => adapter.Fork());
    }

    [Fact]
    public void LegacySequenceKvCacheAdapter_ForkAt_ThrowsNotSupported()
    {
        using var fakeInner = new FakeLegacyCache();
        using var adapter = new LegacySequenceKvCacheAdapter(fakeInner, pageSize: 32);
        adapter.Append(10);

        Assert.Throws<NotSupportedException>(() => adapter.ForkAt(5));
    }

    [Fact]
    public void LegacySequenceKvCacheAdapter_Dispose_OwnsInnerCache_DisposesInner()
    {
        var fakeInner = new FakeLegacyCache();
        var adapter = new LegacySequenceKvCacheAdapter(fakeInner, pageSize: 32, ownsInnerCache: true);

        adapter.Dispose();

        Assert.True(fakeInner.Disposed);
    }

    [Fact]
    public void LegacySequenceKvCacheAdapter_Dispose_DoesNotOwnInnerCache_LeavesInnerCacheOpen()
    {
        var fakeInner = new FakeLegacyCache();
        var adapter = new LegacySequenceKvCacheAdapter(fakeInner, pageSize: 32, ownsInnerCache: false);

        adapter.Dispose();

        Assert.False(fakeInner.Disposed);
        fakeInner.Dispose();
    }

    [Fact]
    public void CpuKvCache_SpeculativeTransaction_CommitAndRollback_PreservesCommittedState()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        using var targetSeq = cache.AllocateSequence();

        // 1. Initial committed prompt: 50 tokens
        targetSeq.Append(50);
        Assert.Equal(50, targetSeq.TokenCount);

        // 2. Begin speculative transaction: draft proposes 5 tokens
        using var specDraft = targetSeq.ForkAt(50);
        specDraft.Append(5); // Propose 5 tokens

        Assert.Equal(55, specDraft.TokenCount);
        Assert.Equal(50, targetSeq.TokenCount);

        // 3. Verification accepts 3 draft tokens and rejects 2 tokens -> commit 3 tokens
        targetSeq.Append(3);
        Assert.Equal(53, targetSeq.TokenCount);

        // 4. Discard speculative draft sequence (rollback unaccepted speculative pages)
        specDraft.Release();

        // 5. Verify target sequence state: exactly 53 tokens, no leak
        Assert.Equal(53, targetSeq.TokenCount);
        Assert.Equal(0, cache.SharedPages);
    }

    private sealed class FakeLegacyCache : ISequenceKvCache
    {
        public bool Disposed { get; private set; }
        public void Dispose() => Disposed = true;
    }
}
