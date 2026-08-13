using System;
using System.Threading.Tasks;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions;

public sealed class KvCorrectnessInvariantsTests
{
    [Fact]
    public void Append_AllocationFailure_ReleasesPagesAllocatedByAppend()
    {
        // 3 pages capacity total
        using var cache = new CpuKvCache(totalPages: 3, pageSizeTokens: 16);
        using var seq = cache.AllocateSequence();

        // Fill 1 page (16 tokens)
        seq.Append(16);
        Assert.Equal(1, seq.PageCount);
        Assert.Equal(2, cache.FreePages);

        // Appending 48 tokens requires 3 additional pages (4 total), but only 2 free pages remain.
        Assert.Throws<InvalidOperationException>(() => seq.Append(48));

        // Invariant check: original 1 page remains intact, and the 2 newly allocated pages during Append
        // were transactionally rolled back and returned to cache.FreePages!
        Assert.Equal(1, seq.PageCount);
        Assert.Equal(16, seq.TokenCount);
        Assert.Equal(2, cache.FreePages);
    }

    [Fact]
    public void TryGrow_AtCapacity_ReturnsFalseWithoutChangingReservation()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 16);
        using var reservation = cache.TryReserve(sequenceId: 1, requiredTokens: 160); // reserves all 10 pages (160 tokens)
        Assert.NotNull(reservation);
        Assert.Equal(160, reservation.ReservedTokens);

        // Attempting to grow beyond 160 tokens when 0 unreserved pages remain must return false
        bool grown = reservation.TryGrow(16);

        Assert.False(grown);
        Assert.Equal(160, reservation.ReservedTokens);
        Assert.Equal(160, (int)(cache.GetStatistics().ReservedBytes / cache.BytesPerToken));
    }

    [Fact]
    public async Task TryGrow_ConcurrentCallers_NeverExceedCapacity()
    {
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 16); // 160 tokens total capacity
        using var reservation = cache.TryReserve(sequenceId: 1, requiredTokens: 16); // 16 tokens initial (1 page)

        Assert.NotNull(reservation);

        // 10 concurrent threads each trying to grow by 32 tokens (total requested = 320 tokens, but capacity is 160)
        int successfulGrows = 0;
        Task[] tasks = new Task[10];

        for (int i = 0; i < 10; i++)
        {
            tasks[i] = Task.Run(() =>
            {
                if (reservation.TryGrow(32))
                {
                    System.Threading.Interlocked.Increment(ref successfulGrows);
                }
            });
        }

        await Task.WhenAll(tasks);

        // 16 initial + (4 grow * 32) = 144 tokens <= 160 capacity
        Assert.True(reservation.ReservedTokens <= 160);
        Assert.True((int)(cache.GetStatistics().ReservedBytes / cache.BytesPerToken) <= 160);
    }

    [Fact]
    public void CpuKvSequence_Append_WhenCowSucceedsButLaterAllocationFails_RollsBackCompletely()
    {
        // Total capacity = 2 pages
        using var cache = new CpuKvCache(totalPages: 2, pageSizeTokens: 16);

        // Sequence A owns Page 0 with 8 tokens (unaligned tail)
        using var seqA = cache.AllocateSequence();
        seqA.Append(8);
        Assert.Equal(1, seqA.PageCount);
        Assert.Equal(1, cache.FreePages);

        // Fork Sequence B (shares Page 0 with A)
        using var seqB = seqA.Fork();
        Assert.Equal(1, seqB.PageCount);
        Assert.True(cache.IsPageShared(seqA.Pages[0]));
        Assert.Equal(1, cache.FreePages); // Page 1 is free

        var originalPageId = seqA.Pages[0];

        // Append 24 tokens to seqA:
        // Needs 1 COW page (duplicating Page 0 to Page 1) AND 1 new page (for tokens 17-32).
        // COW will succeed (allocating Page 1), but the next page allocation will fail because 0 free pages remain.
        Assert.Throws<InvalidOperationException>(() => seqA.Append(24));

        // Invariant check: seqA's page table and token count must be rolled back completely to pre-Append state!
        Assert.Equal(1, seqA.PageCount);
        Assert.Equal(8, seqA.TokenCount);
        Assert.Equal(originalPageId, seqA.Pages[0]);
        Assert.True(cache.IsPageShared(originalPageId));
        Assert.Equal(1, cache.FreePages); // COW page returned to free pool!
    }

    [Fact]
    public void TryReserve_PerSequencePageRounding_IsCorrect()
    {
        // 10 pages total, 32 tokens per page
        using var cache = new CpuKvCache(totalPages: 10, pageSizeTokens: 32);

        // Three 20-token reservations (each 20 tokens rounds to 1 full 32-token page = 3 pages total)
        using var res1 = cache.TryReserve(sequenceId: 1, requiredTokens: 20);
        using var res2 = cache.TryReserve(sequenceId: 2, requiredTokens: 20);
        using var res3 = cache.TryReserve(sequenceId: 3, requiredTokens: 20);

        Assert.NotNull(res1);
        Assert.NotNull(res2);
        Assert.NotNull(res3);

        // Reserve another sequence of 224 tokens (7 pages) -> 3 + 7 = 10 pages total capacity
        using var res4 = cache.TryReserve(sequenceId: 4, requiredTokens: 224);
        Assert.NotNull(res4);

        // Attempting to reserve another 1 token when 0 unreserved pages remain must return null (fail closed admission control)
        using var res5 = cache.TryReserve(sequenceId: 5, requiredTokens: 1);
        Assert.Null(res5);
    }

    [Fact]
    public async Task AppendAsync_KvAllocationFailure_DoesNotModifyTokenHistory()
    {
        // 1 page capacity total (16 tokens)
        using var cache = new CpuKvCache(totalPages: 1, pageSizeTokens: 16);
        await using var session = new InferenceSession(cache);

        // Fill initial 16 tokens (using all 1 page)
        await session.AppendAsync(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
        Assert.Equal(16, session.TokenHistory.Count);
        Assert.Equal(16, session.KvSequence.TokenCount);

        // Attempting to append 1 extra token requires page 2, which fails
        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await session.AppendAsync(new int[] { 17 });
        });

        // Invariant check: TokenHistory.Count == KvSequence.TokenCount remains strictly 16!
        Assert.Equal(16, session.TokenHistory.Count);
        Assert.Equal(16, session.KvSequence.TokenCount);
        Assert.Equal(session.TokenHistory.Count, session.KvSequence.TokenCount);
    }

    [Fact]
    public async Task AppendAsync_KvAllocationFailure_RollsBackForwardPassAndHistory()
    {
        using var cache = new CpuKvCache(totalPages: 1, pageSizeTokens: 16);
        using var fwd = new TestForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        // Fill initial 16 tokens
        await session.AppendAsync(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
        Assert.Equal(16, fwd.LastTruncatedPos);

        // Append 1 extra token (fails due to out of pages)
        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await session.AppendAsync(new int[] { 17 });
        });

        // Invariant check: ForwardPass, TokenHistory, and KvSequence all remain at 16!
        Assert.Equal(16, session.TokenHistory.Count);
        Assert.Equal(16, session.KvSequence.TokenCount);
        Assert.Equal(16, fwd.LastTruncatedPos);
    }

    [Fact]
    public async Task AppendAsync_PrefillFailure_RestoresHistoryKvAndForwardPass()
    {
        // 50 pages capacity (plenty of KV memory)
        using var cache = new CpuKvCache(totalPages: 50, pageSizeTokens: 16);
        using var fwd = new FaultyForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        // Fill initial 5 tokens (prefill succeeds)
        await session.AppendAsync(new int[] { 1, 2, 3, 4, 5 });
        Assert.Equal(5, session.TokenHistory.Count);
        Assert.Equal(5, session.KvSequence.TokenCount);
        Assert.Equal(5, fwd.LastTruncatedPos);

        // Next prefill will throw exception!
        fwd.ShouldThrowOnPrefill = true;

        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await session.AppendAsync(new int[] { 6, 7, 8 });
        });

        // Invariant check: Prefill failed, so ALL 3 components must be restored to exactly 5!
        Assert.Equal(5, session.TokenHistory.Count);
        Assert.Equal(5, session.KvSequence.TokenCount);
        Assert.Equal(5, fwd.LastTruncatedPos);
        Assert.Equal(session.TokenHistory.Count, session.KvSequence.TokenCount);
    }

    /// <summary>
    /// Invariant test: reserve → allocate actual pages → grow reservation → release actual pages → grow again.
    /// Proves reservation accounting cannot permanently starve capacity after pages are returned to the pool.
    ///
    /// Accounting model recap:
    ///   TryGrow checks:  FreePages - _reservedPages >= additionalPagesNeeded
    ///   Actual allocation physically reduces FreePages (pages leave the free pool).
    ///   Releasing those pages physically increases FreePages (pages re-enter the free pool).
    ///   Therefore releasing actual pages MUST restore TryGrow capacity.
    /// </summary>
    [Fact]
    public void TryGrow_ReserveAllocateGrowReleaseGrow_CapacityIsRecovered()
    {
        // 6 pages total, 16 tokens per page (96 tokens capacity)
        using var cache = new CpuKvCache(totalPages: 6, pageSizeTokens: 16);

        // Step 1 — Reserve 2 pages (32 tokens) for a long-lived sequence
        using var reservation = cache.TryReserve(sequenceId: 1, requiredTokens: 32);
        Assert.NotNull(reservation);
        Assert.Equal(32, reservation.ReservedTokens);
        // 6 total - 2 reserved = 4 unreserved free pages remain

        // Step 2 — Allocate 2 actual KV pages for a short-lived session (physically consuming 2 free pages)
        using var tempSeq = cache.AllocateSequence();
        tempSeq.Append(32); // 2 pages physically allocated
        Assert.Equal(4, cache.FreePages); // 6 - 2 physically allocated = 4 free
        // Unreserved free pages: 4 - 2 reserved = 2

        // Step 3 — Grow reservation by 2 more pages (32 tokens) — should succeed as 2 unreserved free pages remain
        bool grownFirst = reservation.TryGrow(32);
        Assert.True(grownFirst, "First TryGrow must succeed: 2 unreserved free pages are available.");
        Assert.Equal(64, reservation.ReservedTokens);
        // Now: 4 free - 4 reserved = 0 unreserved free pages

        // Step 4 — Attempting a further grow should fail: 0 unreserved free pages remain
        bool grownOverCapacity = reservation.TryGrow(16);
        Assert.False(grownOverCapacity, "TryGrow must fail when 0 unreserved free pages remain.");
        Assert.Equal(64, reservation.ReservedTokens); // unchanged

        // Step 5 — Release the physically allocated pages (tempSeq returns 2 pages to the free pool)
        tempSeq.Dispose();
        Assert.Equal(6, cache.FreePages); // all 6 pages are physically free again
        // Unreserved free pages: 6 - 4 reserved = 2

        // Step 6 — Grow reservation again: released pages must restore capacity (proving no permanent starvation)
        bool grownAfterRelease = reservation.TryGrow(16);
        Assert.True(grownAfterRelease, "TryGrow must succeed after physical pages are returned to the pool.");
        Assert.Equal(80, reservation.ReservedTokens);

        // Final sanity: reservation pages count must not exceed total capacity
        var stats = cache.GetStatistics();
        Assert.True(stats.ReservedBytes / cache.BytesPerToken <= cache.TotalPages * cache.PageSizeTokens,
            "Total reserved tokens must never exceed physical cache capacity.");
    }

    private sealed class FaultyForwardPass : OpenTail.Stingray.Core.IForwardPass
    {
        private readonly float[] _logits = new float[100];
        public bool ShouldThrowOnPrefill { get; set; }
        public int LastTruncatedPos { get; private set; }
        public int VocabSize => 100;
        public int MaxSeqLen => 512;
        public ReadOnlySpan<float> Forward(int token, int position) => _logits;
        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            if (ShouldThrowOnPrefill)
            {
                throw new InvalidOperationException("Simulated Prefill CUDA/CPU kernel failure");
            }
            LastTruncatedPos = startPos + tokens.Count;
            return _logits;
        }
        public void ResetCache() { LastTruncatedPos = 0; }
        public void TruncateTo(int newPosition) { LastTruncatedPos = newPosition; }
        public void Dispose() { }
    }

    private sealed class TestForwardPass : OpenTail.Stingray.Core.IForwardPass
    {
        private readonly float[] _logits = new float[100];
        public int LastTruncatedPos { get; private set; }
        public int VocabSize => 100;
        public int MaxSeqLen => 512;
        public ReadOnlySpan<float> Forward(int token, int position) => _logits;
        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            LastTruncatedPos = startPos + tokens.Count;
            return _logits;
        }
        public void ResetCache() { LastTruncatedPos = 0; }
        public void TruncateTo(int newPosition) { LastTruncatedPos = newPosition; }
        public void Dispose() { }
    }
}
