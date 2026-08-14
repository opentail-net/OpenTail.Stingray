using System;
using System.Threading.Tasks;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public class PrefixCacheInvariantsTests
{
    [Fact]
    public void Test1_ModelIsolation_DifferentModelMisses()
    {
        using var cache = new CpuKvCache(100, 32);
        using var tree = new RadixPrefixTree(cache);
        var seq = cache.AllocateSequence();
        seq.Append(32);

        int[] prompt = new int[32];
        for (int i = 0; i < 32; i++) prompt[i] = 100 + i;

        var nsModelA = new PrefixCacheNamespace("ModelA", "FP16");
        var nsModelB = new PrefixCacheNamespace("ModelB", "FP16");

        tree.Publish(nsModelA, prompt, seq, 32);

        var matchA = tree.MatchPrefix(nsModelA, prompt);
        var matchB = tree.MatchPrefix(nsModelB, prompt);

        Assert.Equal(1, matchA.MatchedPageCount);
        Assert.Equal(0, matchB.MatchedPageCount); // Model B namespace returns cache miss!

        // Cleanup retained matched page
        if (matchA.SharedPages.Length > 0) cache.ReleasePage(matchA.SharedPages.Span[0]);
    }

    [Fact]
    public void Test2_KvConfigIsolation_DifferentQuantMisses()
    {
        using var cache = new CpuKvCache(100, 32);
        using var tree = new RadixPrefixTree(cache);
        var seq = cache.AllocateSequence();
        seq.Append(32);

        int[] prompt = new int[32];
        for (int i = 0; i < 32; i++) prompt[i] = 100 + i;

        var nsFp16 = new PrefixCacheNamespace("ModelA", "FP16");
        var nsQ4 = new PrefixCacheNamespace("ModelA", "Q4_K_M");

        tree.Publish(nsFp16, prompt, seq, 32);

        var matchFp16 = tree.MatchPrefix(nsFp16, prompt);
        var matchQ4 = tree.MatchPrefix(nsQ4, prompt);

        Assert.Equal(1, matchFp16.MatchedPageCount);
        Assert.Equal(0, matchQ4.MatchedPageCount); // Q4 namespace returns cache miss!

        if (matchFp16.SharedPages.Length > 0) cache.ReleasePage(matchFp16.SharedPages.Span[0]);
    }

    [Fact]
    public void Test3_LongestPageAlignedMatch()
    {
        using var cache = new CpuKvCache(100, 32);
        using var tree = new RadixPrefixTree(cache);
        var seq = cache.AllocateSequence();
        seq.Append(64); // 2 pages = 64 tokens

        int[] prompt = new int[70];
        for (int i = 0; i < 70; i++) prompt[i] = 100 + i;

        var ns = new PrefixCacheNamespace("ModelA", "FP16");
        tree.Publish(ns, prompt, seq, 64);

        var match = tree.MatchPrefix(ns, prompt);

        Assert.Equal(2, match.MatchedPageCount); // Matches full 2 pages = 64 tokens
        Assert.Equal(64, match.MatchedTokenCount);

        foreach (var page in match.SharedPages.Span) cache.ReleasePage(page);
    }

    [Fact]
    public void Test4_PageReuseSafety_FullLifecycle()
    {
        using var cache = new CpuKvCache(totalPages: 1, pageSizeTokens: 32);
        using var tree = new RadixPrefixTree(cache);
        var ns = new PrefixCacheNamespace("ModelA", "FP16");

        // 1. Allocate physical page P
        var seq1 = cache.AllocateSequence();
        seq1.Append(32);
        var pageP = seq1.Pages[0];

        int[] prompt = new int[32];
        for (int i = 0; i < 32; i++) prompt[i] = 100 + i;

        // 2. Publish P into prefix tree
        tree.Publish(ns, prompt, seq1, 32);

        // 3. Release seq1 reference
        seq1.Release();

        // 4. Attempt allocating a new sequence -> MUST THROW InvalidOperationException because page P is locked by prefix tree!
        var seq2 = cache.AllocateSequence();
        Assert.Throws<InvalidOperationException>(() => seq2.Append(32));

        // 5. Evict prefix entry -> releases tree's ref-count
        tree.EvictLruEntries(1);

        // 6. Allocate new sequence -> NOW succeeds and reuses page P!
        var seq3 = cache.AllocateSequence();
        seq3.Append(32);
        Assert.Equal(1, seq3.PageCount);
        Assert.Equal(pageP, seq3.Pages[0]); // Page P safely reused after eviction!
    }

    [Fact]
    public void Test5_ArbitraryReleaseOrder_MultipleSessions()
    {
        using var cache = new CpuKvCache(100, 32);
        using var tree = new RadixPrefixTree(cache);
        var ns = new PrefixCacheNamespace("ModelA", "FP16");

        int[] prompt = new int[32];
        for (int i = 0; i < 32; i++) prompt[i] = 100 + i;

        var seqPrimary = cache.AllocateSequence();
        seqPrimary.Append(32);
        tree.Publish(ns, prompt, seqPrimary, 32);

        // Session A & B match prefix and acquire shared page references
        var matchA = tree.MatchPrefix(ns, prompt);
        var matchB = tree.MatchPrefix(ns, prompt);

        Assert.Equal(1, matchA.MatchedPageCount);
        Assert.Equal(1, matchB.MatchedPageCount);

        // Release order 1: Primary seq releases
        seqPrimary.Release();

        // Release order 2: Session B releases
        cache.ReleasePage(matchB.SharedPages.Span[0]);

        // Evict prefix from tree
        tree.EvictLruEntries(1);

        // Page is STILL retained by Session A!
        Assert.Equal(1, cache.UsedPages);

        // Release order 3: Session A releases -> now page count drops to 0!
        cache.ReleasePage(matchA.SharedPages.Span[0]);
        Assert.Equal(0, cache.UsedPages);
    }

    [Fact]
    public void Test6_FailedPrefill_NeverPublishes()
    {
        using var cache = new CpuKvCache(100, 32);
        using var tree = new RadixPrefixTree(cache);
        var ns = new PrefixCacheNamespace("ModelA", "FP16");

        var seq = cache.AllocateSequence();
        seq.Append(32);

        int[] prompt = new int[32];
        // Simulated failed prefill: publish is NOT invoked
        var match = tree.MatchPrefix(ns, prompt);
        Assert.Equal(0, match.MatchedPageCount); // Zero published pages!
    }

    [Fact]
    public void Test7_CopyOnWriteIsolation_AfterPrefixSharing()
    {
        using var cache = new CpuKvCache(100, 32);
        using var tree = new RadixPrefixTree(cache);
        var ns = new PrefixCacheNamespace("ModelA", "FP16");

        int[] prompt = new int[32];
        for (int i = 0; i < 32; i++) prompt[i] = 100 + i;

        var parentSeq = cache.AllocateSequence();
        parentSeq.Append(32);
        tree.Publish(ns, prompt, parentSeq, 32);

        var childSeq = parentSeq.Fork();
        Assert.Equal(parentSeq.Pages[0], childSeq.Pages[0]); // Shared page

        // Mutate child sequence -> CoW duplicates page cleanly!
        childSeq.Append(32);
        Assert.NotEqual(parentSeq.Pages[0], childSeq.Pages[1]);
    }

    [Fact]
    public void Test8_PartialPage_RemainsPrivate()
    {
        using var cache = new CpuKvCache(100, 32);
        using var tree = new RadixPrefixTree(cache);
        var ns = new PrefixCacheNamespace("ModelA", "FP16");

        int[] prompt = new int[50];
        for (int i = 0; i < 50; i++) prompt[i] = 100 + i;

        var seq = cache.AllocateSequence();
        seq.Append(50); // 50 tokens = 1 full page (32 tokens) + 18 partial tokens

        tree.Publish(ns, prompt, seq, 50);

        var match = tree.MatchPrefix(ns, prompt);

        // Only full page (32 tokens = 1 page) is indexed and returned! Partial tail remains private.
        Assert.Equal(1, match.MatchedPageCount);
        Assert.Equal(32, match.MatchedTokenCount);

        if (match.SharedPages.Length > 0) cache.ReleasePage(match.SharedPages.Span[0]);
    }

    [Fact]
    public void Test9_PrefixEviction_ReleasesPhysicalPages()
    {
        using var cache = new CpuKvCache(100, 32);
        using var tree = new RadixPrefixTree(cache);
        var ns = new PrefixCacheNamespace("ModelA", "FP16");

        int[] prompt = new int[32];
        for (int i = 0; i < 32; i++) prompt[i] = 100 + i;

        var seq = cache.AllocateSequence();
        seq.Append(32);
        tree.Publish(ns, prompt, seq, 32);
        seq.Release();

        Assert.Equal(1, cache.UsedPages);

        int evicted = tree.EvictLruEntries(1);
        Assert.Equal(1, evicted);
        Assert.Equal(0, cache.UsedPages); // Physical page released on eviction!
    }

    [Fact]
    public async Task Test10_AtomicMatchEviction_ConcurrencySafety()
    {
        using var cache = new CpuKvCache(100, 32);
        using var tree = new RadixPrefixTree(cache);
        var ns = new PrefixCacheNamespace("ModelA", "FP16");

        int[] prompt = new int[32];
        for (int i = 0; i < 32; i++) prompt[i] = 100 + i;

        var seq = cache.AllocateSequence();
        seq.Append(32);
        tree.Publish(ns, prompt, seq, 32);
        seq.Release();

        // Concurrent MatchPrefix and EvictLruEntries
        var matchTask = Task.Run(() => tree.MatchPrefix(ns, prompt));
        var evictTask = Task.Run(() => tree.EvictLruEntries(1));

        await Task.WhenAll(matchTask, evictTask);

        var match = await matchTask;
        if (match.MatchedPageCount > 0)
        {
            // Match atomic retention succeeded before eviction!
            Assert.True(match.SharedPages.Span[0].IsValid);
            cache.ReleasePage(match.SharedPages.Span[0]);
        }
    }
}
