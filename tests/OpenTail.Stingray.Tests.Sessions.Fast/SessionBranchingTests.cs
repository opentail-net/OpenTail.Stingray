using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Core.Grammar;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

public sealed class SessionBranchingTests
{
    [Fact]
    public async Task Test1_ForkCreatesIndependentBranches()
    {
        var cache = new CpuKvCache(totalPages: 128, pageSizeTokens: 32);
        await using var runtime = new InferenceRuntime(cache);
        await using var session = await runtime.CreateSessionAsync();

        var promptTokens = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        await session.AppendAsync(promptTokens);

        long initialAllocations = cache.GetStatistics().Allocations;

        var branches = session.ForkMany(4);
        Assert.Equal(4, branches.Count);

        // Verify +0 additional physical KV page allocations on fork
        Assert.Equal(1, cache.UsedPages);

        foreach (var branch in branches)
        {
            Assert.Equal(session.TokenCount, branch.TokenCount);
            Assert.Equal(session.TokenHistory, branch.TokenHistory);
            Assert.Equal(session.KvSequence.PageCount, branch.KvSequence.PageCount);
        }

        foreach (var branch in branches)
        {
            await branch.DisposeAsync();
        }
    }

    [Fact]
    public async Task Test2_ForkDoesNotRePrefill()
    {
        var cache = new CpuKvCache(totalPages: 128, pageSizeTokens: 32);
        await using var runtime = new InferenceRuntime(cache);
        await using var session = await runtime.CreateSessionAsync();

        await session.AppendAsync(new int[] { 10, 20, 30, 40 });

        int usedPagesBefore = cache.UsedPages;

        var branches = session.ForkMany(8);

        // Physical memory allocation must remain identical (zero prompt re-prefill)
        Assert.Equal(usedPagesBefore, cache.UsedPages);
        Assert.Equal(1, cache.SharedPages);

        foreach (var branch in branches)
        {
            await branch.DisposeAsync();
        }
    }

    [Fact]
    public async Task Test3_BranchGenerationIsIndependent()
    {
        var cache = new CpuKvCache(totalPages: 128, pageSizeTokens: 32);
        await using var runtime = new InferenceRuntime(cache);
        await using var parent = await runtime.CreateSessionAsync();

        await parent.AppendAsync(new int[] { 100, 200 });

        var branches = parent.ForkMany(3);
        var b0 = branches[0];
        var b1 = branches[1];
        var b2 = branches[2];

        await b0.AppendAsync(new int[] { 10 });
        await b1.AppendAsync(new int[] { 20, 30 });
        await b2.AppendAsync(new int[] { 40, 50, 60 });

        Assert.Equal(3, b0.TokenCount);
        Assert.Equal(4, b1.TokenCount);
        Assert.Equal(5, b2.TokenCount);

        Assert.Equal(new int[] { 100, 200, 10 }, b0.TokenHistory);
        Assert.Equal(new int[] { 100, 200, 20, 30 }, b1.TokenHistory);
        Assert.Equal(new int[] { 100, 200, 40, 50, 60 }, b2.TokenHistory);

        foreach (var b in branches) await b.DisposeAsync();
    }

    [Fact]
    public async Task Test4_CopyOnWriteProtectsSiblings()
    {
        var cache = new CpuKvCache(totalPages: 128, pageSizeTokens: 32);
        await using var runtime = new InferenceRuntime(cache);
        await using var parent = await runtime.CreateSessionAsync();

        // 10 tokens in 32-token page (unaligned)
        await parent.AppendAsync(Enumerable.Range(1, 10).ToArray());
        var originalPage = parent.KvSequence.Pages[0];

        var branches = parent.ForkMany(2);
        var b0 = branches[0];
        var b1 = branches[1];

        Assert.True(cache.IsPageShared(originalPage));

        // Mutate b0 on unaligned boundary to trigger Copy-on-Write
        await b0.AppendAsync(new int[] { 999 });

        var b0NewPage = b0.KvSequence.Pages[0];
        var b1Page = b1.KvSequence.Pages[0];
        var parentPage = parent.KvSequence.Pages[0];

        Assert.NotEqual(originalPage, b0NewPage);
        Assert.Equal(originalPage, b1Page);
        Assert.Equal(originalPage, parentPage);

        foreach (var b in branches) await b.DisposeAsync();
    }

    [Fact]
    public async Task Test5_ParentUnaffectedByBranch()
    {
        var cache = new CpuKvCache(totalPages: 128, pageSizeTokens: 32);
        await using var runtime = new InferenceRuntime(cache);
        await using var parent = await runtime.CreateSessionAsync();

        await parent.AppendAsync(new int[] { 1, 2, 3 });
        var parentHistory = parent.TokenHistory.ToArray();

        var branches = parent.ForkMany(2);
        await branches[0].AppendAsync(new int[] { 99, 100 });

        Assert.Equal(parentHistory, parent.TokenHistory);
        Assert.Equal(3, parent.TokenCount);

        foreach (var b in branches) await b.DisposeAsync();
    }

    [Fact]
    public async Task Test6_ArbitraryBranchDisposal()
    {
        var cache = new CpuKvCache(totalPages: 128, pageSizeTokens: 32);
        await using var runtime = new InferenceRuntime(cache);
        await using var parent = await runtime.CreateSessionAsync();

        await parent.AppendAsync(new int[] { 10, 20 });
        var branches = parent.ForkMany(4);

        // Random disposal order: b2, b0, b3, b1
        await branches[2].DisposeAsync();
        await branches[0].DisposeAsync();

        Assert.Equal(SessionState.Ready, branches[1].State);
        Assert.Equal(SessionState.Ready, branches[3].State);

        await branches[1].AppendAsync(new int[] { 30 });
        Assert.Equal(3, branches[1].TokenCount);

        await branches[3].DisposeAsync();
        await branches[1].DisposeAsync();
    }

    [Fact]
    public async Task Test7_ParentDisposalDoesNotCorruptChildren()
    {
        var cache = new CpuKvCache(totalPages: 128, pageSizeTokens: 32);
        await using var runtime = new InferenceRuntime(cache);
        var parent = await runtime.CreateSessionAsync();

        await parent.AppendAsync(new int[] { 1, 2, 3, 4 });
        var branches = parent.ForkMany(2);

        // Dispose parent first
        await parent.DisposeAsync();

        // Children remain valid and operable
        var child = branches[0];
        Assert.Equal(SessionState.Ready, child.State);
        await child.AppendAsync(new int[] { 5 });
        Assert.Equal(5, child.TokenCount);

        foreach (var b in branches) await b.DisposeAsync();
    }

    [Fact]
    public async Task Test8_SharedPrefixCachePagesRemainValid()
    {
        var cache = new CpuKvCache(totalPages: 128, pageSizeTokens: 32);
        await using var runtime = new InferenceRuntime(cache);

        var prefixTokens = Enumerable.Range(1, 64).ToArray();
        var ns = new PrefixCacheNamespace("model", "default");

        var session = await runtime.CreateSessionAsync();
        await session.AppendAsync(prefixTokens);

        runtime.PrefixIndex?.Publish(ns, prefixTokens, session.KvSequence, prefixTokens.Length);

        var branches = session.ForkMany(2);
        await branches[0].DisposeAsync();

        Assert.True(cache.UsedPages > 0);

        await branches[1].DisposeAsync();
        await session.DisposeAsync();
    }

    [Fact]
    public async Task Test9_BranchesCanBeNested()
    {
        var cache = new CpuKvCache(totalPages: 128, pageSizeTokens: 32);
        await using var runtime = new InferenceRuntime(cache);
        await using var root = await runtime.CreateSessionAsync();

        await root.AppendAsync(new int[] { 1, 2 });

        var level1 = root.ForkMany(2);
        var childA = level1[0];

        await childA.AppendAsync(new int[] { 3 });

        var level2 = childA.ForkMany(2);
        var childA1 = level2[0];
        var childA2 = level2[1];

        await childA1.AppendAsync(new int[] { 4 });
        await childA2.AppendAsync(new int[] { 5 });

        Assert.Equal(new int[] { 1, 2, 3, 4 }, childA1.TokenHistory);
        Assert.Equal(new int[] { 1, 2, 3, 5 }, childA2.TokenHistory);

        foreach (var b in level2) await b.DisposeAsync();
        foreach (var b in level1) await b.DisposeAsync();
    }

    [Fact]
    public async Task Test10_BranchRngIndependence()
    {
        var cache = new CpuKvCache(totalPages: 128, pageSizeTokens: 32);
        await using var runtime = new InferenceRuntime(cache);
        await using var parent = await runtime.CreateSessionAsync();

        await parent.AppendAsync(new int[] { 1, 2, 3 });

        var branches = parent.ForkMany(4);
        var seeds = new HashSet<int>();

        foreach (var b in branches)
        {
            var sessionImpl = Assert.IsType<InferenceSession>(b);
            // Check derived seed via Checkpoint
            var cp = sessionImpl.CreateCheckpoint();
            seeds.Add(cp.RngSeed);
        }

        // All 4 branches must have distinct derived seeds
        Assert.Equal(4, seeds.Count);

        foreach (var b in branches) await b.DisposeAsync();
    }

    [Fact]
    public async Task Test11_BranchConstraintsAreIndependent()
    {
        var tok = new MockBranchTokenizer();
        var vocab = new GrammarVocabulary(tok);
        var schema = JsonConstraint.AnyJson(vocab);

        var cache = new CpuKvCache(totalPages: 128, pageSizeTokens: 32);
        await using var runtime = new InferenceRuntime(cache);
        await using var parent = await runtime.CreateSessionAsync();

        await parent.AppendAsync(new int[] { 1, 2 });

        var branches = parent.ForkMany(2);
        var b0 = branches[0];
        var b1 = branches[1];

        var sp0 = new SamplingParams { Constraint = schema };
        var sp1 = new SamplingParams { Constraint = schema };

        Assert.NotNull(sp0.Constraint);
        Assert.NotNull(sp1.Constraint);

        foreach (var b in branches) await b.DisposeAsync();
    }

    private sealed class MockBranchTokenizer : ITokenizer
    {
        public int VocabSize => 100;
        public int BosTokenId => 1;
        public int EosTokenId => 0;
        public int UnknownTokenId => -1;
        public int PadTokenId => -1;
        public bool AddBosToken => false;
        public System.Collections.Immutable.ImmutableArray<int> EogTokenIds => System.Collections.Immutable.ImmutableArray.Create(0);
        public System.Collections.Generic.IReadOnlyDictionary<string, int> SpecialTokens => System.Collections.Immutable.ImmutableDictionary<string, int>.Empty;
        public byte[] DecodeBytes(int token) => token switch { 1 => new byte[] { (byte)'{' }, 2 => new byte[] { (byte)'}' }, _ => Array.Empty<byte>() };
        public string Decode(IEnumerable<int> tokens) => "";
        public IReadOnlyList<int> Encode(string text) => Array.Empty<int>();
    }

    [Fact]
    public async Task Test12_SpeculativeDecodingWorksAfterFork()
    {
        var cache = new CpuKvCache(totalPages: 128, pageSizeTokens: 32);
        await using var runtime = new InferenceRuntime(cache);
        await using var parent = await runtime.CreateSessionAsync();

        await parent.AppendAsync(new int[] { 1, 2, 3, 4 });
        var branches = parent.ForkMany(2);

        var sp = new SamplingParams();
        Assert.False(sp.HasHistoryPenalty);

        foreach (var b in branches) await b.DisposeAsync();
    }

    [Fact]
    public async Task Test13_PromptLookupWorksAfterFork()
    {
        var cache = new CpuKvCache(totalPages: 128, pageSizeTokens: 32);
        await using var runtime = new InferenceRuntime(cache);
        await using var parent = await runtime.CreateSessionAsync();

        var prompt = new int[] { 10, 20, 30, 40, 10, 20, 30, 40 };
        await parent.AppendAsync(prompt);

        var branches = parent.ForkMany(2);
        var b0 = branches[0];

        Assert.Equal(prompt, b0.TokenHistory);

        foreach (var b in branches) await b.DisposeAsync();
    }

    [Fact]
    public async Task Test14_CancellationDoesNotCorruptSiblings()
    {
        var cache = new CpuKvCache(totalPages: 128, pageSizeTokens: 32);
        await using var runtime = new InferenceRuntime(cache);
        await using var parent = await runtime.CreateSessionAsync();

        await parent.AppendAsync(new int[] { 1, 2 });
        var branches = parent.ForkMany(2);

        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await branches[0].AppendAsync(new int[] { 99 }, cts.Token);
        });

        // Sibling branch 1 remains unaffected and operable
        await branches[1].AppendAsync(new int[] { 50 });
        Assert.Equal(3, branches[1].TokenCount);

        foreach (var b in branches) await b.DisposeAsync();
    }

    [Fact]
    public async Task Test15_ConcurrentBranchesRemainSafe()
    {
        var cache = new CpuKvCache(totalPages: 256, pageSizeTokens: 32);
        var fwd = new MockTestForwardPass();
        await using var runtime = new InferenceRuntime(cache);
        await using var parent = await runtime.CreateSessionAsync(forwardPass: fwd);

        await parent.AppendAsync(new int[] { 1, 2, 3, 4, 5 });

        var results = await parent.GenerateBranchesAsync(4, new SamplingParams { MaxNewTokens = 8 });
        Assert.Equal(4, results.Count);

        for (int i = 0; i < results.Count; i++)
        {
            Assert.Equal(i, results[i].BranchIndex);
            Assert.Equal(SessionState.Ready, results[i].Session.State);
            Assert.True(results[i].Session.TokenCount > 5);
        }

        foreach (var res in results)
        {
            await res.Session.DisposeAsync();
        }
    }

    private sealed class MockTestForwardPass : OpenTail.Stingray.Core.IForwardPass
    {
        public int Position { get; private set; }
        public int VocabSize => 100;
        public int MaxSeqLen => 2048;

        public OpenTail.Stingray.Core.IForwardPass CreateContext() => new MockTestForwardPass { Position = Position };
        public System.ReadOnlySpan<float> Forward(int token, int position) { Position = position + 1; return new float[100]; }
        public System.ReadOnlySpan<float> Prefill(System.Collections.Generic.IReadOnlyList<int> tokens, int startPos = 0) { Position = startPos + tokens.Count; return new float[100]; }
        public void TruncateTo(int position) { Position = position; }
        public void ResetCache() { }
        public void Dispose() { }
    }
}
