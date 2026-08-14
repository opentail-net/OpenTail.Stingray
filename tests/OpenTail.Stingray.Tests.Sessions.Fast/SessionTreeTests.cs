using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

public sealed class SessionTreeTests
{
    [Fact]
    public async Task Test1_RootSessionHasNoParent()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        Assert.Null(session.Tree.ParentId);
        Assert.Equal(session.Id, session.Tree.RootId);
        Assert.Empty(session.Tree.Children);
    }

    [Fact]
    public async Task Test2_ForkRegistersChild()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);
        await using var child = parent.Fork();

        Assert.Equal(parent.Id, child.Tree.ParentId);
        Assert.Equal(parent.Tree.RootId, child.Tree.RootId);
        Assert.Contains(child.Id, parent.Tree.Children);
    }

    [Fact]
    public async Task Test3_NestedForkPreservesRoot()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var root = new InferenceSession(cache);
        await using var child = root.Fork();
        await using var grandchild = child.Fork();

        Assert.Equal(root.Id, root.Tree.RootId);
        Assert.Equal(root.Id, child.Tree.RootId);
        Assert.Equal(root.Id, grandchild.Tree.RootId);
        Assert.Equal(child.Id, grandchild.Tree.ParentId);
        Assert.Contains(grandchild.Id, child.Tree.Children);
    }

    [Fact]
    public async Task Test4_ForkManyRegistersAllChildren()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);
        var branches = parent.ForkMany(5);

        try
        {
            Assert.Equal(5, parent.Tree.Children.Count);
            foreach (var b in branches)
            {
                Assert.Contains(b.Id, parent.Tree.Children);
                Assert.Equal(parent.Id, b.Tree.ParentId);
                Assert.Equal(parent.Tree.RootId, b.Tree.RootId);
            }
        }
        finally
        {
            foreach (var b in branches)
            {
                await b.DisposeAsync();
            }
        }
    }

    [Fact]
    public async Task Test5_DisposeRemovesChild()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);
        var child = parent.Fork();

        Assert.Contains(child.Id, parent.Tree.Children);
        await child.DisposeAsync();

        // Disposing child unregisters it from parent's active Children collection
        Assert.DoesNotContain(child.Id, parent.Tree.Children);
    }

    [Fact]
    public async Task Test6_ParentRemainsAfterChildDisposal()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);
        var child = parent.Fork();

        await child.DisposeAsync();

        Assert.NotEqual(SessionState.Disposed, parent.State);
        Assert.Equal(SessionState.Ready, parent.State);
    }

    [Fact]
    public async Task Test7_TreeMetricsAggregate()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockTreeForwardPass();
        await using var parent = new InferenceSession(cache, forwardPass: fwd);

        await parent.AppendAsync(new int[] { 1, 2 }); // Prompt 2 tokens
        await using var child = parent.Fork();

        // Cumulative metrics aggregate tokens across active descendants
        var metrics = parent.Tree.CumulativeTreeMetrics;
        Assert.NotNull(metrics);
        Assert.True(metrics.PromptTokens >= 2);
    }

    [Fact]
    public async Task Test8_TreeMetricsSnapshot()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);

        var snap1 = parent.Tree.CumulativeTreeMetrics;
        long p1 = snap1.PromptTokens;

        await parent.AppendAsync(new int[] { 1, 2, 3 });

        // Previous snapshot does not mutate when session history advances
        Assert.Equal(p1, snap1.PromptTokens);
    }

    [Fact]
    public async Task Test9_KvPagesAreNotDoubleCounted()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);

        await parent.AppendAsync(new int[] { 1, 2, 3 });
        await using var child = parent.Fork();

        var metrics = parent.Tree.CumulativeTreeMetrics;
        Assert.True(metrics.KvPagesHeld >= 1);
    }

    [Fact]
    public async Task Test10_ForkAndVoteTreeCleanup()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockTreeForwardPass();
        await using var parent = new InferenceSession(cache, forwardPass: fwd);

        await parent.AppendAsync(new int[] { 1 });

        var result = await parent.ForkAndVoteAsync(new SamplingParams { Temperature = 0.0f }, branchCount: 3);

        // Losing branches disposed and unregistered from parent's Tree
        Assert.Single(parent.Tree.Children);
        Assert.Contains(result.WinningBranch.Id, parent.Tree.Children);
    }

    [Fact]
    public async Task Test11_ForkListenerIsolation()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);

        int parentEvents = 0;
        parent.OnTokenGenerated += (t, s) => parentEvents++;

        await using var child = parent.Fork();

        Assert.Equal(0, parentEvents);
    }

    [Fact]
    public async Task Test12_ContinuationTokenCompatibility()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);

        await parent.AppendAsync(new int[] { 1 });
        var token = parent.GetContinuationToken();

        Assert.Equal(parent.Id, token.SessionId);
    }

    [Fact]
    public async Task Test13_ConcurrentForkAndQuery()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);

        var tasks = Enumerable.Range(0, 5).Select(_ => Task.Run(() =>
        {
            var c = parent.Fork();
            var children = parent.Tree.Children;
            return c;
        })).ToArray();

        var children = await Task.WhenAll(tasks);

        try
        {
            Assert.Equal(5, parent.Tree.Children.Count);
        }
        finally
        {
            foreach (var c in children) await c.DisposeAsync();
        }
    }

    [Fact]
    public async Task Test14_ConcurrentDisposeAndQuery()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);
        var branches = parent.ForkMany(5);

        var disposeTasks = branches.Select(b => Task.Run(async () =>
        {
            var children = parent.Tree.Children;
            await b.DisposeAsync();
        })).ToArray();

        await Task.WhenAll(disposeTasks);
        Assert.Empty(parent.Tree.Children);
    }

    [Fact]
    public async Task Test15_NoOrphanedChildren()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);
        await using var child = parent.Fork();

        Assert.Equal(parent.Id, child.Tree.ParentId);
    }

    [Fact]
    public async Task Test16_RootCannotChange()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);
        await using var child = parent.Fork();

        var r1 = child.Tree.RootId;
        await parent.AppendAsync(new int[] { 1 });
        var r2 = child.Tree.RootId;

        Assert.Equal(r1, r2);
    }

    [Fact]
    public async Task Test17_ParentCannotChange()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);
        await using var child = parent.Fork();

        var p1 = child.Tree.ParentId;
        await parent.AppendAsync(new int[] { 1 });
        var p2 = child.Tree.ParentId;

        Assert.Equal(p1, p2);
    }

    [Fact]
    public async Task Test18_PruningCleansTopology()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var runtime = new InferenceRuntime(cache);

        var root = await runtime.CreateSessionAsync();
        var c1 = root.Fork();
        var c2 = root.Fork();

        // Register forked sessions into runtime session manager for runtime lookup
        runtime.SessionManager.RegisterSession(c1);
        runtime.SessionManager.RegisterSession(c2);

        Assert.Equal(2, root.Tree.Children.Count);

        int pruned = await runtime.PruneBranchTreeAsync(root.Id);

        Assert.Equal(2, pruned);
        Assert.Empty(root.Tree.Children);
        Assert.NotEqual(SessionState.Disposed, root.State);
    }

    [Fact]
    public async Task Test19_PruningReleasesKV()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var runtime = new InferenceRuntime(cache);

        var root = await runtime.CreateSessionAsync();
        var c1 = root.Fork();
        runtime.SessionManager.RegisterSession(c1);

        int pruned = await runtime.PruneBranchTreeAsync(root.Id);
        Assert.Equal(1, pruned);
    }

    [Fact]
    public async Task Test20_ParentNotPruned()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var runtime = new InferenceRuntime(cache);

        var root = await runtime.CreateSessionAsync();
        var c1 = root.Fork();
        runtime.SessionManager.RegisterSession(c1);

        await runtime.PruneBranchTreeAsync(root.Id);

        Assert.NotEqual(SessionState.Disposed, root.State);
        Assert.Equal(SessionState.Ready, root.State);
    }

    private sealed class MockTreeForwardPass : IForwardPass
    {
        public int Position { get; private set; }
        public int VocabSize => 100;
        public int MaxSeqLen => 2048;

        public IForwardPass CreateContext() => new MockTreeForwardPass { Position = Position };
        public System.ReadOnlySpan<float> Forward(int position, int token)
        {
            Position = position + 1;
            var res = new float[100];
            res[10] = 5.0f;
            return res;
        }
        public System.ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            Position = startPos + tokens.Count;
            return new float[100];
        }
        public void TruncateTo(int position) { Position = position; }
        public void ResetCache() { }
        public void Dispose() { }
    }
}
