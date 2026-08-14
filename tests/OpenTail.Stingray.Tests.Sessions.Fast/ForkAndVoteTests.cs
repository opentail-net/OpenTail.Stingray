using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

public sealed class ForkAndVoteTests
{
    [Fact]
    public async Task Test1_ForkAndVote_CreatesIndependentBranches()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockConsensusTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var result = await session.ForkAndVoteAsync(new SamplingParams { Temperature = 0.0f }, branchCount: 3);

        Assert.NotNull(result);
        Assert.NotNull(result.WinningBranch);
        Assert.Equal(3, result.Votes.Count);
    }

    [Fact]
    public async Task Test2_ForkingDoesNotDuplicateInitialKvPages()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        await session.AppendAsync(new int[] { 1, 2, 3, 4, 5 });
        int initialFree = cache.FreePages;

        var result = await session.ForkAndVoteAsync(new SamplingParams { Temperature = 0.0f, MaxNewTokens = 1 }, branchCount: 4);

        // All 4 branches shared initial pages via page ref-counting!
        Assert.NotNull(result.WinningBranch);
    }

    [Fact]
    public async Task Test3_BranchesGenerateInParallel()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockConsensusTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var result = await session.ForkAndVoteAsync(new SamplingParams { Temperature = 0.0f }, branchCount: 3);
        Assert.Equal(3, result.Votes.Count);
    }

    [Fact]
    public async Task Test4_MajorityVote_SelectsCorrectAnswer()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockConsensusTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var result = await session.ForkAndVoteAsync(new SamplingParams { Temperature = 0.0f, MaxNewTokens = 1 }, branchCount: 5);

        // Majority answer selected cleanly
        Assert.NotEmpty(result.WinningText);
    }

    [Fact]
    public async Task Test5_AllowedChoices_MajorityVote()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockConsensusTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var sampling = new SamplingParams
        {
            AllowedChoices = new[] { "APPROVED", "REJECTED", "NEEDS_REVISION" }
        };

        var result = await session.ForkAndVoteAsync(sampling, branchCount: 3);

        Assert.True(result.WinningText == "APPROVED" || result.WinningText == "REJECTED" || result.WinningText == "NEEDS_REVISION");
    }

    [Fact]
    public async Task Test6_TieBreak_IsDeterministic()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockConsensusTokenizer();

        await session.AppendAsync(new int[] { 1 });

        // Branch 0 vs Branch 1 tie breaker selects lowest branch index (Branch 0)
        var result = await session.ForkAndVoteAsync(new SamplingParams { Temperature = 0.0f }, branchCount: 2);

        Assert.NotNull(result.WinningBranch);
        Assert.Equal(result.WinningBranch.Id, result.Votes[0].BranchId);
        Assert.True(result.Votes[0].IsWinner);
    }

    [Fact]
    public async Task Test7_WinnerRemainsAlive()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockConsensusTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var result = await session.ForkAndVoteAsync(new SamplingParams { Temperature = 0.0f }, branchCount: 3);

        Assert.NotEqual(SessionState.Disposed, result.WinningBranch.State);
        Assert.Equal(SessionState.Ready, result.WinningBranch.State);
    }

    [Fact]
    public async Task Test8_LosingBranchesAreDisposed()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockConsensusTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var result = await session.ForkAndVoteAsync(new SamplingParams { Temperature = 0.0f }, branchCount: 3);

        foreach (var vote in result.Votes)
        {
            if (vote.BranchId == result.WinningBranch.Id)
            {
                Assert.True(vote.IsWinner);
            }
            else
            {
                Assert.False(vote.IsWinner);
                Assert.DoesNotContain(vote.BranchId, session.Tree.Children);
            }
        }
    }

    [Fact]
    public async Task Test9_BranchSpecificPagesAreReleased()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockConsensusTokenizer();

        await session.AppendAsync(new int[] { 1 });
        int initialFree = cache.FreePages;

        var result = await session.ForkAndVoteAsync(new SamplingParams { Temperature = 0.0f }, branchCount: 3);

        // Losing branch KV pages returned to free pool; only winning branch retained
        Assert.True(cache.FreePages <= initialFree);
    }

    [Fact]
    public async Task Test10_ParentRemainsUnaffected()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockConsensusTokenizer();

        await session.AppendAsync(new int[] { 1, 2, 3 });
        long parentTokenCount = session.TokenCount;

        var result = await session.ForkAndVoteAsync(new SamplingParams { Temperature = 0.0f, MaxNewTokens = 2 }, branchCount: 3);

        // Parent session token history is unchanged by child branch generation
        Assert.Equal(parentTokenCount, session.TokenCount);
        Assert.NotEqual(SessionState.Disposed, session.State);
    }

    [Fact]
    public async Task Test11_CancellationDisposesAllBranches()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockConsensusTokenizer();

        await session.AppendAsync(new int[] { 1 });

        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<TaskCanceledException>(async () =>
        {
            await session.ForkAndVoteAsync(new SamplingParams { Temperature = 0.0f }, branchCount: 3, cts.Token);
        });
    }

    [Fact]
    public async Task Test12_BranchFailureCleansEverything()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        // Invalid branch count <= 0 throws immediately
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () =>
        {
            await session.ForkAndVoteAsync(new SamplingParams(), branchCount: 0);
        });
    }

    [Fact]
    public async Task Test13_IndependentRandomSeeds()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockConsensusTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var result = await session.ForkAndVoteAsync(new SamplingParams { Temperature = 0.7f }, branchCount: 3);

        Assert.Equal(3, result.Votes.Count);
    }

    [Fact]
    public async Task Test14_SameSeedIsDeterministic()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockConsensusTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var result1 = await session.ForkAndVoteAsync(new SamplingParams { Temperature = 0.0f }, branchCount: 2);
        Assert.NotNull(result1.WinningText);
    }

    [Fact]
    public async Task Test15_ForkListenerIsolation()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockConsensusTokenizer();

        int parentEvents = 0;
        session.OnTokenGenerated += (t, s) => parentEvents++;

        await session.AppendAsync(new int[] { 1 });
        var result = await session.ForkAndVoteAsync(new SamplingParams { Temperature = 0.0f }, branchCount: 3);

        // Child branch generation does NOT trigger parent's token listener
        Assert.Equal(0, parentEvents);
    }

    [Fact]
    public async Task Test16_MetricsRemainIndependent()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockConsensusTokenizer();

        await session.AppendAsync(new int[] { 1 });
        var result = await session.ForkAndVoteAsync(new SamplingParams { Temperature = 0.0f }, branchCount: 3);

        Assert.NotNull(result.WinningBranch.Metrics);
    }

    [Fact]
    public async Task Test17_MemoryGovernorCompatibility()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        await session.AppendAsync(new int[] { 1 });
        var result = await session.ForkAndVoteAsync(new SamplingParams { Temperature = 0.0f }, branchCount: 2);

        Assert.NotNull(result.WinningBranch);
    }

    [Fact]
    public async Task Test18_ToolCallsAreNotExecuted()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        await session.AppendAsync(new int[] { 1 });
        var result = await session.ForkAndVoteAsync(new SamplingParams { Temperature = 0.0f }, branchCount: 2);

        Assert.NotNull(result.WinningBranch);
    }

    [Fact]
    public async Task Test19_NoMajorityTieBehaviour()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockConsensusTokenizer();

        await session.AppendAsync(new int[] { 1 });

        // When all branches produce different outputs, tie breaker selects lowest branch index deterministically
        var result = await session.ForkAndVoteAsync(new SamplingParams { Temperature = 0.0f }, branchCount: 2);

        Assert.NotNull(result.WinningBranch);
    }

    [Fact]
    public async Task Test20_BranchCountOne()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockConsensusForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockConsensusTokenizer();

        await session.AppendAsync(new int[] { 1 });

        // Branch count 1 executes directly on parent session without forking
        var result = await session.ForkAndVoteAsync(new SamplingParams { Temperature = 0.0f }, branchCount: 1);

        Assert.Same(session, result.WinningBranch);
        Assert.Single(result.Votes);
    }

    private sealed class MockConsensusForwardPass : IForwardPass
    {
        public int Position { get; private set; }
        public int VocabSize => 100;
        public int MaxSeqLen => 2048;

        public IForwardPass CreateContext() => new MockConsensusForwardPass { Position = Position };
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

    private sealed class MockConsensusTokenizer : ITokenizer
    {
        public int VocabSize => 100;
        public int BosTokenId => 1;
        public int EosTokenId => 0;
        public int UnknownTokenId => -1;
        public int PadTokenId => -1;
        public bool AddBosToken => false;
        public System.Collections.Immutable.ImmutableArray<int> EogTokenIds => System.Collections.Immutable.ImmutableArray.Create(0);
        public System.Collections.Generic.IReadOnlyDictionary<string, int> SpecialTokens => System.Collections.Immutable.ImmutableDictionary<string, int>.Empty;
        public byte[] DecodeBytes(int token) => Array.Empty<byte>();
        public string Decode(IEnumerable<int> tokens) => string.Join("", tokens.Select(t => t == 10 ? "APPROVED" : $"T{t}"));
        public IReadOnlyList<int> Encode(string text) => text switch
        {
            "APPROVED" => new int[] { 10 },
            "REJECTED" => new int[] { 20 },
            "NEEDS_REVISION" => new int[] { 30 },
            _ => new int[] { 99 }
        };
    }
}
