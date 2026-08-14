namespace OpenTail.Stingray.Tests.Sessions.Fast;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

/// <summary>
/// docs/028 Phase 3: <see cref="HotSessionRuntime.Fork"/> creates N independent branches from a
/// parent session's current retained state, sharing physical KV pages zero-copy via the same
/// mechanism Phase 2's cross-session sharing uses
/// (<see cref="CrossSessionPrefixForkingTests"/>/<see cref="HotSession.TryForkSharedPrefixCache"/>).
/// This file exercises the orchestration against a fake forward pass: validation, page-alignment
/// flooring, atomic all-or-nothing rollback on failure, and branch/parent independence. The
/// real-model proof that forked branches genuinely share physical pages and correctly
/// copy-on-write on divergence lives in the heavy Tests.Sessions project, since a fake cannot
/// represent shared memory at all.
/// </summary>
public sealed class HotSessionForkTests
{
    private const int Eos = 31;
    private const int NonStopToken = 7;

    private static SessionRequestDigest Digest(string val) => SessionRequestDigest.FromCanonicalValue(val);

    private sealed class Tokenizer : ITokenizer
    {
        public int VocabSize => 64;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;
        public IReadOnlyCollection<int> EogTokenIds => [Eos];
        public IReadOnlyList<int> Encode(string text) => [1, 2];
        public string Decode(IEnumerable<int> tokens) => "tok";
        public byte[] DecodeBytes(int token) => [(byte)('a' + (token % 26))];
    }

    private sealed class FakeCache : IRewindableSequenceKvCache
    {
        public int LogicalPosition { get; set; }
        public bool Disposed { get; private set; }
        public bool CanRewindTo(int logicalPosition) => logicalPosition >= 0 && logicalPosition <= LogicalPosition;
        public void RewindTo(int logicalPosition) => LogicalPosition = logicalPosition;
        public void Dispose() => Disposed = true;
    }

    private sealed class FakeForwardPass : IBatchedForwardPass, IPrefixCacheableBatchedForwardPass
    {
        private static readonly float[] NonStopLogits = CreateLogits(NonStopToken);
        private static readonly float[] EosLogits = CreateLogits(Eos);
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 10;
        public int MaxSeqLen => 64;
        public bool PrefillDequantCacheActive => false;

        // Small and non-default, matching CrossSessionPrefixForkingTests, so alignment-flooring
        // behavior is actually exercised rather than coincidentally matching a round number.
        public int PrefixCacheBlockSize => 4;

        /// <summary>Keeps decode from hitting EOS early, so a turn's length is controlled purely
        /// by MaxNewTokens.</summary>
        public bool EmitNonStopOnDecode { get; set; } = true;
        public bool BlockDecode { get; set; }
        public TaskCompletionSource DecodeStarted { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public ManualResetEventSlim ReleaseDecode { get; } = new(false);

        public ISequenceKvCache CreateCache() => new FakeCache();

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
        {
            var retained = Assert.IsType<FakeCache>(cache);
            retained.LogicalPosition = startPos + tokens.Count;
            return NonStopLogits;
        }

        public float[]?[] PrefillPackedMulti(ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits) =>
            throw new NotSupportedException();

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            for (int i = 0; i < caches.Length; i++)
            {
                var cache = Assert.IsType<FakeCache>(caches[i]);
                cache.LogicalPosition++;
            }
            if (BlockDecode)
            {
                DecodeStarted.TrySetResult();
                if (!ReleaseDecode.Wait(TimeSpan.FromSeconds(10)))
                    throw new TimeoutException("The test did not release decode.");
            }
            return Enumerable.Repeat(EmitNonStopOnDecode ? NonStopLogits : EosLogits, tokens.Length).ToArray();
        }

        public ISequenceKvCache CapturePrefix(ISequenceKvCache cache, int prefixLength)
        {
            var source = Assert.IsType<FakeCache>(cache);
            Assert.InRange(prefixLength, 0, source.LogicalPosition);
            Assert.Equal(0, prefixLength % PrefixCacheBlockSize);
            return new FakeCache { LogicalPosition = prefixLength };
        }

        public ISequenceKvCache ForkPrefix(ISequenceKvCache prefix)
        {
            var source = Assert.IsType<FakeCache>(prefix);
            return new FakeCache { LogicalPosition = source.LogicalPosition };
        }

        private static float[] CreateLogits(int winner)
        {
            var logits = new float[64];
            logits[winner] = 10f;
            return logits;
        }
    }

    /// <summary>Runs two turns of 2 generated tokens each, leaving the session idle with 8
    /// materialized positions ([1,2],[7,7],[1,2],[7,7]) -- exactly two blocks under
    /// PrefixCacheBlockSize=4, so forking the whole thing loses nothing to alignment flooring.
    /// Returns the second turn's committed revision, for tests that need to submit a third.</summary>
    private static async Task<SessionRevision> RunTwoFullTurnsAsync(HotSession session)
    {
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 2 };
        var first = await session.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("t1"));
        var second = await session.RunTurnAsync("world", sampling, first.Operation.CommittedRevision!.Value, SessionOperationId.New(), Digest("t2"));
        return second.Operation.CommittedRevision!.Value;
    }

    private static (FakeForwardPass Fwd, ContinuousBatchingEngine Engine, HotSessionRuntime Runtime) NewRuntime()
    {
        var fwd = new FakeForwardPass();
        var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 4);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        return (fwd, engine, runtime);
    }

    [Fact]
    public async Task Fork_CreatesIndependentBranchesSharingTheParentsPrefix()
    {
        var (_, engine, runtime) = NewRuntime();
        using var _e = engine;

        using var parent = runtime.Create();
        await RunTwoFullTurnsAsync(parent);
        Assert.Equal(8, parent.Cursor.MaterializedPositionCount);
        var parentTokens = parent.Cursor.ExecutionLog.SelectMany(s => ((TokenSegment)s).TokenIds).ToArray();

        var branches = runtime.Fork(parent, 3);
        try
        {
            Assert.Equal(3, branches.Count);
            Assert.Equal(3, branches.Select(b => b.SessionId).Distinct().Count());
            foreach (var branch in branches)
            {
                Assert.Equal(8, branch.Cursor.MaterializedPositionCount);
                var branchTokens = branch.Cursor.ExecutionLog.SelectMany(s => ((TokenSegment)s).TokenIds);
                Assert.Equal(parentTokens, branchTokens);
            }
        }
        finally
        {
            foreach (var b in branches) b.Dispose();
        }
    }

    [Fact]
    public async Task Fork_IsZeroCopy_EachBranchContributesToTheSharedPrefixCounter()
    {
        var (_, engine, runtime) = NewRuntime();
        using var _e = engine;

        using var parent = runtime.Create();
        await RunTwoFullTurnsAsync(parent);

        long before = engine.CrossSessionPrefixTokensShared;
        var branches = runtime.Fork(parent, 3);
        try
        {
            // Each of the 3 forks independently shares the same 8-position prefix.
            Assert.Equal(before + 8 * 3, engine.CrossSessionPrefixTokensShared);
        }
        finally
        {
            foreach (var b in branches) b.Dispose();
        }
    }

    [Fact]
    public async Task Fork_FloorsToLastFullPage_WhenParentPositionIsNotAligned()
    {
        var (_, engine, runtime) = NewRuntime();
        using var _e = engine;

        using var parent = runtime.Create();
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };
        var first = await parent.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("t1"));
        await parent.RunTurnAsync("world", sampling, first.Operation.CommittedRevision!.Value, SessionOperationId.New(), Digest("t2"));
        // 2 turns x (2 prompt + 1 generated) = 6 positions -- not a multiple of the 4-token block.
        Assert.Equal(6, parent.Cursor.MaterializedPositionCount);
        var parentTokens = parent.Cursor.ExecutionLog.SelectMany(s => ((TokenSegment)s).TokenIds).ToArray();

        var branches = runtime.Fork(parent, 1);
        try
        {
            var branch = Assert.Single(branches);
            // floor(6 / 4) * 4 = 4 -- not 6. The branch must never claim more than what's
            // actually page-aligned, even though all 6 parent tokens are known and idle.
            Assert.Equal(4, branch.Cursor.MaterializedPositionCount);
            Assert.Equal(parentTokens[..4], branch.Cursor.ExecutionLog.SelectMany(s => ((TokenSegment)s).TokenIds));
        }
        finally
        {
            foreach (var b in branches) b.Dispose();
        }
    }

    [Fact]
    public void Fork_RejectsNonPositiveCount()
    {
        var (_, engine, runtime) = NewRuntime();
        using var _e = engine;
        using var parent = runtime.Create();

        Assert.Throws<ArgumentOutOfRangeException>(() => runtime.Fork(parent, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => runtime.Fork(parent, -1));
    }

    [Fact]
    public async Task Fork_CountOfOne_StillWorks()
    {
        var (_, engine, runtime) = NewRuntime();
        using var _e = engine;

        using var parent = runtime.Create();
        await RunTwoFullTurnsAsync(parent);

        var branches = runtime.Fork(parent, 1);
        try
        {
            var branch = Assert.Single(branches);
            Assert.Equal(8, branch.Cursor.MaterializedPositionCount);
        }
        finally
        {
            foreach (var b in branches) b.Dispose();
        }
    }

    [Fact]
    public void Fork_FromSessionWithNoMaterializedContent_ProducesColdBranches()
    {
        var (_, engine, runtime) = NewRuntime();
        using var _e = engine;

        using var parent = runtime.Create();
        var branches = runtime.Fork(parent, 2);
        try
        {
            Assert.Equal(2, branches.Count);
            foreach (var branch in branches)
            {
                Assert.Equal(0, branch.Cursor.MaterializedPositionCount);
                Assert.Empty(branch.Cursor.ExecutionLog);
            }
        }
        finally
        {
            foreach (var b in branches) b.Dispose();
        }
    }

    [Fact]
    public async Task Fork_ParentNotIdle_ThrowsAndLeavesNoResidualReservation()
    {
        var (fwd, engine, runtime) = NewRuntime();
        using var _e = engine;

        using var parent = runtime.Create();
        var revision = await RunTwoFullTurnsAsync(parent);
        long residentAfterTurns = runtime.ResidentBytes;

        fwd.BlockDecode = true;
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 2 };
        var blockedTurn = parent.RunTurnAsync("blocked", sampling, revision,
            SessionOperationId.New(), Digest("blocked"));
        await fwd.DecodeStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
        try
        {
            Assert.Throws<InvalidOperationException>(() => runtime.Fork(parent, 3));
            // No branch's reservation survives a rolled-back fork -- residency is exactly what it
            // was before the failed call, not merely "some subset less than 3 branches' worth".
            Assert.Equal(residentAfterTurns, runtime.ResidentBytes);
        }
        finally
        {
            fwd.BlockDecode = false;
            fwd.ReleaseDecode.Set();
            await blockedTurn;
        }
    }

    [Fact]
    public async Task Fork_ParentRemainsUnaffectedByBranchGeneration()
    {
        var (_, engine, runtime) = NewRuntime();
        using var _e = engine;

        using var parent = runtime.Create();
        await RunTwoFullTurnsAsync(parent);
        var parentTokensBefore = parent.Cursor.ExecutionLog.SelectMany(s => ((TokenSegment)s).TokenIds).ToArray();
        int parentPositionBefore = parent.Cursor.MaterializedPositionCount;

        var branches = runtime.Fork(parent, 2);
        try
        {
            var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };
            var result = await branches[0].RunTurnAsync("continue", sampling, SessionRevision.Initial,
                SessionOperationId.New(), Digest("branch-continue"));
            Assert.Equal(SessionOperationState.Completed, result.Operation.State);

            Assert.Equal(parentPositionBefore, parent.Cursor.MaterializedPositionCount);
            Assert.Equal(parentTokensBefore, parent.Cursor.ExecutionLog.SelectMany(s => ((TokenSegment)s).TokenIds));
        }
        finally
        {
            foreach (var b in branches) b.Dispose();
        }
    }

    [Fact]
    public async Task Fork_BranchesAreIndependent_GeneratingOnOneDoesNotAffectSiblings()
    {
        var (_, engine, runtime) = NewRuntime();
        using var _e = engine;

        using var parent = runtime.Create();
        await RunTwoFullTurnsAsync(parent);

        var branches = runtime.Fork(parent, 2);
        try
        {
            int siblingPositionBefore = branches[1].Cursor.MaterializedPositionCount;
            var siblingTokensBefore = branches[1].Cursor.ExecutionLog.SelectMany(s => ((TokenSegment)s).TokenIds).ToArray();

            var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };
            var result = await branches[0].RunTurnAsync("continue", sampling, SessionRevision.Initial,
                SessionOperationId.New(), Digest("branch0-continue"));
            Assert.Equal(SessionOperationState.Completed, result.Operation.State);

            Assert.True(branches[0].Cursor.MaterializedPositionCount > siblingPositionBefore);
            Assert.Equal(siblingPositionBefore, branches[1].Cursor.MaterializedPositionCount);
            Assert.Equal(siblingTokensBefore, branches[1].Cursor.ExecutionLog.SelectMany(s => ((TokenSegment)s).TokenIds));
        }
        finally
        {
            foreach (var b in branches) b.Dispose();
        }
    }

    [Fact]
    public async Task Fork_CanBeNested_GrandchildSharesTheOriginalPrefix()
    {
        var (_, engine, runtime) = NewRuntime();
        using var _e = engine;

        using var root = runtime.Create();
        await RunTwoFullTurnsAsync(root);
        var rootTokens = root.Cursor.ExecutionLog.SelectMany(s => ((TokenSegment)s).TokenIds).ToArray();

        var children = runtime.Fork(root, 2);
        try
        {
            var grandchildren = runtime.Fork(children[0], 2);
            try
            {
                foreach (var grandchild in grandchildren)
                {
                    Assert.Equal(8, grandchild.Cursor.MaterializedPositionCount);
                    Assert.Equal(rootTokens, grandchild.Cursor.ExecutionLog.SelectMany(s => ((TokenSegment)s).TokenIds));
                }
            }
            finally
            {
                foreach (var g in grandchildren) g.Dispose();
            }
        }
        finally
        {
            foreach (var c in children) c.Dispose();
        }
    }
}
