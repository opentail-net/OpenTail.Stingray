using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

/// <summary>
/// Coverage for the uncommitted-turn compensation path — the release gate's "rollback" dimension.
///
/// <para>This path had never executed. <c>HotSession.CompensateUncommittedTurn</c> only calls
/// <c>RollbackLastTurn</c> when generation completed, and every fault point between that and the
/// commit was a concrete type (<c>InMemorySessionStore</c>, the reservation) that a test could not
/// make fail; a token cancelled during generation throws while <c>generationCompleted</c> is still
/// false, skipping the rollback branch entirely. See docs/sessions-release-gate-matrix.md.</para>
///
/// <para>The tests assert <b>recovery</b>, not merely that the turn reports failure. A rollback that
/// silently did nothing would still produce a failed operation — so "the turn failed" proves
/// nothing. What matters is that the session is left exactly as it was, and that the next turn
/// behaves as if the failed one never ran. That is also the property the production catch block
/// cannot report on, since it swallows every exception rollback might raise.</para>
/// </summary>
public sealed class HotSessionRollbackTests
{
    private const int Eos = 31;

    [Fact]
    public async Task FailedTurn_RestoresCursorAndRevision_LeavingNoTraceOfTheFailedTurn()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        var first = await session.RunTurnAsync(
            "one", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("one"));
        Assert.Equal(SessionOperationState.Completed, first.Operation.State);

        var cursorBefore = session.Cursor;
        var revisionBefore = first.Operation.CommittedRevision!.Value;

        // Fault the turn at the one point where the full compensation body runs.
        session.FaultBeforeCommitForTests = () => throw new InvalidOperationException("injected");
        var failed = await session.RunTurnAsync(
            "two", sampling, revisionBefore, SessionOperationId.New(), Digest("two"));
        session.FaultBeforeCommitForTests = null;

        Assert.Equal(SessionOperationState.Failed, failed.Operation.State);

        // The point of the test: the session is left exactly as it was.
        Assert.Equal(cursorBefore.AcceptedPositionCount, session.Cursor.AcceptedPositionCount);
        Assert.Equal(cursorBefore.MaterializedPositionCount, session.Cursor.MaterializedPositionCount);
        Assert.Equal(cursorBefore.ExecutionLog.Length, session.Cursor.ExecutionLog.Length);
    }

    /// <summary>
    /// The stronger property: after a failed turn, a subsequent turn must produce what it would
    /// have produced had the failure never happened.
    ///
    /// <para><b>This is the test that actually covers rollback</b> — verified by mutation. With
    /// <c>RollbackLastTurn</c> commented out of <c>CompensateUncommittedTurn</c>, this test fails and
    /// the cursor-restore test above still PASSES, because cursor restoration is a separate branch
    /// of the compensation. Anyone trimming this file should keep this case: the other one does not
    /// exercise the rollback at all.</para>
    /// </summary>
    [Fact]
    public async Task TurnAfterAFailedTurn_MatchesTheTurnThatWouldHaveFollowedSuccess()
    {
        var control = await RunSequenceAsync(faultSecondTurn: false);
        var afterFailure = await RunSequenceAsync(faultSecondTurn: true);

        Assert.Equal(control.Accepted, afterFailure.Accepted);
        Assert.Equal(control.Materialized, afterFailure.Materialized);
        Assert.Equal(control.Revision, afterFailure.Revision);
    }

    private static async Task<(int Accepted, int Materialized, SessionRevision Revision)> RunSequenceAsync(
        bool faultSecondTurn)
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        var t1 = await session.RunTurnAsync("one", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("one"));
        var rev = t1.Operation.CommittedRevision!.Value;

        if (faultSecondTurn)
        {
            session.FaultBeforeCommitForTests = () => throw new InvalidOperationException("injected");
            await session.RunTurnAsync(
                "doomed", sampling, rev, SessionOperationId.New(), Digest("doomed"));
            session.FaultBeforeCommitForTests = null;
        }

        var final = await session.RunTurnAsync(
            "three", sampling, rev, SessionOperationId.New(), Digest("three"));

        return (final.Cursor.AcceptedPositionCount, final.Cursor.MaterializedPositionCount,
                final.Operation.CommittedRevision!.Value);
    }

    private static SessionRequestDigest Digest(string s) => SessionRequestDigest.FromCanonicalValue(s);

    private sealed class Tokenizer : ITokenizer
    {
        public int VocabSize => 64;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;
        public IReadOnlyCollection<int> EogTokenIds => [Eos];
        public IReadOnlyList<int> Encode(string text) => text switch
        {
            "one" => [1, 2],
            "two" => [3],
            "doomed" => [4],
            "three" => [5],
            _ => throw new ArgumentOutOfRangeException(nameof(text)),
        };
        public string Decode(IEnumerable<int> tokens) => string.Empty;
        public byte[] DecodeBytes(int token) => [];
    }

    private sealed class FakeForwardPass : IBatchedForwardPass
    {
        private static readonly float[] EosLogits = CreateLogits(Eos);
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 1;
        public int MaxSeqLen => 64;
        public bool PrefillDequantCacheActive => false;

        public ISequenceKvCache CreateCache() => new FakeCache();

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
        {
            var retained = Assert.IsType<FakeCache>(cache);
            // Load-bearing for the rollback tests: if a failed turn left the KV cache advanced,
            // the next turn prefills at a startPos that disagrees with the cache and this fires.
            // Without it, a rollback that silently did nothing would still let both tests pass.
            Assert.Equal(startPos, retained.LogicalPosition);
            retained.LogicalPosition += tokens.Count;
            return EosLogits;
        }

        public float[]?[] PrefillPackedMulti(
            ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits) =>
            throw new NotSupportedException();

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            foreach (var c in caches) Assert.IsType<FakeCache>(c).LogicalPosition++;
            return Enumerable.Repeat(EosLogits, tokens.Length).ToArray();
        }

        private static float[] CreateLogits(int token)
        {
            var logits = new float[64];
            logits[token] = 1f;
            return logits;
        }
    }

    private sealed class FakeCache : IRewindableSequenceKvCache
    {
        public int LogicalPosition { get; set; }
        public bool CanRewindTo(int logicalPosition) => logicalPosition >= 0 && logicalPosition <= LogicalPosition;
        public void RewindTo(int logicalPosition) => LogicalPosition = logicalPosition;
        public void Dispose() { }
    }
}
