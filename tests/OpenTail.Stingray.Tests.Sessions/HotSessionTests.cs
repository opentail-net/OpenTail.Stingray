using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;

namespace OpenTail.Stingray.Tests.Sessions;

public sealed class HotSessionTests
{
    private const int Eos = 31;
    /// <summary>The token the fake samples when it is not emitting EOS.</summary>
    private const int NonStopToken = 7;

    [Fact]
    public async Task HotSession_CommitsCursorAndRevision_AndReusesItsRetainedCache()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };
        var firstId = SessionOperationId.New();

        var first = await session.RunTurnAsync("one", sampling, SessionRevision.Initial, firstId, Digest("one"));

        Assert.Equal(SessionOperationState.Completed, first.Operation.State);
        Assert.Equal(new SessionRevision(1), first.Operation.CommittedRevision);
        Assert.Equal(2, first.Cursor.AcceptedPositionCount);
        Assert.Equal(2, first.Cursor.MaterializedPositionCount);
        Assert.Single(fwd.Created);
        Assert.Equal([(0, 2)], fwd.Prefills);

        var second = await session.RunTurnAsync("two", sampling, new SessionRevision(1), SessionOperationId.New(), Digest("two"));

        Assert.Equal(new SessionRevision(2), second.Operation.CommittedRevision);
        Assert.Equal(3, second.Cursor.AcceptedPositionCount);
        Assert.Equal([(0, 2), (2, 1)], fwd.Prefills);
        Assert.Equal(2, second.Cursor.ExecutionLog.Length);
        Assert.IsType<TokenSegment>(second.Cursor.ExecutionLog[0]);
        Assert.IsType<TokenSegment>(second.Cursor.ExecutionLog[1]);
    }

    [Fact]
    public async Task HotSession_CompletedOperationReplay_DoesNotRegenerateOrAdvanceRevision()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };
        var operationId = SessionOperationId.New();

        var first = await session.RunTurnAsync("one", sampling, SessionRevision.Initial, operationId, Digest("one"));
        var replay = await session.RunTurnAsync("one", sampling, SessionRevision.Initial, operationId, Digest("one"));

        Assert.False(first.IsIdempotentReplay);
        Assert.True(replay.IsIdempotentReplay);
        Assert.Equal(new SessionRevision(1), replay.Operation.CommittedRevision);
        Assert.Equal(first.Chunks, replay.Chunks);
        Assert.Single(fwd.Created);
        Assert.Single(fwd.Prefills);
    }

    [Fact]
    public async Task HotSession_RejectsTurnsWhoseResultWouldExceedCaptureLimit()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer(),
            new HotSessionRuntimeOptions(maxCapturedOutputChunks: 4));
        using var session = runtime.Create();

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => session.RunTurnAsync(
            "one", new SamplingParams { Temperature = 0f, MaxNewTokens = 1 },
            SessionRevision.Initial, SessionOperationId.New(), Digest("over-limit")));
        Assert.Empty(fwd.Created);
    }

    [Fact]
    public async Task HotSession_CancellationRollsBackCacheAndDoesNotCommitRevision()
    {
        var fwd = new FakeForwardPass { EmitNonStopOnPrefill = true, BlockDecode = true };
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();
        using var cts = new CancellationTokenSource();
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 4 };

        var cancelledTurn = session.RunTurnAsync("one", sampling, SessionRevision.Initial,
            SessionOperationId.New(), Digest("cancel"), cts.Token);
        await fwd.DecodeStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
        cts.Cancel();
        fwd.ReleaseDecode.Set();
        var cancelled = await cancelledTurn;

        Assert.Equal(SessionOperationState.Cancelled, cancelled.Operation.State);
        Assert.Null(cancelled.Operation.CommittedRevision);
        Assert.Equal(0, cancelled.Cursor.MaterializedPositionCount);

        fwd.EmitNonStopOnPrefill = false;
        fwd.BlockDecode = false;
        var retry = await session.RunTurnAsync("one", new SamplingParams { Temperature = 0f, MaxNewTokens = 1 },
            SessionRevision.Initial, SessionOperationId.New(), Digest("retry"));
        Assert.Equal(new SessionRevision(1), retry.Operation.CommittedRevision);
        Assert.Single(fwd.Created);
        Assert.Equal((0, 2), fwd.Prefills[^1]);
    }

    [Fact]
    public async Task HotSession_CancellationDuringPrefillLeavesTheCursorUnchanged()
    {
        var fwd = new FakeForwardPass { BlockPrefill = true };
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();
        using var cts = new CancellationTokenSource();

        var turn = session.RunTurnAsync("one", new SamplingParams { Temperature = 0f, MaxNewTokens = 1 },
            SessionRevision.Initial, SessionOperationId.New(), Digest("prefill-cancel"), cts.Token);
        await fwd.PrefillStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
        cts.Cancel();
        fwd.ReleasePrefill.Set();

        var cancelled = await turn;

        Assert.Equal(SessionOperationState.Cancelled, cancelled.Operation.State);
        Assert.Null(cancelled.Operation.CommittedRevision);
        Assert.Equal(0, cancelled.Cursor.AcceptedPositionCount);
        Assert.Equal(0, cancelled.Cursor.MaterializedPositionCount);
        Assert.Equal(0, runtime.ResidentBytes);
    }

    [Fact]
    public async Task HotSession_ReservesProjectedBytesAndRejectsOverBudgetAdmission()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer(),
            new HotSessionRuntimeOptions(maxResidentBytes: 3, maxSessionBytes: 3));
        using var first = runtime.Create();
        using var second = runtime.Create();
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        await first.RunTurnAsync("one", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("first"));
        Assert.Equal(2, runtime.ResidentBytes);

        var ex = await Assert.ThrowsAsync<SessionResourceBudgetExceededException>(() => second.RunTurnAsync(
            "one", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("second")));

        Assert.Equal(3, ex.RequestedBytes);
        Assert.Equal(1, ex.AvailableBytes);
        Assert.Equal(2, runtime.ResidentBytes);
    }

    [Fact]
    public async Task HotSessionRuntime_OpensAndDeletesTheActiveSession()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        var session = runtime.Create();

        Assert.Same(session, runtime.Open(session.SessionId));
        await session.RunTurnAsync("one", new SamplingParams { Temperature = 0f, MaxNewTokens = 1 },
            SessionRevision.Initial, SessionOperationId.New(), Digest("delete"));
        Assert.Equal(2, runtime.ResidentBytes);

        Assert.True(runtime.Delete(session.SessionId));
        Assert.Equal(0, runtime.ResidentBytes);
        Assert.False(runtime.Delete(session.SessionId));
        Assert.Throws<SessionNotFoundException>(() => runtime.Open(session.SessionId));
        await Assert.ThrowsAsync<ObjectDisposedException>(() => session.RunTurnAsync(
            "two", new SamplingParams { Temperature = 0f, MaxNewTokens = 1 },
            new SessionRevision(1), SessionOperationId.New(), Digest("after-delete")));
    }

    private static SessionRequestDigest Digest(string value) => SessionRequestDigest.FromCanonicalValue(value);

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
            _ => throw new ArgumentOutOfRangeException(nameof(text)),
        };
        public string Decode(IEnumerable<int> tokens) => string.Empty;
        public byte[] DecodeBytes(int token) => [];
    }

    /// <summary>
    /// Milestone 1 required invariant: EOS completion. The turn must commit normally and stop of
    /// its own accord — strictly BEFORE the token budget — and the cursor must account for exactly
    /// the positions that were materialised.
    /// </summary>
    [Fact]
    public async Task HotSession_EosCompletion_StopsEarlyAndCommits()
    {
        var fwd = new FakeForwardPass { EmitNonStopOnPrefill = true };   // EOS arrives from decode
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();

        const int budget = 8;
        var result = await session.RunTurnAsync("one",
            new SamplingParams { Temperature = 0f, MaxNewTokens = budget },
            SessionRevision.Initial, SessionOperationId.New(), Digest("eos"));

        Assert.True(result.Operation.State == SessionOperationState.Completed,
            $"state={result.Operation.State} reason={result.Operation.FailureReason}");
        Assert.Equal(new SessionRevision(1), result.Operation.CommittedRevision);

        int generated = GeneratedTokenCount(result.Cursor);
        Assert.True(generated < budget,
            $"EOS completion must stop before the {budget}-token budget; generated {generated}.");
        // The cursor is the authority on what was materialised: prompt + generated, nothing else.
        Assert.Equal(result.Cursor.AcceptedPositionCount, result.Cursor.MaterializedPositionCount);
    }

    /// <summary>
    /// Milestone 1 required invariant: maximum-token completion. With EOS never sampled, generation
    /// can only end by exhausting the budget, and it must end there EXACTLY — not one short, not
    /// one over. This is the counterpart to the EOS test: together they pin that the two
    /// termination reasons are distinguishable rather than coincidentally similar.
    /// </summary>
    [Fact]
    public async Task HotSession_MaximumTokenCompletion_StopsExactlyAtTheBudget()
    {
        var fwd = new FakeForwardPass { EmitNonStopOnPrefill = true, EmitNonStopOnDecode = true };
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();

        const int budget = 5;
        var result = await session.RunTurnAsync("one",
            new SamplingParams { Temperature = 0f, MaxNewTokens = budget },
            SessionRevision.Initial, SessionOperationId.New(), Digest("maxtok"));

        Assert.Equal(SessionOperationState.Completed, result.Operation.State);
        Assert.Equal(budget, GeneratedTokenCount(result.Cursor));
        Assert.Equal(result.Cursor.AcceptedPositionCount, result.Cursor.MaterializedPositionCount);
    }

    /// <summary>
    /// Milestone 1 required invariant: stale expected revision rejection. `expectedRevision` is a
    /// parameter on every turn and had **no** test coverage — it appears zero times in the store
    /// tests — despite being the entire optimistic-concurrency mechanism.
    ///
    /// <para>Rejection alone is not the interesting half. The turn must leave the session exactly
    /// as it found it: same revision, same cursor. A rejection that had already mutated state would
    /// be worse than none, because the caller's retry would then build on a half-applied turn.</para>
    /// </summary>
    [Fact]
    public async Task HotSession_StaleExpectedRevision_IsRejectedAndLeavesStateUntouched()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        var first = await session.RunTurnAsync("one", sampling,
            SessionRevision.Initial, SessionOperationId.New(), Digest("one"));
        Assert.Equal(new SessionRevision(1), first.Operation.CommittedRevision);
        var cursorBefore = session.Cursor;
        int prefillsBefore = fwd.Prefills.Count;

        // Submit against the revision that was current BEFORE the first turn committed.
        var ex = await Assert.ThrowsAsync<SessionRevisionConflictException>(() => session.RunTurnAsync(
            "two", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("stale")));

        Assert.Equal(SessionRevision.Initial, ex.ExpectedRevision);
        Assert.Equal(new SessionRevision(1), ex.ActualRevision);
        // Nothing ran and nothing moved.
        Assert.Equal(prefillsBefore, fwd.Prefills.Count);
        Assert.Equal(cursorBefore.AcceptedPositionCount, session.Cursor.AcceptedPositionCount);
        Assert.Equal(cursorBefore.ExecutionLog.Length, session.Cursor.ExecutionLog.Length);
    }

    /// <summary>
    /// Milestone 1 required invariant: allocation failure leaves the prior revision intact.
    /// <see cref="HotSession_ReservesProjectedBytesAndRejectsOverBudgetAdmission"/> already asserts
    /// that an over-budget turn throws, but says nothing about what survives it — which is the part
    /// the invariant is actually about. Here the SAME session commits a turn, then is refused a
    /// second, and must still hold exactly the first turn's revision and cursor.
    /// </summary>
    [Fact]
    public async Task HotSession_AllocationFailure_LeavesThePriorRevisionIntact()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer(),
            new HotSessionRuntimeOptions(maxResidentBytes: 3, maxSessionBytes: 3));
        using var session = runtime.Create();
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        var committed = await session.RunTurnAsync("one", sampling,
            SessionRevision.Initial, SessionOperationId.New(), Digest("one"));
        Assert.Equal(new SessionRevision(1), committed.Operation.CommittedRevision);
        var cursorBefore = session.Cursor;
        long residentBefore = runtime.ResidentBytes;

        // The second turn projects past the session's byte ceiling and must be refused at
        // reservation time, before any forward work.
        await Assert.ThrowsAsync<SessionResourceBudgetExceededException>(() => session.RunTurnAsync(
            "two", sampling, new SessionRevision(1), SessionOperationId.New(), Digest("two")));

        Assert.Equal(cursorBefore.AcceptedPositionCount, session.Cursor.AcceptedPositionCount);
        Assert.Equal(cursorBefore.MaterializedPositionCount, session.Cursor.MaterializedPositionCount);
        Assert.Equal(residentBefore, runtime.ResidentBytes);   // the refused reservation released

        // And the committed revision is still exactly 1 — proved without needing another
        // successful turn (there is no budget headroom for one) by submitting a deliberately
        // stale revision and reading what the conflict reports as ACTUAL.
        var conflict = await Assert.ThrowsAsync<SessionRevisionConflictException>(() =>
            session.RunTurnAsync("two", sampling, SessionRevision.Initial,
                SessionOperationId.New(), Digest("probe")));
        Assert.Equal(new SessionRevision(1), conflict.ActualRevision);
    }

    /// <summary>
    /// Milestone 1 required invariant: mismatch at a prior turn's closing marker.
    ///
    /// <para>The existing cursor tests cannot express this shape — they build a synthetic
    /// single-segment cursor, and this needs a real multi-turn log
    /// <c>[prompt₁, generated₁, prompt₂, …]</c> with the divergence landing on the LAST token of
    /// <c>generated₁</c>, i.e. exactly where one turn closes and the next begins. That boundary is
    /// the dangerous one: everything before it matches, so a reconciler that compared only the
    /// leading segment, or that treated a turn boundary as a resync point, would wrongly report
    /// the continuation as an exact append and reuse state it must not.</para>
    ///
    /// <para>Driven through a real <see cref="HotSession"/> so the cursor under test is one
    /// <c>BuildNextCursor</c> actually assembled.</para>
    /// </summary>
    [Fact]
    public async Task HotSession_MismatchAtPriorTurnClosingMarker_RequiresReplay()
    {
        var fwd = new FakeForwardPass { EmitNonStopOnPrefill = true, EmitNonStopOnDecode = true };
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 2 };

        var t1 = await session.RunTurnAsync("one", sampling,
            SessionRevision.Initial, SessionOperationId.New(), Digest("t1"));
        var t2 = await session.RunTurnAsync("two", sampling,
            t1.Operation.CommittedRevision!.Value, SessionOperationId.New(), Digest("t2"));

        var log = t2.Cursor.ExecutionLog;
        Assert.True(log.Length >= 3, $"expected a multi-turn log, got {log.Length} segments");

        // Control: the cursor's own log must diagnose as an exact append against itself.
        var exact = session.DiagnoseContinuation(log);
        Assert.Equal(ContinuationGrade.ExactLossless, exact.Grade);
        Assert.True(exact.CanAppendWithoutReplay);

        // Now corrupt ONLY the final token of turn 1's generated segment — the closing marker.
        var gen1 = Assert.IsType<TokenSegment>(log[1]);
        var mutatedIds = gen1.TokenIds.ToArray();
        mutatedIds[^1] = mutatedIds[^1] + 1;
        var target = log.SetItem(1, new TokenSegment(mutatedIds));

        var diverged = session.DiagnoseContinuation(target);

        Assert.False(diverged.CanAppendWithoutReplay,
            "a mismatch at a prior turn's closing marker must not be reported as an exact append");
        Assert.Equal(ContinuationGrade.ReplayedFromExecutionLog, diverged.Grade);
        Assert.Equal(SessionReuseReason.PrefixDivergence, diverged.ReuseReason);
        // The divergence must be located at the closing marker, not at the start of the next turn.
        Assert.Equal(1, diverged.DivergenceSegmentIndex);
        Assert.Equal(gen1.TokenIds.Length - 1, diverged.DivergencePositionInSegment);
    }

    /// <summary>
    /// Milestone 1 required invariant: a disconnected transport cannot stall the worker.
    ///
    /// <para>The property is structural rather than incidental: both output channels are
    /// <b>unbounded</b> and the batcher only ever <c>TryWrite</c>s (never awaits a writer), while
    /// admission caps <c>MaxNewTokens</c> at <c>MaxBufferedOutputChunks - 4</c> so an abandoned
    /// reader cannot accumulate without limit. This test pins the consequence: one consumer walking
    /// away mid-generation must not prevent another session from completing.</para>
    ///
    /// <para>Request A is enumerated once and then abandoned while it still has tokens to produce,
    /// so the batcher keeps writing into a channel nobody is reading. Session B must still finish.
    /// If the writer ever blocked, B would hang and the test would time out rather than fail an
    /// assertion — so the timeout IS the assertion, and it is kept short deliberately.</para>
    /// </summary>
    [Fact]
    public async Task HotSession_AbandonedConsumer_DoesNotStallOtherSessions()
    {
        var fwd = new FakeForwardPass { EmitNonStopOnPrefill = true, EmitNonStopOnDecode = true };
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 2);

        // A: a long generation whose reader stops consuming after the first chunk.
        //
        // The enumerator is deliberately NOT disposed until the end. Disposing it CANCELS the
        // request — which is correct behaviour, but it is the opposite of the scenario under test:
        // a cancelled request is tidily retired, so nothing is abandoned and the invariant is never
        // exercised. (An earlier version of this test disposed inside an `await using` and was
        // vacuous for exactly that reason; the non-vacuity guard below caught it on every run.)
        fwd.DecodeDelay = TimeSpan.FromMilliseconds(2);
        var abandoned = engine.GenerateChunksAsync("one",
            new SamplingParams { Temperature = 0f, MaxNewTokens = 64 }, CancellationToken.None);
        var reader = abandoned.GetAsyncEnumerator();
        await reader.MoveNextAsync();   // take exactly one chunk, then stop reading — but stay alive

        // B: an ordinary session turn on the same engine. It must complete promptly.
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();
        var turn = session.RunTurnAsync("two",
            new SamplingParams { Temperature = 0f, MaxNewTokens = 1 },
            SessionRevision.Initial, SessionOperationId.New(), Digest("survivor"));

        var completed = await turn.WaitAsync(TimeSpan.FromSeconds(15));

        Assert.Equal(SessionOperationState.Completed, completed.Operation.State);
        Assert.Equal(new SessionRevision(1), completed.Operation.CommittedRevision);

        // NON-VACUITY GUARD. The assertion above is "B finished", which a timeout expresses only
        // negatively — and it would pass trivially if A had already retired, or was never admitted,
        // before B ran. Then nothing would have been abandoned and nothing tested. A generates 64
        // tokens with EOS never sampled while B needs 1, so A must still be in flight here.
        Assert.True(engine.ActiveRequests > 0 || engine.QueueDepth > 0,
            "the abandoned request had already retired, so this run never exercised an abandoned "
            + "consumer at all — the timeout assertion above passed vacuously.");

        await reader.DisposeAsync();
    }

    /// <summary>
    /// Milestone 1 required invariant: stop-sequence completion at the same token.
    ///
    /// <para>A configured stop token that is NOT the model's EOS must terminate the turn at that
    /// token, and the turn must still commit normally — a stop is a completion, not a failure.</para>
    ///
    /// <para><b>This deliberately drives the `ActivateSeq` early-completion branch</b> — the "first
    /// sampled token is already a stop token" path. §8.2 fixed an ordering bug there by inspection
    /// and recorded that no fixture exercised it. This one does: the fake emits token 7 from
    /// prefill and 7 is registered as an additional stop, so the very first sampled token is a
    /// stop and the turn completes without ever entering the decode loop. If the outcome were
    /// still published after the channel closed, this would surface as a Failed operation exactly
    /// as the §8.2 race did.</para>
    /// </summary>
    [Fact]
    public async Task HotSession_StopTokenOnFirstSample_CompletesAtThatTokenAndCommits()
    {
        var fwd = new FakeForwardPass { EmitNonStopOnPrefill = true, EmitNonStopOnDecode = true };
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();

        // Token 7 is what the fake samples; register it as a stop alongside the EOG set.
        var sampling = new SamplingParams
        {
            Temperature = 0f,
            MaxNewTokens = 8,
            AdditionalStopTokenIds = [NonStopToken],
        };

        var result = await session.RunTurnAsync("one", sampling,
            SessionRevision.Initial, SessionOperationId.New(), Digest("stop"));

        Assert.True(result.Operation.State == SessionOperationState.Completed,
            $"state={result.Operation.State} reason={result.Operation.FailureReason}");
        Assert.Equal(new SessionRevision(1), result.Operation.CommittedRevision);

        // Stopped AT the stop token: it terminates the turn and is not emitted as output, so the
        // budget of 8 is untouched and nothing was generated.
        Assert.Equal(0, GeneratedTokenCount(result.Cursor));
        Assert.Equal(result.Cursor.AcceptedPositionCount, result.Cursor.MaterializedPositionCount);

        // NON-VACUITY: without the stop registration the same fake would run to the full budget,
        // so a passing assertion above must be caused by the stop token and nothing else.
        using var control = runtime.Create();
        var unstopped = await control.RunTurnAsync("one",
            new SamplingParams { Temperature = 0f, MaxNewTokens = 8 },
            SessionRevision.Initial, SessionOperationId.New(), Digest("control"));
        Assert.Equal(8, GeneratedTokenCount(unstopped.Cursor));
    }

    /// <summary>Tokens recorded by the generated segments of a cursor (the odd entries).</summary>
    private static int GeneratedTokenCount(SessionCursor cursor)
    {
        int n = 0;
        for (int i = 1; i < cursor.ExecutionLog.Length; i += 2)
            if (cursor.ExecutionLog[i] is TokenSegment ts) n += ts.TokenIds.Length;
        return n;
    }

    private sealed class FakeForwardPass : IBatchedForwardPass
    {
        private static readonly float[] EosLogits = CreateLogits(Eos);
        private static readonly float[] NonStopLogits = CreateLogits(NonStopToken);
        public List<FakeCache> Created { get; } = [];
        public List<(int Start, int Count)> Prefills { get; } = [];
        public bool EmitNonStopOnPrefill { get; set; }
        /// <summary>Never emit EOS from decode, so generation can only end by hitting MaxNewTokens.</summary>
        public bool EmitNonStopOnDecode { get; set; }

        /// <summary>
        /// Per-decode-step delay. Without it the fake retires a 64-token request in ~3 ms, so a
        /// test that needs one request to still be in flight while another completes cannot
        /// construct that state at all. Each batcher iteration advances every active sequence, so
        /// a short delay makes step COUNT the thing that separates a long request from a short one.
        /// </summary>
        public TimeSpan DecodeDelay { get; set; }
        public bool BlockPrefill { get; set; }
        public bool BlockDecode { get; set; }
        public TaskCompletionSource PrefillStarted { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public ManualResetEventSlim ReleasePrefill { get; } = new(false);
        public TaskCompletionSource DecodeStarted { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public ManualResetEventSlim ReleaseDecode { get; } = new(false);
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 1;
        public int MaxSeqLen => 64;
        public bool PrefillDequantCacheActive => false;

        public ISequenceKvCache CreateCache()
        {
            var cache = new FakeCache();
            Created.Add(cache);
            return cache;
        }

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
        {
            var retained = Assert.IsType<FakeCache>(cache);
            Assert.Equal(startPos, retained.LogicalPosition);
            retained.LogicalPosition += tokens.Count;
            Prefills.Add((startPos, tokens.Count));
            if (BlockPrefill)
            {
                PrefillStarted.TrySetResult();
                if (!ReleasePrefill.Wait(TimeSpan.FromSeconds(10)))
                    throw new TimeoutException("The test did not release prefill.");
            }
            return EmitNonStopOnPrefill ? NonStopLogits : EosLogits;
        }

        public float[]?[] PrefillPackedMulti(
            ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits) =>
            throw new NotSupportedException();

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            for (int i = 0; i < caches.Length; i++)
            {
                var cache = Assert.IsType<FakeCache>(caches[i]);
                Assert.Equal(positions[i], cache.LogicalPosition);
                cache.LogicalPosition++;
            }
            if (DecodeDelay > TimeSpan.Zero) Thread.Sleep(DecodeDelay);
            if (BlockDecode)
            {
                DecodeStarted.TrySetResult();
                if (!ReleaseDecode.Wait(TimeSpan.FromSeconds(10)))
                    throw new TimeoutException("The test did not release decode.");
            }
            return Enumerable.Repeat(EmitNonStopOnDecode ? NonStopLogits : EosLogits, tokens.Length).ToArray();
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
