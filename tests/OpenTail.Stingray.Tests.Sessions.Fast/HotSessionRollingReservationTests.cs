using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;

namespace OpenTail.Stingray.Tests.Sessions;

public sealed class HotSessionRollingReservationTests
{
    private const int NonStopToken = 7;
    private const int Eos = 31;
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
        // One byte per token so emitted TEXT is observable. With an empty DecodeBytes the
        // engine produces no text chunks at all, and "was anything emitted that was not logged?"
        // becomes unaskable — which is exactly how the bug below went unnoticed.
        public byte[] DecodeBytes(int token) => [(byte)('a' + (token % 26))];
    }

    private sealed class FakeCache : IRewindableSequenceKvCache
    {
        public int LogicalPosition { get; set; }
        public bool CanRewindTo(int logicalPosition) => logicalPosition >= 0 && logicalPosition <= LogicalPosition;
        public void RewindTo(int logicalPosition) => LogicalPosition = logicalPosition;
        public void Dispose() { }
    }

    private sealed class FakeForwardPass : IBatchedForwardPass
    {
        private static readonly float[] NonStopLogits = CreateLogits(NonStopToken);
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 10;
        public int MaxSeqLen => 64;
        public bool PrefillDequantCacheActive => false;

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
            return Enumerable.Repeat(NonStopLogits, tokens.Length).ToArray();
        }

        private static float[] CreateLogits(int token)
        {
            var logits = new float[64];
            logits[token] = 1f;
            return logits;
        }
    }

    [Fact]
    public async Task HotSession_RollingReservation_AllowsTurnToStart_AndStopsAtResourceBudgetCeiling()
    {
        // 1 token = 10 bytes.
        // Prompt = 2 tokens (20 bytes).
        // MaxSessionBytes = 50 bytes (fits up to 5 tokens total: 2 prompt + 3 generated).
        // MaxNewTokens = 10 (without rolling reservation, initial request = 12 tokens = 120 bytes -> budget exceeded).
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var options = new HotSessionRuntimeOptions(maxResidentBytes: 50, maxSessionBytes: 50);
        var runtime = new HotSessionRuntime(engine, new Tokenizer(), options);
        using var session = runtime.Create();

        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };
        var operationId = SessionOperationId.New();

        var result = await session.RunTurnAsync("hello", sampling, SessionRevision.Initial, operationId, Digest("hello"));

        Assert.True(result.Operation.State == SessionOperationState.Completed,
            $"Expected Completed but state was {result.Operation.State}, failureReason={result.Operation.FailureReason}");
        // MaterializedPositionCount should stop at 5 (50 bytes / 10 bytes-per-token), not 12.
        Assert.Equal(5, result.Cursor.MaterializedPositionCount);

        // EMITTED == LOGGED. The decode loop always carries one sampled-but-unmaterialised token.
        // An earlier version of the rolling reservation stopped BEFORE the step that would have
        // materialised it, then deleted it from the execution log to keep accepted == materialized
        // — after its text had already gone to the consumer. The caller then held text for a token
        // the log did not contain, and any later turn continued from a history the user never saw.
        int emittedChars = result.Chunks.Where(c => c.Kind == GenerateChunkKind.Text).Sum(c => c.Text.Length);
        int loggedGenerated = 0;
        for (int i = 1; i < result.Cursor.ExecutionLog.Length; i += 2)
            if (result.Cursor.ExecutionLog[i] is TokenSegment ts) loggedGenerated += ts.TokenIds.Length;

        Assert.True(emittedChars == loggedGenerated,
            $"caller received {emittedChars} characters of generated text but the execution log "
            + $"records {loggedGenerated} generated tokens (1 char per token in this fixture). "
            + "Nothing may be emitted that the log does not contain.");
        Assert.True(loggedGenerated > 0, "test is vacuous: no tokens were generated at all.");

        // STOP REASON HONESTY. Budget exhaustion must not be reported as "you hit your token
        // limit": the caller's remedy is opposite (free state / back off, versus retry with a
        // larger MaxNewTokens). MaxNewTokens was 10 and only 3 tokens were generated, so a
        // TruncatedByMaxTokens here would be a plain falsehood.
        var stop = result.Chunks.Single(c => c.Kind == GenerateChunkKind.Stop);
        Assert.True(stop.TruncatedByResourceBudget,
            "generation stopped because the KV byte budget ran out; the Stop chunk must say so.");
        Assert.False(stop.TruncatedByMaxTokens,
            $"stop reported TruncatedByMaxTokens, but only {loggedGenerated} of "
            + "10 permitted tokens were generated — the limit was never reached.");
    }

    /// <summary>
    /// Characterises what rolling reservation ACTUALLY does when two sessions share a global
    /// budget: it is first-come-first-served, and a session that starts earlier can grow into the
    /// whole budget and starve one that starts later.
    ///
    /// <para><b>This is a known gap, not a passing feature.</b> Plan §7 still lists "bounded
    /// admission and output queues", "fair waiters" and "per-model/device budget partitions" as
    /// unchecked. Renewal today asks only "is there capacity right now", with no notion of
    /// reserving a share for an admitted peer — so A grows to the ceiling and B is refused at
    /// admission.</para>
    ///
    /// <para>An earlier version of this test asserted BOTH sessions complete and failed for
    /// exactly this reason. A permanently red test trains people to ignore failures, so it is
    /// written here as an assertion about current behaviour instead. <b>When fairness lands this
    /// test must fail</b> — that failure is the signal to rewrite it as the fairness spec, not to
    /// relax it.</para>
    /// </summary>
    [Fact]
    public async Task HotSession_RollingReservation_StarvesAPeerOnlyWhenSessionCapEqualsGlobalBudget()
    {
        // Global budget 80 bytes = 8 tokens at 10 bytes/token. Each prompt is 2 tokens, so the
        // initial reservation is prompt + 1 = 30 bytes; A's renewals then grow into the rest.
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test-concurrent", maxBatchSize: 2);
        var options = new HotSessionRuntimeOptions(maxResidentBytes: 80, maxSessionBytes: 80);
        var runtime = new HotSessionRuntime(engine, new Tokenizer(), options);

        using var sessionA = runtime.Create();
        using var sessionB = runtime.Create();
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };

        var first = await sessionA.RunTurnAsync("hello", sampling,
            SessionRevision.Initial, SessionOperationId.New(), Digest("turn-A"));

        Assert.Equal(SessionOperationState.Completed, first.Operation.State);
        // A grew until the global ceiling stopped it, and the stop is reported honestly.
        var stopA = first.Chunks.Single(c => c.Kind == GenerateChunkKind.Stop);
        Assert.True(stopA.TruncatedByResourceBudget);

        // B is now refused at admission — not throttled, not queued. THIS is the gap.
        var starved = await Assert.ThrowsAsync<SessionResourceBudgetExceededException>(() =>
            sessionB.RunTurnAsync("world", sampling,
                SessionRevision.Initial, SessionOperationId.New(), Digest("turn-B")));

        Assert.True(starved.AvailableBytes < starved.RequestedBytes,
            $"expected B to be starved, but {starved.AvailableBytes} bytes were available for a "
            + $"{starved.RequestedBytes}-byte request — capacity was not actually exhausted.");

        // And the global accounting is exact: everything resident belongs to A.
        Assert.Equal(first.Cursor.MaterializedPositionCount * 10, runtime.ResidentBytes);

        // ── THE REMEDY, demonstrated in the same test so the gap is never read as unfixable ──
        // Starvation above is a CONFIGURATION outcome, not a missing mechanism: MaxSessionBytes
        // was set equal to the global budget, so one session was entitled to all of it. Give each
        // session a share and the same workload no longer starves anyone.
        // Stating the expected concurrency derives the same 40-byte share, without the caller
        // having to compute it — the footgun fix.
        var fairOptions = new HotSessionRuntimeOptions(
            maxResidentBytes: 80, expectedConcurrentSessions: 2);
        Assert.Equal(40, fairOptions.MaxSessionBytes);
        var fairRuntime = new HotSessionRuntime(engine, new Tokenizer(), fairOptions);
        using var fairA = fairRuntime.Create();
        using var fairB = fairRuntime.Create();

        var a = await fairA.RunTurnAsync("hello", sampling,
            SessionRevision.Initial, SessionOperationId.New(), Digest("fair-A"));
        var b = await fairB.RunTurnAsync("world", sampling,
            SessionRevision.Initial, SessionOperationId.New(), Digest("fair-B"));

        Assert.Equal(SessionOperationState.Completed, a.Operation.State);
        Assert.Equal(SessionOperationState.Completed, b.Operation.State);
        Assert.True(fairRuntime.ResidentBytes <= 80,
            $"the pair must still respect the global ceiling; resident was {fairRuntime.ResidentBytes}.");
    }

    /// <summary>
    /// The derivation must never silently override a cap the caller stated explicitly, and must
    /// stay inert when there is no global budget to divide.
    /// </summary>
    [Fact]
    public void ExpectedConcurrentSessions_DerivesShare_ButNeverOverridesAnExplicitCap()
    {
        // Derived when the caller gave a global budget and no per-session cap.
        Assert.Equal(25, new HotSessionRuntimeOptions(
            maxResidentBytes: 100, expectedConcurrentSessions: 4).MaxSessionBytes);

        // An explicit cap wins — stating one is a deliberate decision.
        Assert.Equal(90, new HotSessionRuntimeOptions(
            maxResidentBytes: 100, maxSessionBytes: 90, expectedConcurrentSessions: 4).MaxSessionBytes);

        // Nothing to divide: no global budget means no derivation.
        Assert.Equal(long.MaxValue, new HotSessionRuntimeOptions(
            expectedConcurrentSessions: 4).MaxSessionBytes);

        // Concurrency of 1 is legal and means "this session may use everything".
        Assert.Equal(100, new HotSessionRuntimeOptions(
            maxResidentBytes: 100, expectedConcurrentSessions: 1).MaxSessionBytes);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new HotSessionRuntimeOptions(maxResidentBytes: 100, expectedConcurrentSessions: 0));
    }
}
