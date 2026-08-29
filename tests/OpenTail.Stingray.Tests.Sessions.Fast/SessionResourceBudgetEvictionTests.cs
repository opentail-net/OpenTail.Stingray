namespace OpenTail.Stingray.Tests.Sessions.Fast;


/// <summary>
/// docs/028 Phase 1: <see cref="SessionResourceBudget"/> correctly tracks and bounds total
/// resident bytes, but on its own "does not evict: a session with running work keeps its
/// reservation until the operation commits, rolls back, or fails" — no reclaim from IDLE
/// (turn-completed, not disposed) sessions under pressure. <see cref="HotSessionRuntime"/> now
/// closes that gap: a reservation that fails under pressure triggers one reclaim attempt against
/// idle sibling sessions (<see cref="HotSession.EvictRetainedCacheIfIdle"/>) before failing for
/// real. This test is the regression guard for that path — it started as a characterization test
/// proving the gap (asserting the hard-rejection), and was updated in the same change that
/// implemented the fix to assert successful reclaim instead. See git history for the pre-fix
/// version if the original characterization is ever needed again.
/// </summary>
public sealed class SessionResourceBudgetEvictionTests
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

        private static float[] CreateLogits(int winner)
        {
            var logits = new float[64];
            logits[winner] = 10f;
            return logits;
        }
    }

    [Fact]
    public async Task NewAdmissionUnderPressure_ReclaimsFromIdleSiblingSession()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);

        // Sized for exactly two sessions' committed state, with zero slack for a third -- measured,
        // not guessed: run two sessions to completion first, read the actual committed total, then
        // use that as the ceiling. A generous/round number would leave the "does 3 fit" question to
        // luck.
        var probeOptions = new HotSessionRuntimeOptions(maxResidentBytes: long.MaxValue);
        var probeRuntime = new HotSessionRuntime(engine, new Tokenizer(), probeOptions);
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        // Deliberately not disposed before the measurement below: HotSession.Dispose() calls
        // _resources.Remove(SessionId), which would zero out exactly the contribution being
        // measured. Disposing after the read is correct here.
        using var probeA = probeRuntime.Create();
        await probeA.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));
        long perSessionBytes = probeRuntime.ResidentBytes;
        Assert.True(perSessionBytes > 0, "A completed turn must leave committed resident bytes behind.");

        long twoSessionBudget = perSessionBytes * 2;

        // Fresh engine/runtime under the measured, exact two-session budget.
        using var engine2 = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test2", maxBatchSize: 1);
        var options = new HotSessionRuntimeOptions(maxResidentBytes: twoSessionBudget);
        var runtime = new HotSessionRuntime(engine2, new Tokenizer(), options);

        // Sessions A and B each complete one turn, then go idle (retained, not disposed) -- exactly
        // the "no eviction" scenario docs/028 Phase 1 describes.
        using var sessionA = runtime.Create();
        using var sessionB = runtime.Create();
        await sessionA.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));
        await sessionB.RunTurnAsync("world", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("world"));

        Assert.Equal(twoSessionBudget, runtime.ResidentBytes);

        // A and B are both idle now -- neither has running work. C's turn exhausts the budget on
        // first attempt, triggers reclaim (HotSessionRuntime.ReclaimIdleBytes), which evicts one
        // idle sibling to free exactly enough room, then the reservation is retried and succeeds.
        using var sessionC = runtime.Create();
        var resultC = await sessionC.RunTurnAsync("third", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("third"));
        Assert.Equal(SessionOperationState.Completed, resultC.Operation.State);

        // Exactly one sibling was evicted to make room -- total resident is back at the two-session
        // ceiling (one original survivor + C), not three sessions' worth.
        Assert.Equal(twoSessionBudget, runtime.ResidentBytes);
    }
}
