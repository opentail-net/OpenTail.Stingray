namespace OpenTail.Stingray.Tests.Sessions.Fast;

/// <summary>
/// Phase 3 retained-state budgeting (Qwen 3.8 plan): a held retained state costs its per-token KV plus its fixed
/// sequence state, and reclaiming an idle session frees both. Synthetic, no checkpoint.
/// </summary>
public sealed class HotSessionFixedStateTests
{
    private const int NonStopToken = 7;
    private const int Eos = 31;
    private const long KvPerToken = 10;
    private const long Fixed = 1000;

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

    /// <summary>Never stops on its own; reports the given fixed state per sequence.</summary>
    private sealed class FakeForwardPass(long fixedBytes) : IBatchedForwardPass
    {
        private static readonly float[] NonStopLogits = CreateLogits(NonStopToken);
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => KvPerToken;
        public long FixedBytesPerSequence => fixedBytes;
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
                Assert.IsType<FakeCache>(caches[i]).LogicalPosition++;
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
    public async Task CompletedTurn_ChargesTheFixedStateOnTopOfThePerTokenKv()
    {
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        // Read residency while the session is still open: disposing it removes its committed bytes.
        using var plainEngine = new ContinuousBatchingEngine(new FakeForwardPass(0), new Tokenizer(), "plain", maxBatchSize: 1);
        var plainRuntime = new HotSessionRuntime(plainEngine, new Tokenizer(), new HotSessionRuntimeOptions(maxResidentBytes: long.MaxValue));
        long plain;
        using (var s = plainRuntime.Create())
        {
            await s.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));
            plain = plainRuntime.ResidentBytes;
        }

        using var fixedEngine = new ContinuousBatchingEngine(new FakeForwardPass(Fixed), new Tokenizer(), "fixed", maxBatchSize: 1);
        var fixedRuntime = new HotSessionRuntime(fixedEngine, new Tokenizer(), new HotSessionRuntimeOptions(maxResidentBytes: long.MaxValue));
        long withFixed;
        using (var s = fixedRuntime.Create())
        {
            await s.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));
            withFixed = fixedRuntime.ResidentBytes;
        }

        Assert.True(plain > 0, "A completed turn must leave committed resident bytes behind.");
        Assert.Equal(plain + Fixed, withFixed);
    }

    [Fact]
    public async Task ReclaimingAnIdleSession_FreesItsFixedStateToo()
    {
        // Same shape as SessionResourceBudgetEvictionTests, with fixed state: the budget is exactly two sessions, so a
        // third turn can only be admitted by evicting an idle sibling. If eviction freed only the per-token part, the
        // residency left behind would not match and the third admission would fail.
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        using var probeEngine = new ContinuousBatchingEngine(new FakeForwardPass(Fixed), new Tokenizer(), "probe", maxBatchSize: 1);
        var probeRuntime = new HotSessionRuntime(probeEngine, new Tokenizer(), new HotSessionRuntimeOptions(maxResidentBytes: long.MaxValue));
        long perSession;
        using (var probe = probeRuntime.Create())
        {
            await probe.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));
            perSession = probeRuntime.ResidentBytes;
        }
        long twoSessionBudget = perSession * 2;

        using var engine = new ContinuousBatchingEngine(new FakeForwardPass(Fixed), new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer(), new HotSessionRuntimeOptions(maxResidentBytes: twoSessionBudget));

        using var sessionA = runtime.Create();
        using var sessionB = runtime.Create();
        await sessionA.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));
        await sessionB.RunTurnAsync("world", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("world"));
        Assert.Equal(twoSessionBudget, runtime.ResidentBytes);

        using var sessionC = runtime.Create();
        var resultC = await sessionC.RunTurnAsync("third", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("third"));
        Assert.Equal(SessionOperationState.Completed, resultC.Operation.State);
        Assert.Equal(twoSessionBudget, runtime.ResidentBytes);
    }
}
