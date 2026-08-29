namespace OpenTail.Stingray.Tests.Sessions.Fast;


/// <summary>
/// docs/028 Phase 2: <see cref="HotSessionRuntime.CreateWithSharedPrefixHint"/> seeds a brand-new
/// session from an idle sibling's already-materialized prefix instead of starting cold, using the
/// same <see cref="IPrefixCacheableBatchedForwardPass"/> capture/fork primitives
/// <see cref="ContinuousBatchingEngine"/>'s own cross-request prefix cache already relies on. This
/// file exercises the orchestration (matching, page-alignment flooring, cursor/budget seeding,
/// and every fallback-to-cold path) against a fake forward pass. The real-model proof that a
/// forked cache genuinely shares physical pages (not just produces the same test outcome) lives in
/// the heavy Tests.Sessions project, since a fake cannot represent shared memory at all.
/// </summary>
public sealed class CrossSessionPrefixForkingTests
{
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
        public bool Disposed { get; private set; }
        public bool CanRewindTo(int logicalPosition) => logicalPosition >= 0 && logicalPosition <= LogicalPosition;
        public void RewindTo(int logicalPosition) => LogicalPosition = logicalPosition;
        public void Dispose() => Disposed = true;
    }

    private sealed class FakeForwardPass : IBatchedForwardPass, IPrefixCacheableBatchedForwardPass
    {
        private const int GeneratedToken = 7;
        private static readonly float[] GenLogits = CreateLogits(GeneratedToken);
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 10;
        public int MaxSeqLen => 64;
        public bool PrefillDequantCacheActive => false;

        // Small and non-default so the alignment-flooring behavior is actually exercised by these
        // tests rather than coincidentally matching a round number.
        public int PrefixCacheBlockSize => 4;

        public ISequenceKvCache CreateCache() => new FakeCache();

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
        {
            var retained = Assert.IsType<FakeCache>(cache);
            retained.LogicalPosition = startPos + tokens.Count;
            return GenLogits;
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
            return Enumerable.Repeat(GenLogits, tokens.Length).ToArray();
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

    /// <summary>Runs two turns on <paramref name="session"/>, leaving it idle afterward with 6
    /// materialized positions ([1,2],[7],[1,2],[7]) — enough for a clean, block-aligned (4-token)
    /// partial match once floored by <see cref="FakeForwardPass.PrefixCacheBlockSize"/>.</summary>
    private static async Task<ImmutableArray<int>> RunTwoTurnsAsync(HotSession session)
    {
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };
        var first = await session.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("hello1"));
        await session.RunTurnAsync("world", sampling, first.Operation.CommittedRevision!.Value, SessionOperationId.New(), Digest("hello2"));
        return [.. session.Cursor.ExecutionLog.SelectMany(s => ((TokenSegment)s).TokenIds)];
    }

    [Fact]
    public async Task CreateWithSharedPrefixHint_MatchingIdleSibling_SeedsAlignedPrefix()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());

        using var sessionA = runtime.Create();
        var history = await RunTwoTurnsAsync(sessionA);
        Assert.Equal(6, history.Length);

        var (sessionB, seeded) = runtime.CreateWithSharedPrefixHint(history);
        using var _ = sessionB;

        // 6 matched positions floors to 4 under a block size of 4 -- the fork must never claim more
        // than what's actually page-aligned, even though every one of the 6 tokens genuinely matched.
        Assert.Equal(4, seeded);
        Assert.Equal(4, sessionB.Cursor.MaterializedPositionCount);
        Assert.Equal(4, sessionB.Cursor.AcceptedPositionCount);
        var seededSegment = Assert.Single(sessionB.Cursor.ExecutionLog);
        Assert.Equal(history[..4], Assert.IsType<TokenSegment>(seededSegment).TokenIds);

        // The seeded session's first turn only has to submit whatever comes after the shared
        // prefix -- exercised here with a distinct suffix to confirm the seeded state composes
        // correctly with a real subsequent turn rather than just looking right at rest.
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };
        var result = await sessionB.RunTurnAsync("suffix", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("suffix"));
        Assert.Equal(SessionOperationState.Completed, result.Operation.State);
        Assert.Equal(4 + 2 + 1, sessionB.Cursor.MaterializedPositionCount);
    }

    [Fact]
    public async Task CreateWithSharedPrefixHint_NoMatchingSibling_FallsBackToColdSession()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());

        using var sessionA = runtime.Create();
        await RunTwoTurnsAsync(sessionA);

        // Shares no leading token at all with A's recorded history.
        ImmutableArray<int> unrelated = [99, 98, 97, 96];
        var (sessionB, seeded) = runtime.CreateWithSharedPrefixHint(unrelated);
        using var _ = sessionB;

        Assert.Equal(0, seeded);
        Assert.Empty(sessionB.Cursor.ExecutionLog);
        Assert.Equal(0, sessionB.Cursor.MaterializedPositionCount);
    }

    [Fact]
    public async Task CreateWithSharedPrefixHint_MatchBelowAlignmentBlock_FallsBackToColdSession()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());

        // One turn only -> 3 materialized positions ([1,2],[7]), below the 4-token alignment block,
        // so even an exact full match must floor to 0 rather than share a partial, unaligned page.
        using var sessionA = runtime.Create();
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };
        await sessionA.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));
        ImmutableArray<int> history = [.. sessionA.Cursor.ExecutionLog.SelectMany(s => ((TokenSegment)s).TokenIds)];
        Assert.Equal(3, history.Length);

        var (sessionB, seeded) = runtime.CreateWithSharedPrefixHint(history);
        using var _ = sessionB;

        Assert.Equal(0, seeded);
        Assert.Empty(sessionB.Cursor.ExecutionLog);
    }

    [Fact]
    public async Task CreateWithSharedPrefixHint_NoIdleSiblings_FallsBackToColdSession()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());

        ImmutableArray<int> desired = [1, 2, 3, 4];
        var (session, seeded) = runtime.CreateWithSharedPrefixHint(desired);
        using var _ = session;

        Assert.Equal(0, seeded);
        Assert.Empty(session.Cursor.ExecutionLog);
    }
}
