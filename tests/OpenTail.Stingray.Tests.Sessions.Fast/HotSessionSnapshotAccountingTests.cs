namespace OpenTail.Stingray.Tests.Sessions.Fast;

/// <summary>
/// Job 3 part (b), Qwen 3.8 plan: a retained session that holds a turn-start snapshot is charged for it. The snapshot's
/// bytes are zero before a snapshot exists, counted once a turn has started from one, and freed on eviction. Synthetic.
/// </summary>
public sealed class HotSessionSnapshotAccountingTests
{
    private const int NonStopToken = 7;
    private const int Eos = 31;
    private const long KvPerToken = 10;
    private const long Fixed = 1000;

    private static SessionRequestDigest Digest(string val) => SessionRequestDigest.FromCanonicalValue(val);

    private static readonly SamplingParams Sampling = new() { Temperature = 0f, MaxNewTokens = 1 };

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

    /// <summary>A cache that reports a snapshot's held bytes once a turn has started from a snapshot.</summary>
    private sealed class SnapshotCache(long snapshotBytes) : IRewindableSequenceKvCache, ISnapshotAccountedSequenceKvCache
    {
        public int LogicalPosition { get; set; }
        public bool HasSnapshot { get; set; }
        public long HeldSnapshotBytes => HasSnapshot ? snapshotBytes : 0;
        public bool CanRewindTo(int logicalPosition) => logicalPosition >= 0 && logicalPosition <= LogicalPosition;
        public void RewindTo(int logicalPosition) => LogicalPosition = logicalPosition;
        public void Dispose() { }
    }

    /// <summary>Never stops on its own. A turn that starts after earlier positions takes a snapshot, as the Qwen pass does.</summary>
    private sealed class SnapshotForwardPass(long snapshotBytes) : IBatchedForwardPass
    {
        private static readonly float[] NonStopLogits = CreateLogits(NonStopToken);
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => KvPerToken;
        public long FixedBytesPerSequence => Fixed;
        public bool KeepsTurnSnapshot => true;
        public int MaxSeqLen => 64;
        public bool PrefillDequantCacheActive => false;

        public ISequenceKvCache CreateCache() => new SnapshotCache(snapshotBytes);

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
        {
            var c = Assert.IsType<SnapshotCache>(cache);
            if (startPos > 0 && startPos == c.LogicalPosition) c.HasSnapshot = true;
            c.LogicalPosition = startPos + tokens.Count;
            return NonStopLogits;
        }

        public float[]?[] PrefillPackedMulti(ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits) =>
            throw new NotSupportedException();

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            for (int i = 0; i < caches.Length; i++)
                Assert.IsType<SnapshotCache>(caches[i]).LogicalPosition++;
            return Enumerable.Repeat(NonStopLogits, tokens.Length).ToArray();
        }

        private static float[] CreateLogits(int winner)
        {
            var logits = new float[64];
            logits[winner] = 10f;
            return logits;
        }
    }

    /// <summary>Two completed turns on one session. Returns the session's residency after each turn, read while it is open.</summary>
    private static async Task<(long AfterTurn1, long AfterTurn2)> TwoTurnsAsync(long snapshotBytes)
    {
        using var engine = new ContinuousBatchingEngine(new SnapshotForwardPass(snapshotBytes), new Tokenizer(), "snap", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer(), new HotSessionRuntimeOptions(maxResidentBytes: long.MaxValue));
        using var session = runtime.Create();
        await session.RunTurnAsync("hello", Sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));
        long after1 = runtime.ResidentBytes;
        await session.RunTurnAsync("world", Sampling, runtime.GetSessionSnapshot(session.SessionId).CommittedRevision, SessionOperationId.New(), Digest("world"));
        long after2 = runtime.ResidentBytes;
        return (after1, after2);
    }

    [Fact]
    public async Task SnapshotBytes_AreZeroBeforeASnapshotExists_AndCountedAfterTheTurnThatStartsFromIt()
    {
        var without = await TwoTurnsAsync(snapshotBytes: 0);
        var with = await TwoTurnsAsync(snapshotBytes: 500);

        // Turn 1 starts from nothing, so no snapshot exists and the figures match exactly.
        Assert.Equal(without.AfterTurn1, with.AfterTurn1);
        // Turn 2 starts from turn 1's boundary, so the snapshot exists and is charged.
        Assert.Equal(without.AfterTurn2 + 500, with.AfterTurn2);
    }

    [Fact]
    public async Task EvictingASessionWithASnapshot_ReturnsTheSnapshotBytesToo()
    {
        using var engine = new ContinuousBatchingEngine(new SnapshotForwardPass(500), new Tokenizer(), "evict", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer(), new HotSessionRuntimeOptions(maxResidentBytes: long.MaxValue));
        using var session = runtime.Create();
        await session.RunTurnAsync("hello", Sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));
        await session.RunTurnAsync("world", Sampling, runtime.GetSessionSnapshot(session.SessionId).CommittedRevision, SessionOperationId.New(), Digest("world"));
        long held = runtime.ResidentBytes;
        Assert.True(held > 500, "The session holds a snapshot, so its residency must include the snapshot's bytes.");

        long reclaimed = session.EvictRetainedCacheIfIdle();

        Assert.Equal(held, reclaimed);
        Assert.Equal(0, runtime.ResidentBytes);
    }
}
