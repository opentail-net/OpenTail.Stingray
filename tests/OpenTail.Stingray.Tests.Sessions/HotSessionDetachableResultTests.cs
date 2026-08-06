using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;

namespace OpenTail.Stingray.Tests.Sessions;

public sealed class HotSessionDetachableResultTests
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
        public byte[] DecodeBytes(int token) => [];
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
        private static readonly float[] EosLogits = CreateLogits(Eos);
        private static readonly float[] NonStopLogits = CreateLogits(7);
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 1;
        public int MaxSeqLen => 64;
        public bool PrefillDequantCacheActive => false;
        public bool EmitNonStopOnPrefill { get; set; }
        public TimeSpan DecodeDelay { get; set; }

        public ISequenceKvCache CreateCache() => new FakeCache();

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
        {
            var retained = Assert.IsType<FakeCache>(cache);
            retained.LogicalPosition = startPos + tokens.Count;
            return EmitNonStopOnPrefill ? NonStopLogits : EosLogits;
        }

        public float[]?[] PrefillPackedMulti(ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits) =>
            throw new NotSupportedException();

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            if (DecodeDelay > TimeSpan.Zero) Thread.Sleep(DecodeDelay);
            for (int i = 0; i < caches.Length; i++)
            {
                if (caches[i] is FakeCache fc) fc.LogicalPosition++;
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
    public async Task HotSession_GetOperation_ReturnsCompletedOperationWithResultChunks()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };
        var operationId = SessionOperationId.New();

        var result = await session.RunTurnAsync("hello", sampling, SessionRevision.Initial, operationId, Digest("hello"));

        Assert.Equal(SessionOperationState.Completed, result.Operation.State);

        var retrieved = runtime.GetOperation(session.SessionId, operationId);

        Assert.Equal(operationId, retrieved.OperationId);
        Assert.Equal(SessionOperationState.Completed, retrieved.State);
        Assert.Equal(new SessionRevision(1), retrieved.CommittedRevision);
        Assert.NotNull(retrieved.ResultChunks);
        Assert.NotEmpty(retrieved.ResultChunks);
        Assert.Equal(result.Chunks.Length, retrieved.ResultChunks.Count);
    }

    [Fact]
    public void HotSession_GetOperation_UnknownOperationOrSession_ThrowsException()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();

        Assert.Throws<SessionNotFoundException>(() => runtime.GetOperation(SessionId.New(), SessionOperationId.New()));
        Assert.Throws<KeyNotFoundException>(() => runtime.GetOperation(session.SessionId, SessionOperationId.New()));
    }

    [Fact]
    public async Task HotSession_GetSessionSnapshot_ReturnsActiveSessionState()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };
        var operationId = SessionOperationId.New();

        await session.RunTurnAsync("hello", sampling, SessionRevision.Initial, operationId, Digest("hello"));

        var snapshot = runtime.GetSessionSnapshot(session.SessionId);

        Assert.Equal(session.SessionId, snapshot.SessionId);
        Assert.Equal(new SessionRevision(1), snapshot.CommittedRevision);
        Assert.Single(snapshot.Operations);
        Assert.Equal(operationId, snapshot.Operations.First().OperationId);
    }

    [Fact]
    public async Task HotSession_GetOperation_ReturnsCancelledOperation_OnCancellation()
    {
        var fwd = new FakeForwardPass { EmitNonStopOnPrefill = true, DecodeDelay = TimeSpan.FromMilliseconds(50) };
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();
        using var cts = new CancellationTokenSource();

        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };
        var operationId = SessionOperationId.New();

        // Cancel while turn is in-flight
        cts.CancelAfter(5);

        try
        {
            await session.RunTurnAsync("hello", sampling, SessionRevision.Initial, operationId, Digest("hello"), cts.Token);
        }
        catch (OperationCanceledException) { }

        var retrieved = runtime.GetOperation(session.SessionId, operationId);

        Assert.Equal(operationId, retrieved.OperationId);
        Assert.Equal(SessionOperationState.Cancelled, retrieved.State);
        Assert.Null(retrieved.CommittedRevision);
    }
}
