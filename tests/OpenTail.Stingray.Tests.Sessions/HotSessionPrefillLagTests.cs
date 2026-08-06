using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;

namespace OpenTail.Stingray.Tests.Sessions;

public sealed class HotSessionPrefillLagTests
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
        public IReadOnlyList<int> Encode(string text) => [1, 2, 3]; // 3 prompt tokens
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
        public long KvBytesPerToken => 1;
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

    /// <summary>
    /// <b>This does NOT satisfy the "first token after prefill and its materialisation lag"
    /// invariant, and must not be used to tick it.</b>
    ///
    /// <para>It asserts that <c>AcceptedPositionCount == MaterializedPositionCount</c> — that
    /// there is NO lag. <c>HotSession.BuildNextCursor</c> ends with
    /// <c>new SessionCursor(log, accepted, accepted, accepted, accepted, ...)</c>, so this holds by
    /// construction and would keep holding if every line of lag-handling were deleted. Plan §4.2
    /// says the two counts are distinct precisely because a sampled token is accepted before it
    /// enters KV; the runtime cannot represent that state at all today (§8.3).</para>
    ///
    /// <para>What it IS worth: a regression guard pinning the current collapsed behaviour, so that
    /// when §4.2 is implemented this test fails loudly and forces the invariant to be written
    /// properly rather than silently drifting.</para>
    /// </summary>
    [Fact]
    public async Task HotSession_Cursor_CollapsesAcceptedAndMaterialized_UntilSection42IsImplemented()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();

        // MaxNewTokens = 1 -> turn ends after 1 sampled token
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };
        var operationId = SessionOperationId.New();

        var result = await session.RunTurnAsync("hello", sampling, SessionRevision.Initial, operationId, Digest("hello"));

        Assert.Equal(SessionOperationState.Completed, result.Operation.State);

        // Prompt = 3 tokens. 1 sampled token generated.
        // MaterializedPositionCount = 3 + 1 = 4.
        // AcceptedPositionCount = 3 + 1 = 4.
        // After completion, the materialisation lag is 0 (all sampled tokens materialized into KV cache).
        Assert.Equal(4, result.Cursor.MaterializedPositionCount);
        Assert.Equal(4, result.Cursor.AcceptedPositionCount);
        Assert.Equal(result.Cursor.MaterializedPositionCount, result.Cursor.AcceptedPositionCount);
    }
}
