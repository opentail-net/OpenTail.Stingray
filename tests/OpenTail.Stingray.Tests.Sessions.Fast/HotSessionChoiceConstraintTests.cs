
namespace OpenTail.Stingray.Tests.Sessions.Fast;

/// <summary>
/// docs/030/051: <c>SamplingParams.AllowedChoices</c> constrained-choice sampling used to be
/// implemented only by the now-deleted <c>InferenceSession</c>; it is now wired into
/// <see cref="ContinuousBatchingEngine"/>'s batched decode loop (per-sequence
/// <c>TokenChoiceTrie.ChoiceState</c>) so <see cref="HotSession"/> gets it too. This file is the
/// HotSession-native replacement for <c>ConstrainedChoiceSamplingTests.cs</c>'s coverage.
/// </summary>
public sealed class HotSessionChoiceConstraintTests
{
    private const int Yes = 10;
    private const int Sir = 20;
    private const int Eos = 99;

    [Fact]
    public async Task AllowedChoices_StopsAtFirstToken_WhenThatTokenIsItselfAnAllowedChoice()
    {
        var fwd = new FakeChoiceForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new ChoiceTokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new ChoiceTokenizer());
        using var session = runtime.Create();

        var sampling = new SamplingParams
        {
            Temperature = 0f,
            MaxNewTokens = 10,
            AllowedChoices = ["YES", "YESSIR"],
        };

        var result = await session.RunTurnAsync("start", sampling, SessionRevision.Initial,
            SessionOperationId.New(), SessionRequestDigest.FromCanonicalValue("choice"));

        Assert.Equal(SessionOperationState.Completed, result.Operation.State);
        var text = string.Concat(result.Chunks
            .Where(c => c.Kind == GenerateChunkKind.Text)
            .Select(c => c.Text));

        // Must stop cleanly at "YES" instead of being forced to continue toward "YESSIR" merely
        // because the trie node after token 10 also has a child (the IsComplete-vs-HasChildren
        // regression InferenceSession's Test17 guarded against).
        Assert.Equal("YES", text);
    }

    [Fact]
    public async Task AllowedChoices_ContinuesThroughTheDecodeLoop_UntilTheChoiceCompletes()
    {
        var fwd = new FakeChoiceForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new ChoiceTokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new ChoiceTokenizer());
        using var session = runtime.Create();

        var sampling = new SamplingParams
        {
            Temperature = 0f,
            MaxNewTokens = 10,
            AllowedChoices = ["YESSIR"],
        };

        var result = await session.RunTurnAsync("start", sampling, SessionRevision.Initial,
            SessionOperationId.New(), SessionRequestDigest.FromCanonicalValue("choice-multi"));

        Assert.Equal(SessionOperationState.Completed, result.Operation.State);
        var text = string.Concat(result.Chunks
            .Where(c => c.Kind == GenerateChunkKind.Text)
            .Select(c => c.Text));
        Assert.Equal("YESSIR", text);

        // Stopped by the choice completing, not by exhausting MaxNewTokens.
        var stop = Assert.Single(result.Chunks.Where(c => c.Kind == GenerateChunkKind.Stop));
        Assert.False(stop.TruncatedByMaxTokens);
    }

    private sealed class ChoiceTokenizer : ITokenizer
    {
        public int VocabSize => 32;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;
        public IReadOnlyCollection<int> EogTokenIds => [Eos];
        public IReadOnlyList<int> Encode(string text) => text switch
        {
            "start" => [1],
            "YES" => [Yes],
            "YESSIR" => [Yes, Sir],
            _ => throw new ArgumentOutOfRangeException(nameof(text)),
        };
        public string Decode(IEnumerable<int> tokens) => string.Empty;
        public byte[] DecodeBytes(int token) => token switch
        {
            Yes => "YES"u8.ToArray(),
            Sir => "SIR"u8.ToArray(),
            _ => [],
        };
    }

    /// <summary>
    /// Uniform logits: with an active choice constraint every non-allowed token gets masked to
    /// -inf, so the greedy pick within a single-token-wide allowed set is deterministic regardless
    /// of the raw values here.
    /// </summary>
    private sealed class FakeChoiceForwardPass : IBatchedForwardPass
    {
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 1;
        public int MaxSeqLen => 64;
        public bool PrefillDequantCacheActive => false;

        public ISequenceKvCache CreateCache() => new FakeCache();

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
        {
            var retained = Assert.IsType<FakeCache>(cache);
            retained.LogicalPosition += tokens.Count;
            return new float[32];
        }

        public float[]?[] PrefillPackedMulti(
            ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits) =>
            throw new NotSupportedException();

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            for (int i = 0; i < caches.Length; i++)
                Assert.IsType<FakeCache>(caches[i]).LogicalPosition++;
            return Enumerable.Repeat(new float[32], tokens.Length).ToArray();
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
