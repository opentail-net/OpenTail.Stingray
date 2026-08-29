
namespace OpenTail.Stingray.Tests.Sessions.Fast;

/// <summary>
/// docs/030: HotSession-native coverage for the two capabilities ported from InferenceSession
/// before its deletion — session-scoped ISessionMetrics/ISessionMetadata, and FinishReason/
/// ToolCalls bundled into HotSessionTurnResult the way GenerationResult used to bundle them.
/// </summary>
public sealed class HotSessionMetricsMetadataTests
{
    private const int Eos = 31;
    private const int NonStopToken = 7;

    [Fact]
    public async Task HotSession_Metrics_TrackPromptAndGeneratedTokensOnCommit()
    {
        var fwd = new FakeForwardPass { EmitNonStopOnPrefill = true, EmitNonStopOnDecode = true };
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();

        Assert.Equal(0, session.Metrics.PromptTokens);
        Assert.Equal(0, session.Metrics.GeneratedTokens);

        var result = await session.RunTurnAsync("one",
            new SamplingParams { Temperature = 0f, MaxNewTokens = 5 },
            SessionRevision.Initial, SessionOperationId.New(), Digest("metrics"));

        Assert.Equal(SessionOperationState.Completed, result.Operation.State);
        Assert.Equal(2, session.Metrics.PromptTokens);   // "one" encodes to [1, 2]
        Assert.Equal(5, session.Metrics.GeneratedTokens);
        Assert.True(session.Metrics.TotalGenerationTime >= TimeSpan.Zero);
        Assert.True(session.Metrics.KvPagesHeld >= 0);
    }

    [Fact]
    public async Task HotSession_Metrics_DoNotCountAFailedTurn()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer(),
            new HotSessionRuntimeOptions(maxResidentBytes: 1, maxSessionBytes: 1));
        using var session = runtime.Create();

        await Assert.ThrowsAsync<SessionResourceBudgetExceededException>(() => session.RunTurnAsync(
            "one", new SamplingParams { Temperature = 0f, MaxNewTokens = 1 },
            SessionRevision.Initial, SessionOperationId.New(), Digest("over-budget")));

        Assert.Equal(0, session.Metrics.PromptTokens);
        Assert.Equal(0, session.Metrics.GeneratedTokens);
    }

    [Fact]
    public async Task HotSession_Metadata_SetGetRemoveRoundTrip()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();

        Assert.Null(session.Metadata.Get("user"));
        session.Metadata.Set("user", "alice");
        Assert.Equal("alice", session.Metadata.Get<string>("user"));
        Assert.True(session.Metadata.TryGet<string>("user", out var val));
        Assert.Equal("alice", val);
        Assert.True(session.Metadata.Remove("user"));
        Assert.Null(session.Metadata.Get("user"));
    }

    [Fact]
    public void HotSession_Fork_CopiesParentMetadataIntoEachBranch()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var parent = runtime.Create();
        parent.Metadata.Set("tenant", "acme");

        var branches = runtime.Fork(parent, 2);
        try
        {
            foreach (var branch in branches)
                Assert.Equal("acme", branch.Metadata.Get<string>("tenant"));
        }
        finally
        {
            foreach (var branch in branches) runtime.Delete(branch.SessionId);
        }
    }

    [Fact]
    public async Task HotSession_TurnResult_ReportsMaxTokensFinishReasonWhenBudgetExhausted()
    {
        var fwd = new FakeForwardPass { EmitNonStopOnPrefill = true, EmitNonStopOnDecode = true };
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();

        var result = await session.RunTurnAsync("one",
            new SamplingParams { Temperature = 0f, MaxNewTokens = 3 },
            SessionRevision.Initial, SessionOperationId.New(), Digest("maxtok"));

        Assert.Equal(FinishReason.MaxTokens, result.FinishReason);
        Assert.Empty(result.ToolCalls);
    }

    [Fact]
    public async Task HotSession_TurnResult_ReportsCompletedFinishReasonOnEos()
    {
        var fwd = new FakeForwardPass { EmitNonStopOnPrefill = true };
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();

        var result = await session.RunTurnAsync("one",
            new SamplingParams { Temperature = 0f, MaxNewTokens = 8 },
            SessionRevision.Initial, SessionOperationId.New(), Digest("eos"));

        Assert.Equal(FinishReason.Completed, result.FinishReason);
    }

    [Fact]
    public async Task HotSession_TurnResult_ReportsCancelledFinishReasonOnCancellation()
    {
        var fwd = new FakeForwardPass { EmitNonStopOnPrefill = true, BlockDecode = true };
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();
        using var cts = new CancellationTokenSource();

        var turn = session.RunTurnAsync("one",
            new SamplingParams { Temperature = 0f, MaxNewTokens = 4 },
            SessionRevision.Initial, SessionOperationId.New(), Digest("cancel"), cts.Token);
        await fwd.DecodeStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
        cts.Cancel();
        fwd.ReleaseDecode.Set();

        var result = await turn;
        Assert.Equal(FinishReason.Cancelled, result.FinishReason);
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

    private sealed class FakeForwardPass : IBatchedForwardPass
    {
        private static readonly float[] EosLogits = CreateLogits(Eos);
        private static readonly float[] NonStopLogits = CreateLogits(NonStopToken);
        public List<FakeCache> Created { get; } = [];
        public List<(int Start, int Count)> Prefills { get; } = [];
        public bool EmitNonStopOnPrefill { get; set; }
        public bool EmitNonStopOnDecode { get; set; }
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
