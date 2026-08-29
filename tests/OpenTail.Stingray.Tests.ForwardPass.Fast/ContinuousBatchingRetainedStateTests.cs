
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>Ownership and append-resume coverage for the retained CPU-dense lifecycle seam.</summary>
public sealed class ContinuousBatchingRetainedStateTests
{
    private const int Eos = 31;

    [Fact]
    public async Task RetainedState_ReusesOneCacheAndAppendsAtPriorMaterializedPosition()
    {
        var fwd = new RetainedFakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new SuffixTokenizer(), "test", maxBatchSize: 1);
        using var state = new RetainedSequenceState();
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        await Drain(engine.GenerateRetainedChunksAsync("one", sp, state));

        var cache = Assert.Single(fwd.Created);
        Assert.True(state.HasRetainedState);
        Assert.Equal(2, state.MaterializedPosition);
        Assert.False(cache.Disposed);
        Assert.Equal([(0, 2)], fwd.Prefills);

        await Drain(engine.GenerateRetainedChunksAsync("two", sp, state));

        Assert.Single(fwd.Created);
        Assert.True(state.HasRetainedState);
        Assert.Equal(3, state.MaterializedPosition);
        Assert.Equal([(0, 2), (2, 1)], fwd.Prefills);
        Assert.False(cache.Disposed);
    }

    [Fact]
    public async Task RetainedState_RejectsConcurrentTurns()
    {
        var fwd = new RetainedFakeForwardPass { BlockPrefill = true };
        using var engine = new ContinuousBatchingEngine(fwd, new SuffixTokenizer(), "test", maxBatchSize: 1);
        using var state = new RetainedSequenceState();
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        var first = engine.GenerateRetainedChunksAsync("one", sp, state).GetAsyncEnumerator();
        try
        {
            var firstMove = first.MoveNextAsync().AsTask();
            await fwd.PrefillStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
            var ex = Assert.Throws<InvalidOperationException>(() =>
                state.Reserve());
            Assert.Contains("queued or active", ex.Message, StringComparison.Ordinal);

            fwd.ReleasePrefill.Set();
            Assert.True(await firstMove);
        }
        finally
        {
            fwd.ReleasePrefill.Set();
            await first.DisposeAsync();
        }
    }

    [Fact]
    public async Task RetainedState_CancellationRollsBackToTurnStart()
    {
        var fwd = new RetainedFakeForwardPass { EmitNonStopOnPrefill = true, BlockDecode = true };
        using var engine = new ContinuousBatchingEngine(fwd, new SuffixTokenizer(), "test", maxBatchSize: 1);
        using var state = new RetainedSequenceState();
        using var cts = new CancellationTokenSource();
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 4 };

        var turn = Drain(engine.GenerateRetainedChunksAsync("one", sp, state, cts.Token));
        await fwd.DecodeStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
        cts.Cancel();
        fwd.ReleaseDecode.Set();

        try
        {
            await turn;
        }
        catch (OperationCanceledException) { }

        var cache = Assert.Single(fwd.Created);
        await WaitUntilAsync(() => state.HasRetainedState && state.MaterializedPosition == 0,
            TimeSpan.FromSeconds(10));
        Assert.Equal(0, cache.LogicalPosition);
        Assert.False(cache.Disposed);
    }

    [Fact]
    public async Task RetainedState_FailureDiscardsStalePositionAndCanStartFresh()
    {
        var fwd = new RetainedFakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new SuffixTokenizer(), "test", maxBatchSize: 1);
        using var state = new RetainedSequenceState();
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        await Drain(engine.GenerateRetainedChunksAsync("one", sp, state));
        var failedLease = state.Reserve();
        state.Fail(failedLease.Cache);

        Assert.False(state.HasRetainedState);
        Assert.Equal(0, state.MaterializedPosition);

        await Drain(engine.GenerateRetainedChunksAsync("two", sp, state));

        Assert.Equal(2, fwd.Created.Count);
        Assert.Equal((0, 1), fwd.Prefills[^1]);
    }

    private static async Task Drain(IAsyncEnumerable<GenerateChunk> chunks)
    {
        await foreach (var _ in chunks) { }
    }

    private static async Task WaitUntilAsync(Func<bool> condition, TimeSpan timeout)
    {
        var until = DateTime.UtcNow + timeout;
        while (!condition())
        {
            if (DateTime.UtcNow >= until)
                throw new TimeoutException("Condition did not become true before the test timeout.");
            await Task.Delay(10);
        }
    }

    private sealed class SuffixTokenizer : ITokenizer
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

    private sealed class RetainedFakeForwardPass : IBatchedForwardPass
    {
        private readonly float[] _eosLogits = CreateLogits(Eos);
        private readonly float[] _nonStopLogits = CreateLogits(7);
        public List<RetainedFakeCache> Created { get; } = [];
        public List<(int Start, int Count)> Prefills { get; } = [];
        public bool BlockPrefill { get; init; }
        public bool EmitNonStopOnPrefill { get; init; }
        public bool BlockDecode { get; init; }
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
            var cache = new RetainedFakeCache();
            Created.Add(cache);
            return cache;
        }

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
        {
            var retained = Assert.IsType<RetainedFakeCache>(cache);
            Assert.Equal(startPos, retained.LogicalPosition);
            Prefills.Add((startPos, tokens.Count));
            if (BlockPrefill)
            {
                PrefillStarted.TrySetResult();
                if (!ReleasePrefill.Wait(TimeSpan.FromSeconds(10)))
                    throw new TimeoutException("The test did not release the blocked prefill.");
            }
            retained.LogicalPosition += tokens.Count;
            return EmitNonStopOnPrefill ? _nonStopLogits : _eosLogits;
        }

        public float[]?[] PrefillPackedMulti(
            ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits) =>
            throw new NotSupportedException("This test uses one request at a time.");

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            for (int i = 0; i < caches.Length; i++)
            {
                var retained = Assert.IsType<RetainedFakeCache>(caches[i]);
                Assert.Equal(positions[i], retained.LogicalPosition);
                retained.LogicalPosition++;
            }
            if (BlockDecode)
            {
                DecodeStarted.TrySetResult();
                if (!ReleaseDecode.Wait(TimeSpan.FromSeconds(10)))
                    throw new TimeoutException("The test did not release the blocked decode.");
            }
            return Enumerable.Repeat(_eosLogits, tokens.Length).ToArray();
        }

        private static float[] CreateLogits(int token)
        {
            var logits = new float[64];
            logits[token] = 1f;
            return logits;
        }
    }

    private sealed class RetainedFakeCache : IRewindableSequenceKvCache
    {
        public int LogicalPosition { get; set; }
        public bool Disposed { get; private set; }
        public bool CanRewindTo(int logicalPosition) => !Disposed && logicalPosition >= 0 && logicalPosition <= LogicalPosition;

        public void RewindTo(int logicalPosition)
        {
            if (!CanRewindTo(logicalPosition)) throw new InvalidOperationException();
            LogicalPosition = logicalPosition;
        }

        public void Dispose() => Disposed = true;
    }
}
