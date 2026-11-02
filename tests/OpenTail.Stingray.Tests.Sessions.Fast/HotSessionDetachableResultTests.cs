
namespace OpenTail.Stingray.Tests.Sessions.Fast;

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
        /// <summary>Runs inside every decode step, on the engine's batcher thread: a test cancels from here so the cancellation provably lands while the turn is generating.</summary>
        public Action? OnDecodeStep { get; set; }

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
            OnDecodeStep?.Invoke();
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

    /// <summary>
    /// G6 (plan 6): a queued turn whose token is cancelled while the batcher is admitting it. The batcher saw the cancellation, completed the request's output
    /// channel with no chunk and no exception, and only then published the rolled-back turn; the caller, parked on the channel, woke to a NORMAL end and found no
    /// retained turn outcome, so the operation was recorded as Failed ("The retained engine completed without a retained turn outcome") instead of Cancelled.
    /// It showed up as about 1 failure in 4,000 randomly timed cancels and once on the Windows CI runner.
    /// A token runs its callbacks one after another after it is marked cancelled, and the caller's wake-up is one of them; holding callbacks registered from many
    /// live threads (one lands in every callback partition) keep <c>Cancel()</c> inside that window so the batcher gets there first. If the callback order ever
    /// changed, the test would pass without exercising the race: it can under-detect, never false-fail.
    /// </summary>
    [Fact]
    public async Task HotSession_QueuedTurnCancelledWhileTheBatcherAdmitsIt_IsRecordedAsCancelled()
    {
        using var ctsB = new CancellationTokenSource();
        using var inDecode = new ManualResetEventSlim(false);
        using var proceed = new ManualResetEventSlim(false);
        using var cancelCallbackRunning = new ManualResetEventSlim(false);
        using var releaseCancel = new ManualResetEventSlim(false);
        using var holdersDone = new ManualResetEventSlim(false);
        var holders = new List<Thread>();
        var registrations = new List<CancellationTokenRegistration>();
        var bound = TimeSpan.FromSeconds(30);   // nothing here may wait forever: an assertion that fails early must not leave the batcher blocked in the decode hook
        int steps = 0;
        var fwd = new FakeForwardPass
        {
            EmitNonStopOnPrefill = true,
            OnDecodeStep = () => { if (Interlocked.Increment(ref steps) == 1) { inDecode.Set(); proceed.Wait(bound); } },
        };
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);   // one slot: B queues behind A
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var sessionA = runtime.Create();
        using var sessionB = runtime.Create();
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 2 };
        var operationA = SessionOperationId.New();
        var operationB = SessionOperationId.New();

        try
        {
        var turnA = Task.Run(() => sessionA.RunTurnAsync("hello", sampling, SessionRevision.Initial, operationA, Digest("hello")));
        Assert.True(inDecode.Wait(TimeSpan.FromSeconds(10)), "turn A never reached its first decode step");
        var turnB = Task.Run(async () =>
        {
            try { await sessionB.RunTurnAsync("hello", sampling, SessionRevision.Initial, operationB, Digest("hello"), ctsB.Token); }
            catch (OperationCanceledException) { }
        });
        await Task.Delay(200);   // B is queued behind A and its caller is parked on its output channel

        for (int i = 0; i < 64; i++)
        {
            var holder = new Thread(() =>
            {
                var registration = ctsB.Token.Register(() => { cancelCallbackRunning.Set(); releaseCancel.Wait(bound); });
                lock (registrations) registrations.Add(registration);
                holdersDone.Wait(bound);
            }) { IsBackground = true };
            holders.Add(holder);
            holder.Start();
        }
        SpinWait.SpinUntil(() => { lock (registrations) return registrations.Count == holders.Count; }, TimeSpan.FromSeconds(10));
        // A dedicated thread, not Task.Run: Cancel() blocks inside the holding callback, and on a pool that starts with two threads (a small CI runner,
        // or DOTNET_PROCESSOR_COUNT=2) the batcher, the blocked cancel and the continuations of both turns compete for them, so the cancel could wait for
        // thread injection (about 500 ms each) and "Cancel() never reached the holding callback" failed 2 runs in 40.
        var canceller = new Thread(() => ctsB.Cancel()) { IsBackground = true };
        canceller.Start();
        Assert.True(cancelCallbackRunning.Wait(TimeSpan.FromSeconds(10)), "Cancel() never reached the holding callback");
        Assert.True(ctsB.IsCancellationRequested);

        proceed.Set();   // A finishes, the slot frees, and the batcher admits (and finds cancelled) B while B's caller has not been woken by its token
        await turnA.WaitAsync(TimeSpan.FromSeconds(10));
        await turnB.WaitAsync(TimeSpan.FromSeconds(10));
        releaseCancel.Set();
        Assert.True(canceller.Join(bound), "the cancelling thread never returned from Cancel()");
        holdersDone.Set();
        foreach (var holder in holders) holder.Join();
        foreach (var registration in registrations) registration.Dispose();

        var retrieved = runtime.GetOperation(sessionB.SessionId, operationB);
        Assert.Equal(SessionOperationState.Cancelled, retrieved.State);
        Assert.Null(retrieved.CommittedRevision);
        }
        finally
        {
            // On any early exit the batcher may still be blocked in the decode hook and the holders in their callbacks: release them, or disposing the
            // sessions (which waits for the batcher) hangs the whole test run instead of failing this one test.
            proceed.Set();
            releaseCancel.Set();
            holdersDone.Set();
        }
    }

    [Fact]
    public async Task HotSession_GetOperation_ReturnsCancelledOperation_OnCancellation()
    {
        using var cts = new CancellationTokenSource();
        // Cancel from inside the first decode step. A timer (CancelAfter(5)) raced the start of the turn: on a slow or cold run it fired before the operation
        // existed (GetOperation threw KeyNotFoundException, 41 of 200 runs locally) or at a point the Windows CI run recorded as Failed.
        var fwd = new FakeForwardPass { EmitNonStopOnPrefill = true, DecodeDelay = TimeSpan.FromMilliseconds(50), OnDecodeStep = () => cts.Cancel() };
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();

        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };
        var operationId = SessionOperationId.New();

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
