
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// <see cref="ContinuousBatchingEngine"/> with an <see cref="IExecutionTurnGate"/> (plan 032, Phase 6, slice 5). The gate here is
/// manual: the test grants each turn itself, so what the engine does inside a turn is exact rather than a race between threads.
/// Pinned: all forward work happens inside a granted turn, a turn contains at most one decode step, the turn is given back before
/// the engine asks for the next one, and disposing a queued engine does not hang.
/// </summary>
public sealed class ContinuousBatchingTurnGateTests
{
    private const int Eos = 31;
    private const int Busy = 7;

    [Fact]
    public async Task TwoEngines_ForwardWorkOnlyInsideATurn_AndAtMostOneDecodeStepPerTurn()
    {
        var gate = new ManualGate();
        var fwdA = new GateFakeForwardPass(() => gate.IsHeld("A"));
        var fwdB = new GateFakeForwardPass(() => gate.IsHeld("B"));
        using var a = new ContinuousBatchingEngine(fwdA, new GateFakeTokenizer(), "A", maxBatchSize: 1, turnGate: gate);
        using var b = new ContinuousBatchingEngine(fwdB, new GateFakeTokenizer(), "B", maxBatchSize: 1, turnGate: gate);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 12 };

        var runA = Drain(a.GenerateChunksAsync("go", sp));
        var runB = Drain(b.GenerateChunksAsync("go", sp));

        // Nothing may run before the first grant: the engines are queued, not executing.
        await gate.WaitPending("A");
        await gate.WaitPending("B");
        Assert.Equal(0, fwdA.Calls + fwdB.Calls);

        int turns = 0;
        var deadline = DateTime.UtcNow.AddSeconds(30);
        while (!(runA.IsCompleted && runB.IsCompleted))
        {
            Assert.True(DateTime.UtcNow < deadline, $"engines did not finish within 30 s after {turns} turns");
            foreach (var (name, fwd) in new[] { ("A", fwdA), ("B", fwdB) })
            {
                if (!gate.HasPending(name)) continue;
                int decodesBefore = fwd.DecodeCalls;
                gate.Grant(name);
                await gate.WaitReleased(name);                  // the engine gave the turn back (before asking for another)
                Assert.True(fwd.DecodeCalls - decodesBefore <= 1, $"{name} ran {fwd.DecodeCalls - decodesBefore} decode steps in one turn");
                turns++;
            }
            Assert.True(turns < 200, "engines never finished");
            await Task.Delay(1);   // let a finishing engine reach its idle wait or its next request for a turn
        }
        await Task.WhenAll(runA, runB).WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(0, fwdA.OutsideTurn + fwdB.OutsideTurn);   // no forward call without holding the turn
        Assert.True(fwdA.DecodeCalls >= 10 && fwdB.DecodeCalls >= 10, $"decodes A={fwdA.DecodeCalls} B={fwdB.DecodeCalls}");
        Assert.Equal(0, gate.DoubleHolds);                       // never asked for a second turn while holding one
    }

    [Fact]
    public async Task AnEngineWithoutAGate_RunsExactlyAsBefore()
    {
        var fwd = new GateFakeForwardPass(() => true);
        using var engine = new ContinuousBatchingEngine(fwd, new GateFakeTokenizer(), "plain", maxBatchSize: 1);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 6 };

        await Drain(engine.GenerateChunksAsync("go", sp)).WaitAsync(TimeSpan.FromSeconds(30));

        Assert.True(fwd.DecodeCalls >= 5);
    }

    [Fact]
    public async Task DisposingAnEngineQueuedForATurn_DoesNotHang()
    {
        var gate = new FairExecutionGate();
        var holder = await gate.AcquireTurnAsync("someone else's long turn");
        var engine = new ContinuousBatchingEngine(new GateFakeForwardPass(() => true), new GateFakeTokenizer(), "queued",
            maxBatchSize: 1, turnGate: gate);
        var run = Drain(engine.GenerateChunksAsync("go", new SamplingParams { Temperature = 0f, MaxNewTokens = 6 }));
        await WaitUntil(() => gate.GetStats().Waiting == 1);

        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        engine.Dispose();
        var elapsed = System.Diagnostics.Stopwatch.GetElapsedTime(started);

        Assert.True(engine.DrainedOnDispose);
        Assert.True(elapsed < TimeSpan.FromSeconds(10), $"dispose took {elapsed} while queued behind a held turn");
        await run.WaitAsync(TimeSpan.FromSeconds(10));   // the caller is released, not left waiting
        Assert.Equal(0, gate.GetStats().Waiting);

        holder.Dispose();
        Assert.True(gate.AcquireTurnAsync("after").IsCompletedSuccessfully);   // nothing leaked the turn
    }

    private static async Task Drain(IAsyncEnumerable<GenerateChunk> chunks)
    {
        await foreach (var _ in chunks) { }
    }

    private static async Task WaitUntil(Func<bool> condition)
    {
        var deadline = DateTime.UtcNow.AddSeconds(10);
        while (!condition())
        {
            if (DateTime.UtcNow > deadline) throw new TimeoutException("condition not reached");
            await Task.Delay(1);
        }
    }

    /// <summary>A gate whose grants the test makes by hand.</summary>
    private sealed class ManualGate : IExecutionTurnGate
    {
        private readonly object _lock = new();
        private readonly Dictionary<string, TaskCompletionSource<IDisposable>> _pending = [];
        private readonly HashSet<string> _held = [];
        private int _doubleHolds;

        public int DoubleHolds => Volatile.Read(ref _doubleHolds);

        public ValueTask<IDisposable> AcquireTurnAsync(string participant, CancellationToken cancellationToken = default)
        {
            var tcs = new TaskCompletionSource<IDisposable>(TaskCreationOptions.RunContinuationsAsynchronously);
            lock (_lock)
            {
                if (_held.Contains(participant)) Interlocked.Increment(ref _doubleHolds);
                _pending[participant] = tcs;
            }
            cancellationToken.Register(() => tcs.TrySetCanceled());
            return new ValueTask<IDisposable>(tcs.Task);
        }

        public bool HasPending(string participant) { lock (_lock) return _pending.ContainsKey(participant); }
        public bool IsHeld(string participant) { lock (_lock) return _held.Contains(participant); }

        public void Grant(string participant)
        {
            TaskCompletionSource<IDisposable> tcs;
            lock (_lock)
            {
                tcs = _pending[participant];
                _pending.Remove(participant);
                _held.Add(participant);
            }
            tcs.SetResult(new Lease(this, participant));
        }

        public Task WaitPending(string participant) => WaitUntil(() => HasPending(participant));
        public Task WaitReleased(string participant) => WaitUntil(() => !IsHeld(participant));

        private sealed class Lease(ManualGate gate, string participant) : IDisposable
        {
            public void Dispose() { lock (gate._lock) gate._held.Remove(participant); }
        }
    }

    private sealed class GateFakeForwardPass(Func<bool> inTurn) : IBatchedForwardPass
    {
        private readonly float[] _busyLogits = Logits(Busy);
        private int _calls, _decodeCalls, _outsideTurn;

        public int Calls => Volatile.Read(ref _calls);
        public int DecodeCalls => Volatile.Read(ref _decodeCalls);
        public int OutsideTurn => Volatile.Read(ref _outsideTurn);

        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 1;
        public int MaxSeqLen => 64;
        public bool PrefillDequantCacheActive => false;

        public ISequenceKvCache CreateCache() => new GateFakeCache();

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
        {
            Note();
            ((GateFakeCache)cache).LogicalPosition += tokens.Count;
            return _busyLogits;
        }

        public float[]?[] PrefillPackedMulti(ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits) =>
            throw new NotSupportedException("One request per engine in these tests.");

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            Note();
            Interlocked.Increment(ref _decodeCalls);
            foreach (var cache in caches) ((GateFakeCache)cache).LogicalPosition++;
            return Enumerable.Repeat(_busyLogits, tokens.Length).ToArray();
        }

        private void Note()
        {
            Interlocked.Increment(ref _calls);
            if (!inTurn()) Interlocked.Increment(ref _outsideTurn);
        }

        private static float[] Logits(int token)
        {
            var logits = new float[64];
            logits[token] = 1f;
            return logits;
        }
    }

    private sealed class GateFakeCache : IRewindableSequenceKvCache
    {
        public int LogicalPosition { get; set; }
        public bool CanRewindTo(int logicalPosition) => logicalPosition >= 0 && logicalPosition <= LogicalPosition;
        public void RewindTo(int logicalPosition) => LogicalPosition = logicalPosition;
        public void Dispose() { }
    }

    private sealed class GateFakeTokenizer : ITokenizer
    {
        public int VocabSize => 64;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;
        public IReadOnlyCollection<int> EogTokenIds => [Eos];
        public IReadOnlyList<int> Encode(string text) => [1, 2];
        public string Decode(IEnumerable<int> tokens) => string.Empty;
        public byte[] DecodeBytes(int token) => [];
    }
}
