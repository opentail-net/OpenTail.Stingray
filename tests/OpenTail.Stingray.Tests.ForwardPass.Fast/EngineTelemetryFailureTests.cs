using System.Diagnostics;
using System.Diagnostics.Metrics;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// OpenTelemetry plan acceptance item: "a telemetry or listener failure cannot break inference". .NET runs <c>MeterListener</c> and
/// <c>ActivityListener</c> callbacks synchronously inside the instrumented call, so a listener that throws would otherwise throw out
/// of the engine (or, worse, out of the batcher loop, which ends every in-flight request). These tests install listeners that throw
/// on every event and require the generated output to be exactly what an unobserved run produces. Each also counts how often the
/// broken listener fired, so a pass cannot be vacuous.
/// </summary>
[Collection("EngineTelemetry")]
public sealed class EngineTelemetryFailureTests
{
    private const int TokHi = 0, TokThere = 1, TokEos = 6;

    private sealed class FiredCounter { public int Metrics; public int Activities; }

    private static MeterListener ThrowingMeterListener(FiredCounter fired)
    {
        var l = new MeterListener();
        l.InstrumentPublished = (i, ml) =>
        {
            if (i.Meter.Name == EngineTelemetry.MeterName) ml.EnableMeasurementEvents(i);
        };
        l.SetMeasurementEventCallback<long>((i, v, t, s) => { Interlocked.Increment(ref fired.Metrics); throw new InvalidOperationException("metrics listener failure"); });
        l.SetMeasurementEventCallback<double>((i, v, t, s) => { Interlocked.Increment(ref fired.Metrics); throw new InvalidOperationException("metrics listener failure"); });
        l.SetMeasurementEventCallback<int>((i, v, t, s) => { Interlocked.Increment(ref fired.Metrics); throw new InvalidOperationException("metrics listener failure"); });
        l.Start();
        return l;
    }

    private static ActivityListener ThrowingActivityListener(FiredCounter fired, string extraSource = "")
    {
        var al = new ActivityListener
        {
            ShouldListenTo = s => s.Name == EngineTelemetry.ActivitySourceName || s.Name == extraSource,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllData,
            ActivityStarted = _ => { Interlocked.Increment(ref fired.Activities); throw new InvalidOperationException("activity listener failure"); },
            ActivityStopped = _ => { Interlocked.Increment(ref fired.Activities); throw new InvalidOperationException("activity listener failure"); },
        };
        ActivitySource.AddActivityListener(al);
        return al;
    }

    private static InferenceEngine NewEngine()
    {
        var tokenizer = new InferenceEngineChunkTests.ScriptedTokenizer();
        var fwd = new InferenceEngineChunkTests.ScriptedForwardPass([TokHi, TokThere, TokEos], tokenizer.VocabSize);
        return new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);
    }

    private static ContinuousBatchingEngine NewBatchingEngine() => new(
        new ContinuousBatchingConstraintTests.FakeBatchedForwardPass(),
        new ContinuousBatchingConstraintTests.CharTokenizer(), "test", maxBatchSize: 1);

    private static async Task<(string Text, GenerateChunkKind Last)> Collect(IAsyncEnumerable<GenerateChunk> chunks)
    {
        var text = new System.Text.StringBuilder();
        var last = default(GenerateChunkKind);
        await foreach (var c in chunks) { text.Append(c.Text); last = c.Kind; }
        return (text.ToString(), last);
    }

    private static readonly SamplingParams Greedy = new() { Temperature = 0f, MaxNewTokens = 10 };

    [Fact]
    public async Task InferenceEngine_ThrowingMeterListener_DoesNotChangeTheOutput()
    {
        using var reference = NewEngine();
        var expected = await Collect(reference.GenerateChunksAsync("seed", Greedy));

        var fired = new FiredCounter();
        using var listener = ThrowingMeterListener(fired);
        using var engine = NewEngine();
        var actual = await Collect(engine.GenerateChunksAsync("seed", Greedy));

        Assert.True(fired.Metrics > 0, "the failing listener never fired, so this proves nothing");
        Assert.Equal(expected, actual);
        Assert.Equal(GenerateChunkKind.Stop, actual.Last);
    }

    [Fact]
    public async Task InferenceEngine_ThrowingActivityListener_DoesNotChangeTheOutput()
    {
        using var reference = NewEngine();
        var expected = await Collect(reference.GenerateChunksAsync("seed", Greedy));

        var fired = new FiredCounter();
        using var listener = ThrowingActivityListener(fired);
        using var engine = NewEngine();
        var actual = await Collect(engine.GenerateChunksAsync("seed", Greedy));

        Assert.True(fired.Activities > 0, "the failing listener never fired, so this proves nothing");
        Assert.Equal(expected, actual);
    }

    [Fact]
    public async Task BatchingEngine_ThrowingListeners_DoNotKillTheBatcher_AndLaterRequestsStillWork()
    {
        using var reference = NewBatchingEngine();
        var expected = await Collect(reference.GenerateChunksAsync("prompt", new SamplingParams { Temperature = 0f, MaxNewTokens = 6 }));

        var fired = new FiredCounter();
        using var meters = ThrowingMeterListener(fired);
        using var activities = ThrowingActivityListener(fired);
        using var engine = NewBatchingEngine();

        // Two requests in a row: if the first one's telemetry had killed the batcher loop, the second would hang or fail.
        for (int i = 0; i < 2; i++)
        {
            var actual = await Collect(engine.GenerateChunksAsync("prompt", new SamplingParams { Temperature = 0f, MaxNewTokens = 6 }))
                .WaitAsync(TimeSpan.FromSeconds(30));
            Assert.Equal(expected, actual);
        }

        Assert.True(fired.Metrics > 0 && fired.Activities > 0, $"listeners fired metrics={fired.Metrics} activities={fired.Activities}");
    }

    [Fact]
    public async Task Concurrent_ThrowingListeners_EveryRequestStillCompletes()
    {
        var fired = new FiredCounter();
        using var meters = ThrowingMeterListener(fired);
        using var activities = ThrowingActivityListener(fired);
        using var engine = NewEngine();

        var results = await Task.WhenAll(Enumerable.Range(0, 4).Select(_ => Collect(engine.GenerateChunksAsync("seed", Greedy))))
            .WaitAsync(TimeSpan.FromSeconds(30));

        Assert.All(results, r => Assert.Equal(GenerateChunkKind.Stop, r.Last));
        Assert.Single(results.Select(r => r.Text).Distinct());
        Assert.True(fired.Metrics > 0);
    }

    [Fact]
    public void SelfTestSpan_ThrowingActivityListener_DoesNotThrow()
    {
        var fired = new FiredCounter();
        using var activities = ThrowingActivityListener(fired);

        var ex = Record.Exception(EngineTelemetry.RunSelfTestSpan);

        Assert.Null(ex);
        Assert.True(fired.Activities > 0, "the failing listener never fired, so this proves nothing");
    }

    [Fact]
    public void ModelLoadRecording_ThrowingListeners_DoNotThrow()
    {
        var fired = new FiredCounter();
        using var meters = ThrowingMeterListener(fired);
        using var activities = ThrowingActivityListener(fired);

        var ex = Record.Exception(() =>
        {
            using var load = EngineTelemetry.StartModelLoad();
            EngineTelemetry.RecordModelLoad(load.Activity, "llama", "ForwardPass", EngineTelemetry.OutcomeSuccess, 1.5);
        });

        Assert.Null(ex);
        Assert.True(fired.Metrics > 0, "the failing listener never fired, so this proves nothing");
    }
}
