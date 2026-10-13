using System.Diagnostics;
using System.Diagnostics.Metrics;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Requests that wait for a shared model load record the wait separately and carry a span link to the load, instead of the
/// load becoming the child of one arbitrary request (plan section 5, "model load is its own operation").
/// </summary>
public sealed class ModelLoadWaitTelemetryTests
{
    private sealed class FakeEngine(string id) : IInferenceEngine
    {
        public string ModelId { get; } = id;
        public int QueueDepth => 0;
        public int ActiveRequests => 0;
        public bool PrefixCacheEnabled => false;
        public long PrefillTokensReused => 0;
        public async IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(string prompt, SamplingParams sp,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default, string? canonicalHistoryPrefix = null)
        {
            await Task.CompletedTask;
            yield break;
        }
        public async IAsyncEnumerable<string> GenerateAsync(string prompt, SamplingParams sp,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
        {
            await Task.CompletedTask;
            yield break;
        }
    }

    [Fact]
    public async Task SharedLoadWaiters_EachRecordTheWait_AndLinkToTheLoadSpan_WhichIsNotTheirChild()
    {
        var acquireSpans = new List<Activity>();
        using var al = new ActivityListener
        {
            ShouldListenTo = s => s.Name == ServerMetrics.ActivitySourceName,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllData,
            ActivityStopped = a => { if (a.OperationName == "model.acquire") lock (acquireSpans) acquireSpans.Add(a); },
        };
        ActivitySource.AddActivityListener(al);
        int waits = 0;
        using var ml = new MeterListener();
        ml.InstrumentPublished = (i, l) => { if (i.Name == "opentail.models.load.wait") l.EnableMeasurementEvents(i); };
        ml.SetMeasurementEventCallback<double>((i, v, t, s) => Interlocked.Increment(ref waits));
        ml.Start();

        // The "engine" side of the load: publishes a load-span context through the holder, as EngineTelemetry.StartModelLoad does.
        var loadContext = new ActivityContext(ActivityTraceId.CreateRandom(), ActivitySpanId.CreateRandom(), ActivityTraceFlags.Recorded);
        int loads = 0;
        var gate = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var manager = new ModelRuntimeManager(id =>
        {
            Interlocked.Increment(ref loads);
            EngineTelemetry.LoadHolder.Value!.Context = loadContext;
            gate.Task.GetAwaiter().GetResult();
            return new LoadedEngine(new FakeEngine(id.Value), "qwen2", null);
        });

        var callers = Enumerable.Range(0, 3).Select(_ => manager.AcquireAsync(new ModelId("shared")).AsTask()).ToArray();
        await Task.Delay(75);
        gate.SetResult();
        var handles = await Task.WhenAll(callers);
        foreach (var h in handles) h.Dispose();
        al.Dispose();
        ml.Dispose();

        Assert.Equal(1, loads);                       // one physical load
        Assert.Equal(3, waits);                       // three recorded waits
        lock (acquireSpans)
        {
            Assert.Equal(3, acquireSpans.Count);
            Assert.All(acquireSpans, s =>
            {
                var link = Assert.Single(s.Links);
                Assert.Equal(loadContext.TraceId, link.Context.TraceId);
                Assert.Equal(loadContext.SpanId, link.Context.SpanId);
                Assert.NotEqual(loadContext.TraceId, s.TraceId);        // the load is not part of the waiter's trace
                Assert.Equal("success", s.GetTagItem("outcome"));
            });
        }
    }
}
