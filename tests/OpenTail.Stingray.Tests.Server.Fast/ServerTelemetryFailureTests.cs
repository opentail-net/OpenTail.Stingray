using System.Diagnostics;
using System.Diagnostics.Metrics;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// OpenTelemetry plan acceptance item: "a telemetry or listener failure cannot break inference", for the server's own metrics and
/// spans. Listeners run synchronously inside the instrumented call, so one that throws must not fail the request it observes. The
/// listeners here throw on every event and count how often they fired, so a pass cannot be vacuous.
/// </summary>
// Its listeners are process-wide and THROW. A throwing meter callback stops the later listeners from receiving that measurement, and a throwing
// ActivityStarted makes the server serve the request without a span, so any telemetry test running at the same time loses its measurement or its span (run
// together with ServingRequestTimingTests it failed 19 times in 20). The non-parallel collection runs it on its own, after the parallel ones.
[Collection(ModelLoadWaitTelemetryCollection.Name)]
public sealed class ServerTelemetryFailureTests
{
    private sealed class Fired { public int Metrics; public int Activities; }

    private static MeterListener ThrowingMeterListener(Fired fired)
    {
        var l = new MeterListener();
        l.InstrumentPublished = (i, ml) => { if (i.Meter.Name == "OpenTail.Stingray.Server") ml.EnableMeasurementEvents(i); };
        l.SetMeasurementEventCallback<long>((i, v, t, s) => { Interlocked.Increment(ref fired.Metrics); throw new InvalidOperationException("metrics listener failure"); });
        l.SetMeasurementEventCallback<double>((i, v, t, s) => { Interlocked.Increment(ref fired.Metrics); throw new InvalidOperationException("metrics listener failure"); });
        l.SetMeasurementEventCallback<int>((i, v, t, s) => { Interlocked.Increment(ref fired.Metrics); throw new InvalidOperationException("metrics listener failure"); });
        l.Start();
        return l;
    }

    private static ActivityListener ThrowingActivityListener(Fired fired)
    {
        var al = new ActivityListener
        {
            ShouldListenTo = s => s.Name == ServerMetrics.ActivitySourceName,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllData,
            ActivityStarted = _ => { Interlocked.Increment(ref fired.Activities); throw new InvalidOperationException("activity listener failure"); },
            ActivityStopped = _ => { Interlocked.Increment(ref fired.Activities); throw new InvalidOperationException("activity listener failure"); },
        };
        ActivitySource.AddActivityListener(al);
        return al;
    }

    [Fact]
    public void ServingRequest_ThrowingListeners_DoNotBreakTheRequestPath()
    {
        var fired = new Fired();
        using var meters = ThrowingMeterListener(fired);
        using var activities = ThrowingActivityListener(fired);
        var metrics = new ServerMetrics();

        var ex = Record.Exception(() =>
        {
            var timing = metrics.BeginServingRequest("test.listener_failure", "test-model");
            timing.MarkAdmitted();
            timing.MarkFirstToken();
            timing.MarkGenerationComplete();
            timing.SetOutputTokens(5);
            timing.Dispose();
        });

        Assert.Null(ex);
        Assert.True(fired.Metrics > 0, "the failing metrics listener never fired, so this proves nothing");
        Assert.True(fired.Activities > 0, "the failing activity listener never fired, so this proves nothing");
    }

    [Fact]
    public void OtherServerRecording_ThrowingListeners_DoNotThrow()
    {
        var fired = new Fired();
        using var meters = ThrowingMeterListener(fired);
        using var activities = ThrowingActivityListener(fired);
        var metrics = new ServerMetrics();

        var ex = Record.Exception(() =>
        {
            metrics.RecordOperation("tts", "/v1/audio/speech", TimeSpan.FromMilliseconds(5), "success");
            metrics.RecordAudio("tts", "/v1/audio/speech", 2.0, TimeSpan.FromMilliseconds(500));
            ServerMetrics.RecordLoadWait(TimeSpan.FromMilliseconds(5), "success", default, default, DateTimeOffset.UtcNow);
        });

        Assert.Null(ex);
        Assert.True(fired.Metrics > 0, "the failing metrics listener never fired, so this proves nothing");
    }
}
