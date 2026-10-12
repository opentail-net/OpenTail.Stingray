
namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// <see cref="ServingRequestTiming"/> in isolation. The endpoint-level tests
/// (<see cref="ResponseTimingExtensionTests"/>, <see cref="StatusDocumentTests"/>) only see values
/// well under a millisecond from an in-memory fake engine, so they cannot distinguish "generation
/// duration excludes delivery time" from "generation duration happens to be small" — these tests
/// use real, deliberate sleeps to make that distinction unambiguous.
/// </summary>
public sealed class ServingRequestTimingTests
{
    /// <summary>
    /// Regression test: <c>Complete()</c>/<c>Dispose()</c> fires from a <c>using</c> block at
    /// endpoint-method exit, which on the non-streaming path is after the full response has been
    /// serialized and written to the client. Recording generation duration at that point — as the
    /// code did before <see cref="ServingRequestTiming.MarkGenerationComplete"/> existed — means a
    /// slow client or a large response inflates the reported decode/inter-token rate with delivery
    /// time that has nothing to do with the engine. This pins that calling
    /// <see cref="ServingRequestTiming.MarkGenerationComplete"/> right after the token loop finishes
    /// keeps that delivery time out of the sample.
    /// </summary>
    [Fact]
    public void MarkGenerationComplete_ExcludesTimeSpentAfterItFromGenerationDuration()
    {
        var metrics = new ServerMetrics();
        var timing = metrics.BeginServingRequest();

        timing.MarkFirstToken();
        Thread.Sleep(30); // stands in for real decode time between tokens
        timing.MarkGenerationComplete();

        // Stands in for slow response serialization/delivery to the client — must NOT be counted
        // as generation time. Comfortably larger than the 30ms above and any scheduling jitter.
        Thread.Sleep(200);
        timing.Dispose();

        var generation = metrics.GenerationDurationSummary;
        Assert.Equal(1, generation.Count);
        // A hard ceiling well below the 200ms "delivery" sleep: if delivery time leaked in, this
        // would be >= 200ms. Generous above the 30ms "decode" sleep to absorb CI scheduling jitter.
        Assert.True(generation.TotalSeconds < 0.15,
            $"generation duration was {generation.TotalSeconds * 1000:F0} ms; " +
            "post-MarkGenerationComplete delivery time leaked into the sample.");

        // Request duration is the OTHER histogram and is documented to include delivery — it must
        // still see the full ~230ms, or this test would not be distinguishing the two at all.
        var request = metrics.RequestDurationSummary;
        Assert.True(request.TotalSeconds >= 0.2,
            $"request duration was {request.TotalSeconds * 1000:F0} ms; expected it to include the delivery sleep.");
    }

    /// <summary>
    /// Calling <see cref="ServingRequestTiming.MarkGenerationComplete"/> a second time (e.g. a
    /// call site that both finishes its loop AND later re-enters an error path) must not record a
    /// second, larger sample — the first call wins.
    /// </summary>
    [Fact]
    public void MarkGenerationComplete_SecondCallIsANoOp()
    {
        var metrics = new ServerMetrics();
        var timing = metrics.BeginServingRequest();

        timing.MarkFirstToken();
        timing.MarkGenerationComplete();
        Thread.Sleep(50);
        timing.MarkGenerationComplete(); // must be ignored
        timing.Dispose();

        Assert.Equal(1, metrics.GenerationDurationSummary.Count);
    }

    /// <summary>
    /// A request that is cancelled or throws before the generation loop finishes never calls
    /// <see cref="ServingRequestTiming.MarkGenerationComplete"/>. <c>Complete()</c>'s fallback must
    /// still record a sample (using dispose time, same as the pre-fix behavior) so those requests
    /// don't silently vanish from the histogram.
    /// </summary>
    [Fact]
    public void Complete_FallsBackToDisposeTimeWhenGenerationWasNeverMarkedComplete()
    {
        var metrics = new ServerMetrics();
        var timing = metrics.BeginServingRequest();

        timing.MarkFirstToken();
        timing.Dispose(); // no MarkGenerationComplete() call — simulates an error/cancellation path

        Assert.Equal(1, metrics.GenerationDurationSummary.Count);
    }

    /// <summary>A request with no tokens at all (error before the first token) records neither histogram entry.</summary>
    [Fact]
    public void NoFirstToken_RecordsNoGenerationDurationSample()
    {
        var metrics = new ServerMetrics();
        var timing = metrics.BeginServingRequest();

        timing.Dispose();

        Assert.Equal(0, metrics.GenerationDurationSummary.Count);
        Assert.Equal(1, metrics.RequestDurationSummary.Count);
    }

    /// <summary>
    /// The GenAI-named instruments: seconds, closed attribute set, and time per output token is
    /// decode time over (tokens - 1) with delivery time excluded. The Meter is process-wide, so the
    /// test looks only at its own distinctive endpoint tag.
    /// </summary>
    [Fact]
    public void GenAiServerMetrics_AreRecordedWithTimePerOutputToken()
    {
        const string endpoint = "test.genai_metrics";
        var samples = new List<(string Name, string Unit, double Value, Dictionary<string, object?> Tags)>();
        using var listener = new System.Diagnostics.Metrics.MeterListener();
        listener.InstrumentPublished = (i, l) => { if (i.Meter.Name == "OpenTail.Stingray.Server") l.EnableMeasurementEvents(i); };
        listener.SetMeasurementEventCallback<double>((i, v, t, s) =>
        {
            var tags = new Dictionary<string, object?>();
            foreach (var kv in t) tags[kv.Key] = kv.Value;
            if (tags.TryGetValue("opentail.endpoint", out var e) && (string?)e == endpoint)
                lock (samples) samples.Add((i.Name, i.Unit ?? "", v, tags));
        });
        listener.Start();

        var metrics = new ServerMetrics();
        var timing = metrics.BeginServingRequest(endpoint, "test-model");
        timing.MarkFirstToken();
        Thread.Sleep(100);              // decode time
        timing.MarkGenerationComplete();
        timing.SetOutputTokens(11);     // 10 intervals after the first token
        Thread.Sleep(300);              // delivery, must not count towards time per token
        timing.Dispose();
        listener.Dispose();

        lock (samples)
        {
            Assert.All(samples, s => Assert.Equal("s", s.Unit));
            Assert.All(samples, s => Assert.Equal(
                s.Name == "gen_ai.server.request.duration"
                    ? new[] { "gen_ai.operation.name", "gen_ai.request.model", "opentail.endpoint", "outcome" }
                    : new[] { "gen_ai.operation.name", "gen_ai.request.model", "opentail.endpoint" },
                s.Tags.Keys.Order().ToArray()));
            Assert.Equal("success", Assert.Single(samples, s => s.Name == "gen_ai.server.request.duration").Tags["outcome"]);
            var ttft = Assert.Single(samples, s => s.Name == "gen_ai.server.time_to_first_token");
            Assert.True(ttft.Value < 0.05);
            var total = Assert.Single(samples, s => s.Name == "gen_ai.server.request.duration");
            Assert.True(total.Value >= 0.4);
            var tpot = Assert.Single(samples, s => s.Name == "gen_ai.server.time_per_output_token");
            Assert.InRange(tpot.Value, 0.009, 0.05);   // ~100ms / 10 tokens, nowhere near (400ms / 10)
        }
    }

    [Fact]
    public void ActiveQueueAndRejectedInstruments_AreRecorded()
    {
        const string endpoint = "test.active_queue";
        long active = 0, rejected = 0; int queue = 0;
        using var listener = new System.Diagnostics.Metrics.MeterListener();
        listener.InstrumentPublished = (i, l) => { if (i.Meter.Name == "OpenTail.Stingray.Server") l.EnableMeasurementEvents(i); };
        listener.SetMeasurementEventCallback<long>((i, v, t, s) =>
        {
            var tags = new Dictionary<string, object?>();
            foreach (var kv in t) tags[kv.Key] = kv.Value;
            if (i.Name == "opentail.requests.active" && (string?)tags["opentail.endpoint"] == endpoint) Interlocked.Add(ref active, v);
            if (i.Name == "opentail.requests.rejected" && (string?)tags["opentail.reason"] == "overload") Interlocked.Increment(ref rejected);
        });
        listener.SetMeasurementEventCallback<double>((i, v, t, s) =>
        {
            foreach (var kv in t)
                if (i.Name == "opentail.queue.wait" && kv.Key == "opentail.endpoint" && (string?)kv.Value == endpoint) Interlocked.Increment(ref queue);
        });
        listener.Start();

        var metrics = new ServerMetrics();
        var timing = metrics.BeginServingRequest(endpoint, "test-model");
        Assert.Equal(1, Interlocked.Read(ref active));
        timing.MarkAdmitted();
        timing.Dispose();
        timing.Dispose(); // idempotent: must not decrement twice
        metrics.RecordOverloadRejection();
        listener.Dispose();

        Assert.Equal(0, Interlocked.Read(ref active));
        Assert.Equal(1, queue);
        Assert.True(Interlocked.Read(ref rejected) >= 1);
    }

    [Theory]
    [InlineData("llama", "llama")]
    [InlineData(null, "unknown")]
    [InlineData("", "unknown")]
    [InlineData("C:\\models\\secret-project\\x.gguf", "other")]
    [InlineData("/home/user/x.gguf", "other")]
    [InlineData("hf:org/model", "other")]
    public void SanitizeModel_NeverExposesPaths(string? input, string expected) =>
        Assert.Equal(expected, ServerMetrics.SanitizeModel(input));

    [Fact]
    public void SanitizeModel_CapsLength() =>
        Assert.Equal("other", ServerMetrics.SanitizeModel(new string('a', 65)));

    [Theory]
    [InlineData("success")]
    [InlineData("cancelled")]
    [InlineData("incomplete")]
    public void RequestDuration_OutcomeReflectsHowTheRequestEnded(string expected)
    {
        string endpoint = "test.outcome." + expected;
        string? seen = null;
        using var listener = new System.Diagnostics.Metrics.MeterListener();
        listener.InstrumentPublished = (i, l) => { if (i.Name == "gen_ai.server.request.duration") l.EnableMeasurementEvents(i); };
        listener.SetMeasurementEventCallback<double>((i, v, t, s) =>
        {
            string? ep = null, oc = null;
            foreach (var kv in t) { if (kv.Key == "opentail.endpoint") ep = (string?)kv.Value; if (kv.Key == "outcome") oc = (string?)kv.Value; }
            if (ep == endpoint) seen = oc;
        });
        listener.Start();

        using var cts = new CancellationTokenSource();
        var metrics = new ServerMetrics();
        var timing = metrics.BeginServingRequest(endpoint, "m", cts.Token);
        timing.MarkFirstToken();
        if (expected == "success") timing.MarkGenerationComplete();
        if (expected == "cancelled") cts.Cancel();
        timing.Dispose();
        listener.Dispose();

        Assert.Equal(expected, seen);
    }

    /// <summary>
    /// One source of truth: the Meter instruments (legacy and GenAI-named) and the summaries that
    /// feed /metrics and `stingray status` are all driven from the same recording methods, so for
    /// the same requests their counts must agree.
    /// </summary>
    [Fact]
    public void MeterInstrumentsAndMetricsSummaries_AgreeOnCounts()
    {
        const string endpoint = "test.agreement";
        int ttftGenAi = 0, durationGenAi = 0;
        long requestsLegacyDelta = 0;
        using var listener = new System.Diagnostics.Metrics.MeterListener();
        listener.InstrumentPublished = (i, l) => { if (i.Meter.Name == "OpenTail.Stingray.Server") l.EnableMeasurementEvents(i); };
        listener.SetMeasurementEventCallback<double>((i, v, t, s) =>
        {
            bool ours = false;
            foreach (var kv in t) if (kv.Key == "opentail.endpoint" && (string?)kv.Value == endpoint) ours = true;
            if (!ours) return;
            if (i.Name == "gen_ai.server.time_to_first_token") Interlocked.Increment(ref ttftGenAi);
            if (i.Name == "gen_ai.server.request.duration") Interlocked.Increment(ref durationGenAi);
        });
        listener.Start();

        var metrics = new ServerMetrics();
        const int n = 5;
        for (int k = 0; k < n; k++)
        {
            metrics.RecordRequest();
            using var timing = metrics.BeginServingRequest(endpoint, "m");
            timing.MarkAdmitted();
            timing.MarkFirstToken();
            timing.MarkGenerationComplete();
            timing.SetOutputTokens(3);
            metrics.RecordTokens(3);
        }
        listener.Dispose();
        requestsLegacyDelta = metrics.TotalRequests;

        Assert.Equal(n, requestsLegacyDelta);
        Assert.Equal(n, metrics.TimeToFirstTokenSummary.Count);
        Assert.Equal(n, ttftGenAi);
        Assert.Equal(n, metrics.RequestDurationSummary.Count);
        Assert.Equal(n, durationGenAi);
        Assert.Equal(n, metrics.QueueLatencySummary.Count);
        Assert.Equal(3 * n, metrics.TotalTokens);
    }

    [Fact]
    public void OperationSpan_IsChildOfCurrentActivity_AndCarriesAllowListedTags()
    {
        var stopped = new List<System.Diagnostics.Activity>();
        using var al = new System.Diagnostics.ActivityListener
        {
            ShouldListenTo = s => s.Name is ServerMetrics.ActivitySourceName or "test.http",
            Sample = (ref System.Diagnostics.ActivityCreationOptions<System.Diagnostics.ActivityContext> _) =>
                System.Diagnostics.ActivitySamplingResult.AllData,
            ActivityStopped = a => { lock (stopped) stopped.Add(a); },
        };
        System.Diagnostics.ActivitySource.AddActivityListener(al);
        using var http = new System.Diagnostics.ActivitySource("test.http");

        var metrics = new ServerMetrics();
        using (var parent = http.StartActivity("http"))
        {
            var timing = metrics.BeginServingRequest("test.span", "my-model");
            timing.MarkFirstToken();
            timing.MarkGenerationComplete();
            timing.SetOutputTokens(5);
            timing.Dispose();
        }
        al.Dispose();

        lock (stopped)
        {
            var op = Assert.Single(stopped, a => a.OperationName == "chat my-model");
            var parent = Assert.Single(stopped, a => a.OperationName == "http");
            Assert.Equal(parent.SpanId, op.ParentSpanId);
            Assert.Equal("success", op.GetTagItem("outcome"));
            Assert.Equal(5L, op.GetTagItem("gen_ai.usage.output_tokens"));
            Assert.All(op.TagObjects, t => Assert.Contains(t.Key, new[]
            {
                "gen_ai.operation.name", "gen_ai.request.model", "opentail.endpoint", "outcome",
                "gen_ai.usage.output_tokens", "gen_ai.usage.input_tokens", "opentail.reused_tokens", "opentail.time_to_first_token_ms",
            }));
        }
    }
}
