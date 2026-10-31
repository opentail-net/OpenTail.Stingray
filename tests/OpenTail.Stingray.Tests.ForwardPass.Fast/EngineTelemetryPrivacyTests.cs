using System.Diagnostics;
using System.Diagnostics.Metrics;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// OpenTelemetry plan section 7: no prompt or completion text, and no model path, anywhere in telemetry, with no debug flag. The
/// per-instrument tests check tag KEYS one case at a time; this one pushes a unique canary through both engines and searches
/// EVERYTHING captured (metric names, units, tag keys and values; span names, display names, tags, events, links, baggage) for it.
/// </summary>
[Collection("EngineTelemetry")]
public sealed class EngineTelemetryPrivacyTests
{
    private const string Canary = "CANARY-7f3a91-private";
    private const string PathCanary = @"C:\Users\canary-user\secret-project\model.gguf";

    private static void Collect(List<string> sink, string where, string? value)
    {
        if (value is not null) lock (sink) sink.Add($"{where}={value}");
    }

    private static MeterListener ListenToEverything(List<string> sink)
    {
        var l = new MeterListener();
        l.InstrumentPublished = (i, ml) =>
        {
            if (i.Meter.Name != EngineTelemetry.MeterName) return;
            Collect(sink, "instrument", i.Name);
            Collect(sink, "unit", i.Unit);
            Collect(sink, "description", i.Description);
            ml.EnableMeasurementEvents(i);
        };
        void Tags(ReadOnlySpan<KeyValuePair<string, object?>> t)
        {
            foreach (var kv in t) { Collect(sink, "metric tag key", kv.Key); Collect(sink, "metric tag value", kv.Value?.ToString()); }
        }
        l.SetMeasurementEventCallback<long>((i, v, t, s) => Tags(t));
        l.SetMeasurementEventCallback<double>((i, v, t, s) => Tags(t));
        l.SetMeasurementEventCallback<int>((i, v, t, s) => Tags(t));
        l.Start();
        return l;
    }

    private static ActivityListener ListenToSpans(List<string> sink)
    {
        void Dump(Activity a)
        {
            Collect(sink, "span name", a.OperationName);
            Collect(sink, "span display name", a.DisplayName);
            Collect(sink, "span status", a.StatusDescription);
            foreach (var t in a.TagObjects) { Collect(sink, "span tag key", t.Key); Collect(sink, "span tag value", t.Value?.ToString()); }
            foreach (var e in a.Events)
            {
                Collect(sink, "span event", e.Name);
                foreach (var t in e.Tags) { Collect(sink, "span event tag key", t.Key); Collect(sink, "span event tag value", t.Value?.ToString()); }
            }
            foreach (var link in a.Links) foreach (var t in link.Tags ?? []) Collect(sink, "span link tag", $"{t.Key}={t.Value}");
            foreach (var b in a.Baggage) Collect(sink, "baggage", $"{b.Key}={b.Value}");
        }
        var al = new ActivityListener
        {
            ShouldListenTo = s => s.Name == EngineTelemetry.ActivitySourceName,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllData,
            ActivityStopped = Dump,
        };
        ActivitySource.AddActivityListener(al);
        return al;
    }

    private static async Task Drain(IAsyncEnumerable<GenerateChunk> chunks)
    {
        await foreach (var _ in chunks) { }
    }

    [Fact]
    public async Task APromptContainingACanary_NeverAppearsInAnyMetricOrSpan_OfEitherEngine()
    {
        var captured = new List<string>();
        using var meters = ListenToEverything(captured);
        using var spans = ListenToSpans(captured);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 6 };

        var tokenizer = new InferenceEngineChunkTests.ScriptedTokenizer();
        var fwd = new InferenceEngineChunkTests.ScriptedForwardPass([0, 1, 6], tokenizer.VocabSize);
        using (var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1))
            await Drain(engine.GenerateChunksAsync(Canary, sp));

        using (var batching = new ContinuousBatchingEngine(
                   new ContinuousBatchingConstraintTests.FakeBatchedForwardPass(),
                   new ContinuousBatchingConstraintTests.CharTokenizer(), "test", maxBatchSize: 1))
            await Drain(batching.GenerateChunksAsync(Canary, sp));

        using (var load = EngineTelemetry.StartModelLoad())
            EngineTelemetry.RecordModelLoad(load.Activity, "llama", "ForwardPass", EngineTelemetry.OutcomeSuccess, 0.5);

        string[] seen;
        lock (captured) seen = [.. captured];
        // Not vacuous: both engines and the load really did emit telemetry.
        Assert.Contains(seen, s => s.StartsWith("instrument=opentail.engine.requests", StringComparison.Ordinal));
        Assert.Contains(seen, s => s.StartsWith("span name=opentail.engine.generate", StringComparison.Ordinal));
        Assert.True(seen.Length > 30, $"only {seen.Length} telemetry strings captured");

        Assert.DoesNotContain(seen, s => s.Contains(Canary, StringComparison.OrdinalIgnoreCase));
        Assert.DoesNotContain(seen, s => s.Contains("canary", StringComparison.OrdinalIgnoreCase));
        // Paths and user directories never appear either.
        Assert.DoesNotContain(seen, s => s.Contains(PathCanary, StringComparison.OrdinalIgnoreCase)
                                         || s.Contains(".gguf", StringComparison.OrdinalIgnoreCase)
                                         || s.Contains(@"C:\", StringComparison.OrdinalIgnoreCase));
    }
}
