using System.Diagnostics.Metrics;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Phase 1 of the OpenTelemetry plan: engine instruments. The Meter is process-wide and other
/// tests generate too, so assertions look for the expected measurements, never exact totals.
/// </summary>
[Collection("EngineTelemetry")]
public sealed class EngineTelemetryTests
{
    private const int TokHi = 0, TokThere = 1, TokEos = 6;

    private sealed record Sample(string Name, string Unit, double Value, Dictionary<string, object?> Tags);

    private static MeterListener Listen(List<Sample> sink, Action<Instrument>? onPublished = null)
    {
        var l = new MeterListener();
        l.InstrumentPublished = (i, ml) =>
        {
            if (i.Meter.Name != EngineTelemetry.MeterName) return;
            onPublished?.Invoke(i);
            ml.EnableMeasurementEvents(i);
        };
        void Add(Instrument i, double v, ReadOnlySpan<KeyValuePair<string, object?>> t)
        {
            var d = new Dictionary<string, object?>();
            foreach (var kv in t) d[kv.Key] = kv.Value;
            lock (sink) sink.Add(new Sample(i.Name, i.Unit ?? "", v, d));
        }
        l.SetMeasurementEventCallback<long>((i, v, t, s) => Add(i, v, t));
        l.SetMeasurementEventCallback<double>((i, v, t, s) => Add(i, v, t));
        l.Start();
        return l;
    }

    private static InferenceEngine NewEngine(int length = 0)
    {
        var tokenizer = new InferenceEngineChunkTests.ScriptedTokenizer();
        var script = length == 0 ? new[] { TokHi, TokThere, TokEos } : Enumerable.Repeat(TokHi, length).ToArray();
        var fwd = new InferenceEngineChunkTests.ScriptedForwardPass(script, tokenizer.VocabSize);
        return new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);
    }

    private static async Task RunOnce(InferenceEngine e, CancellationToken ct = default)
    {
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };
        await foreach (var _ in e.GenerateChunksAsync("seed", sp, ct)) { }
    }

    [Fact]
    public async Task Success_RecordsExpectedInstrumentsUnitsAndTags()
    {
        var samples = new List<Sample>();
        using var l = Listen(samples);
        using var engine = NewEngine();
        await RunOnce(engine);
        l.Dispose();

        Sample[] ok;
        lock (samples) ok = samples.Where(s => s.Tags.TryGetValue("outcome", out var o) && (string?)o == "success").ToArray();
        string[] names =
        [
            "opentail.engine.requests", "opentail.engine.prefill.duration", "opentail.engine.prefill.tokens",
            "opentail.engine.decode.duration", "opentail.engine.decode.tokens",
            "opentail.engine.first_token.duration", "opentail.engine.generation.duration",
        ];
        foreach (var n in names) Assert.Contains(ok, s => s.Name == n);
        Assert.All(ok.Where(s => s.Name.EndsWith(".duration")), s => Assert.Equal("s", s.Unit));
        Assert.Contains(ok, s => s.Name == "opentail.engine.decode.tokens" && s.Value == 2);
        // Allow-list: only these attribute keys may ever appear.
        Assert.All(samples, s => Assert.All(s.Tags.Keys, k => Assert.Contains(k, new[] { "outcome", "opentail.forward_pass", "opentail.architecture" })));
    }

    [Fact]
    public async Task Cancelled_RecordsCancelledOutcome()
    {
        var samples = new List<Sample>();
        using var l = Listen(samples);
        using var cts = new CancellationTokenSource();
        var tok = new InferenceEngineChunkTests.ScriptedTokenizer();
        var fwd = new InferenceEngineChunkTests.ScriptedForwardPass(Enumerable.Repeat(TokHi, 100).ToArray(), tok.VocabSize) { OnPrefill = cts.Cancel };
        using var engine = new InferenceEngine(fwd, tok, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        try
        {
            var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 100 };
            await foreach (var _ in engine.GenerateChunksAsync("seed", sp, cts.Token)) { }
        }
        catch (OperationCanceledException) { }
        l.Dispose();

            Assert.True(samples.Any(s => s.Name == "opentail.engine.requests" && (string?)s.Tags["outcome"] == "cancelled"), "outcomes: " + string.Join(",", samples.Where(s => s.Name == "opentail.engine.requests").Select(s => s.Tags["outcome"])));
            Assert.Contains(samples, s => s.Name == "opentail.engine.requests" && (string?)s.Tags["outcome"] == "cancelled");
    }

    [Fact]
    public async Task Activity_IsEmittedWithCountsOnlyTags()
    {
        var stopped = new List<System.Diagnostics.Activity>();
        using var al = new System.Diagnostics.ActivityListener
        {
            ShouldListenTo = s => s.Name == EngineTelemetry.ActivitySourceName,
            Sample = (ref System.Diagnostics.ActivityCreationOptions<System.Diagnostics.ActivityContext> _) =>
                System.Diagnostics.ActivitySamplingResult.AllData,
            ActivityStopped = a => { lock (stopped) stopped.Add(a); },
        };
        System.Diagnostics.ActivitySource.AddActivityListener(al);
        using var engine = NewEngine();
        await RunOnce(engine);
        al.Dispose();

        lock (stopped)
        {
            var a = Assert.Single(stopped, x => x.OperationName == "opentail.engine.generate");
            Assert.Equal("success", a.GetTagItem("outcome"));
            Assert.Equal(2, a.GetTagItem("opentail.output_tokens"));
            Assert.All(a.TagObjects, t => Assert.Contains(t.Key,
                new[] { "outcome", "opentail.prompt_tokens", "opentail.reused_tokens", "opentail.output_tokens" }));
        }
    }

    [Fact]
    public void ModelLoad_RecordsDurationWithAllowListedTags()
    {
        var samples = new List<Sample>();
        using var l = Listen(samples);
        EngineTelemetry.RecordModelLoad("llama", "ForwardPass", EngineTelemetry.OutcomeSuccess, 1.5);
        l.Dispose();

        lock (samples)
        {
            var s = Assert.Single(samples, x => x.Name == "opentail.engine.model.load.duration" && x.Value == 1.5);
            Assert.Equal("s", s.Unit);
            Assert.Equal(new[] { "opentail.architecture", "opentail.forward_pass", "outcome" }, s.Tags.Keys.Order().ToArray());
        }
    }

    [Fact]
    public async Task NoListener_GenerationStillWorks()
    {
        using var engine = NewEngine();
        await RunOnce(engine);
    }

    [Fact]
    public async Task ConcurrentRequests_AllCountedAndNotCrossContaminated()
    {
        var samples = new List<Sample>();
        using var l = Listen(samples);
        using var engine = NewEngine();
        await Task.WhenAll(Enumerable.Range(0, 4).Select(_ => RunOnce(engine)));
        l.Dispose();

        lock (samples)
        {
            Assert.True(samples.Count(s => s.Name == "opentail.engine.decode.tokens" && s.Value == 2) >= 4);
            Assert.DoesNotContain(samples, s => s.Name == "opentail.engine.decode.tokens" && s.Value is < 0 or > 2);
        }
    }

    [Fact]
    public async Task StopChunk_CarriesPerRequestStats()
    {
        using var engine = NewEngine();
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };
        GenerationStats? stats = null;
        await foreach (var c in engine.GenerateChunksAsync("seed", sp))
            if (c.Kind == GenerateChunkKind.Stop) stats = c.Stats;

        Assert.NotNull(stats);
        Assert.Equal(1, stats!.PromptTokens);
        Assert.Equal(2, stats.OutputTokens);
        Assert.Equal(0, stats.ReusedTokens);
        Assert.True(stats.FirstTokenMs >= 0);
        Assert.True(stats.TotalMs >= stats.FirstTokenMs);
    }
}

[CollectionDefinition("EngineTelemetry", DisableParallelization = true)]
public sealed class EngineTelemetryCollection;
