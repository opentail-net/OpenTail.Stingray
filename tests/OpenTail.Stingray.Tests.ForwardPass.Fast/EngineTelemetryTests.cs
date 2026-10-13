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
        Assert.All(samples, s => Assert.All(s.Tags.Keys, k => Assert.Contains(k, new[] { "outcome", "opentail.forward_pass", "opentail.architecture", "opentail.speculative.mode" })));
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
        using (var load = EngineTelemetry.StartModelLoad())
            EngineTelemetry.RecordModelLoad(load.Activity, "llama", "ForwardPass", EngineTelemetry.OutcomeSuccess, 1.5);
        l.Dispose();

        lock (samples)
        {
            var s = Assert.Single(samples, x => x.Name == "opentail.engine.model.load.duration" && x.Value == 1.5);
            Assert.Equal("s", s.Unit);
            Assert.Equal(new[] { "opentail.architecture", "opentail.backend", "opentail.forward_pass", "outcome" }, s.Tags.Keys.Order().ToArray());
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

    [Fact]
    public async Task EngineSpan_IsParentedToTheCallersActivity_AndConcurrentRequestsDoNotCross()
    {
        var stopped = new List<System.Diagnostics.Activity>();
        using var al = new System.Diagnostics.ActivityListener
        {
            ShouldListenTo = s => s.Name is EngineTelemetry.ActivitySourceName or "test.callers",
            Sample = (ref System.Diagnostics.ActivityCreationOptions<System.Diagnostics.ActivityContext> _) =>
                System.Diagnostics.ActivitySamplingResult.AllData,
            ActivityStopped = a => { lock (stopped) stopped.Add(a); },
        };
        System.Diagnostics.ActivitySource.AddActivityListener(al);
        using var callers = new System.Diagnostics.ActivitySource("test.callers");
        using var engine = NewEngine();

        async Task<System.Diagnostics.ActivityTraceId> CallerRun()
        {
            using var parent = callers.StartActivity("caller")!;
            await RunOnce(engine);
            return parent.TraceId;
        }

        var traceIds = await Task.WhenAll(Enumerable.Range(0, 4).Select(_ => Task.Run(CallerRun)));
        al.Dispose();

        lock (stopped)
        {
            var engineSpans = stopped.Where(a => a.OperationName == "opentail.engine.generate").ToArray();
            Assert.Equal(4, engineSpans.Length);
            var callerSpans = stopped.Where(a => a.OperationName == "caller").ToArray();
            Assert.Equal(4, callerSpans.Length);
            // Every engine span's parent is exactly one caller span, and each caller owns exactly one engine span.
            Assert.All(engineSpans, e => Assert.Single(callerSpans, c => c.SpanId == e.ParentSpanId && c.TraceId == e.TraceId));
            Assert.Equal(4, engineSpans.Select(e => e.ParentSpanId).Distinct().Count());
        }
    }

    [Fact]
    public void ModelLoadSpan_IsARootEvenInsideAnotherActivity_WithAllowListedTags()
    {
        var stopped = new List<System.Diagnostics.Activity>();
        using var al = new System.Diagnostics.ActivityListener
        {
            ShouldListenTo = s => s.Name is EngineTelemetry.ActivitySourceName or "test.requester",
            Sample = (ref System.Diagnostics.ActivityCreationOptions<System.Diagnostics.ActivityContext> _) =>
                System.Diagnostics.ActivitySamplingResult.AllData,
            ActivityStopped = a => { lock (stopped) stopped.Add(a); },
        };
        System.Diagnostics.ActivitySource.AddActivityListener(al);
        using var requester = new System.Diagnostics.ActivitySource("test.requester");

        System.Diagnostics.Activity? currentAfterLoad;
        using (var req = requester.StartActivity("request"))
        {
            using (var load = EngineTelemetry.StartModelLoad())
                EngineTelemetry.RecordModelLoad(load.Activity, "llama", "ForwardPass", EngineTelemetry.OutcomeSuccess, 0.1);
            currentAfterLoad = System.Diagnostics.Activity.Current;
            Assert.Same(req, currentAfterLoad); // the caller context survives the load span
        }
        al.Dispose();

        lock (stopped)
        {
            var request = Assert.Single(stopped, a => a.OperationName == "request");
            var load = Assert.Single(stopped, a => a.OperationName == "model.load");
            Assert.Equal(default, load.ParentSpanId);
            Assert.NotEqual(request.TraceId, load.TraceId);
            Assert.All(load.TagObjects, t => Assert.Contains(t.Key, new[] { "opentail.architecture", "opentail.backend", "opentail.forward_pass", "outcome" }));
        }
    }

    [Fact]
    public void EventBridge_TurnsPrefillAndDecodeEventsIntoEngineInstruments()
    {
        var samples = new List<Sample>();
        using var l = Listen(samples);
        using var sub = OpenTail.Stingray.Core.EngineEvents.Subscribe(new EngineEventsTelemetryBridge("bridge-test"));
        OpenTail.Stingray.Core.EngineEvents.Emit(new(OpenTail.Stingray.Core.EngineEventKind.PrefillCompleted, A: 31, Ms: 250));
        OpenTail.Stingray.Core.EngineEvents.Emit(new(OpenTail.Stingray.Core.EngineEventKind.DecodeCompleted, A: 9, B: 8, Ms: 170));
        OpenTail.Stingray.Core.EngineEvents.Emit(new(OpenTail.Stingray.Core.EngineEventKind.GcStats, A: 1)); // ignored
        l.Dispose();

        lock (samples)
        {
            var mine = samples.Where(s => (string?)s.Tags["opentail.forward_pass"] == "bridge-test").ToArray();
            Assert.Contains(mine, s => s.Name == "opentail.engine.prefill.tokens" && s.Value == 31);
            Assert.Contains(mine, s => s.Name == "opentail.engine.prefill.duration" && Math.Abs(s.Value - 0.25) < 1e-9 && s.Unit == "s");
            Assert.Contains(mine, s => s.Name == "opentail.engine.decode.tokens" && s.Value == 9);
            Assert.Contains(mine, s => s.Name == "opentail.engine.decode.duration" && Math.Abs(s.Value - 0.17) < 1e-9);
            Assert.Single(mine, s => s.Name == "opentail.engine.requests");
        }
    }

    [Fact]
    public async Task ContinuousBatching_StopChunkHasStats_MetricsAndParentedSpanAreRecorded()
    {
        var samples = new List<Sample>();
        var stopped = new List<System.Diagnostics.Activity>();
        using var l = Listen(samples);
        using var al = new System.Diagnostics.ActivityListener
        {
            ShouldListenTo = s => s.Name is EngineTelemetry.ActivitySourceName or "test.batch.callers",
            Sample = (ref System.Diagnostics.ActivityCreationOptions<System.Diagnostics.ActivityContext> _) =>
                System.Diagnostics.ActivitySamplingResult.AllData,
            ActivityStopped = a => { lock (stopped) stopped.Add(a); },
        };
        System.Diagnostics.ActivitySource.AddActivityListener(al);
        using var callers = new System.Diagnostics.ActivitySource("test.batch.callers");
        using var engine = new ContinuousBatchingEngine(
            new ContinuousBatchingConstraintTests.FakeBatchedForwardPass(),
            new ContinuousBatchingConstraintTests.CharTokenizer(), "test", maxBatchSize: 1);

        GenerationStats? stats = null;
        System.Diagnostics.Activity? current;
        using (var parent = callers.StartActivity("caller"))
        {
            var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 6 };
            await foreach (var c in engine.GenerateChunksAsync("prompt", sp))
                if (c.Kind == GenerateChunkKind.Stop) stats = c.Stats;
            current = System.Diagnostics.Activity.Current;
            Assert.Same(parent, current); // emitting the span must not disturb the caller's context
        }
        l.Dispose();
        al.Dispose();

        Assert.NotNull(stats);
        Assert.Equal(6, stats!.PromptTokens);       // CharTokenizer: one token per character of "prompt"
        Assert.Equal(6, stats.OutputTokens);
        Assert.True(stats.FirstTokenMs >= 0 && stats.TotalMs >= stats.FirstTokenMs);

        lock (samples)
        {
            var mine = samples.Where(s => (string?)s.Tags["opentail.forward_pass"] == "continuous_batching").ToArray();
            Assert.Contains(mine, s => s.Name == "opentail.engine.requests" && (string?)s.Tags["outcome"] == "success");
            Assert.Contains(mine, s => s.Name == "opentail.engine.decode.tokens" && s.Value == 6);
            Assert.Contains(mine, s => s.Name == "opentail.engine.prefill.tokens" && s.Value == 6);
        }
        lock (stopped)
        {
            var caller = Assert.Single(stopped, a => a.OperationName == "caller");
            var span = Assert.Single(stopped, a => a.OperationName == "opentail.engine.generate");
            Assert.Equal(caller.SpanId, span.ParentSpanId);
            Assert.Equal("success", span.GetTagItem("outcome"));
            Assert.Equal(6, span.GetTagItem("opentail.output_tokens"));
        }
    }

    [Fact]
    public void EventBridge_TurnsMtpStepsIntoSpeculativeCounters_WithAcceptanceDerivable()
    {
        var samples = new List<Sample>();
        using var l = Listen(samples);
        using var sub = OpenTail.Stingray.Core.EngineEvents.Subscribe(new EngineEventsTelemetryBridge("spec-test"));
        // Two verify steps: 3 drafts proposed / 2 accepted, then 3 proposed / 3 accepted.
        OpenTail.Stingray.Core.EngineEvents.Emit(new(OpenTail.Stingray.Core.EngineEventKind.MtpStep, A: 10, B: 3, C: 2));
        OpenTail.Stingray.Core.EngineEvents.Emit(new(OpenTail.Stingray.Core.EngineEventKind.MtpStep, A: 13, B: 3, C: 3));
        l.Dispose();

        lock (samples)
        {
            var spec = samples.Where(s => s.Tags.TryGetValue("opentail.speculative.mode", out var m) && (string?)m == "mtp").ToArray();
            Assert.All(spec, s => Assert.Equal(new[] { "opentail.speculative.mode" }, s.Tags.Keys.ToArray()));
            // The Meter is process-wide and other tests may emit MTP events, so compare against this test's own two steps
            // by asserting the minimum totals and the exact recorded values it must contain.
            Assert.True(spec.Count(s => s.Name == "opentail.engine.speculative.drafts") >= 2);
            Assert.Contains(spec, s => s.Name == "opentail.engine.speculative.draft_tokens" && s.Value == 3);
            Assert.Contains(spec, s => s.Name == "opentail.engine.speculative.accepted_tokens" && s.Value == 2);
            Assert.Contains(spec, s => s.Name == "opentail.engine.speculative.accepted_tokens" && s.Value == 3);
        }
    }

    [Fact]
    public void Placement_RecordsGpuLayerShare_AndAFallbackWithAClosedReason()
    {
        var samples = new List<Sample>();
        using var l = Listen(samples);
        // CUDA explicitly requested, but the plan selected CPU and CUDA is unavailable: a fallback with reason "unavailable".
        var plan = new ExecutionPlan(
            SchemaVersion: 2, ModelPath: "x", Goal: "g", Backend: "cuda", GpuLayers: 10, TotalLayers: 20, ContextSize: 1,
            KvDtype: "f16", EstimatedVramMb: 0, EstimatedRamMb: 0, Decisions: null, Warnings: null,
            Request: new PlanRequest("t", "cuda", 10, 1, false, "f16", "none", 1, false),
            BackendPlan: new BackendPlan(ForwardPassBackend.Cpu, "cpu", 0, CudaAvailable: false, VulkanAvailable: false, ThreadCount: 4),
            Placement: new PlacementPlan(GpuLayers: 10, CpuLayers: 10, TotalLayers: 20, GpuWeightBytes: 0, CpuWeightBytes: 0));
        EngineTelemetry.RecordPlacement(plan);
        l.Dispose();

        lock (samples)
        {
            var share = Assert.Single(samples, s => s.Name == "opentail.engine.placement.gpu_layer_share");
            Assert.Equal(0.5, share.Value);
            Assert.Equal("cpu", share.Tags["opentail.backend"]);
            var fb = Assert.Single(samples, s => s.Name == "opentail.engine.backend.fallbacks");
            Assert.Equal("unavailable", fb.Tags["opentail.fallback.reason"]);
            Assert.Equal("cuda", fb.Tags["opentail.backend.requested"]);
            Assert.Equal("cpu", fb.Tags["opentail.backend"]);
        }
    }
}

[CollectionDefinition("EngineTelemetry", DisableParallelization = true)]
public sealed class EngineTelemetryCollection;
