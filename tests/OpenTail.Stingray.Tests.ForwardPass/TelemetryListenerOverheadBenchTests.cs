using System.Diagnostics;
using System.Diagnostics.Metrics;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// OpenTelemetry plan section 8, the "in-process listener" case (threshold: at most 2% on the median). The same binary, the same loaded model and the same
/// engine generate 64 greedy tokens with and without an in-process listener attached (a <see cref="MeterListener"/> enabling every engine instrument, and an
/// <see cref="ActivityListener"/> sampling everything), toggled between runs and interleaved, so the only difference is the listener: no uninstrumented baseline
/// build, and none of the other changes made between two commits, is involved. The listener only counts, which is what an in-process listener costs before any
/// exporter work is added (the exporter case is measured separately, with real server binaries). Real weights, timed; the generated text must be identical.
/// Prints the table; the assertions are equal output and that the listener really received measurements and activities (not vacuous).
/// </summary>
public sealed class TelemetryListenerOverheadBenchTests : HeavyTestBase
{
    private const int NewTokens = 64;
    private const int Rounds = 100;

    private static string? FindModel(string fileName)
    {
        for (var dir = new DirectoryInfo(Directory.GetCurrentDirectory()); dir is not null; dir = dir.Parent)
            foreach (string c in new[] { Path.Combine(dir.FullName, "models", fileName), Path.Combine(dir.FullName, "models", "_models", fileName) })
                if (File.Exists(c)) return c;
        string home = Path.Combine(OpenTail.Stingray.Core.Catalog.ModelHome.DefaultRoot(), fileName);
        if (File.Exists(home)) return home;
        foreach (var drive in DriveInfo.GetDrives().Where(d => d.DriveType == DriveType.Fixed && d.IsReady))
        {
            string c = Path.Combine(drive.RootDirectory.FullName, "_models", fileName);
            if (File.Exists(c)) return c;
        }
        return null;
    }

    [Fact]
    public async Task InProcessListener_CostOnARealModel_IsMeasuredInterleavedAgainstNoListener()
    {
        const string modelFile = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
        string? path = FindModel(modelFile);
        Assert.SkipUnless(path is not null, $"{modelFile} is required for this benchmark");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ArchitectureModelResolver.ResolveHyperparams(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new OpenTail.Stingray.Engine.ForwardPass(model, backend, hp);
        using var engine = new InferenceEngine(fwd, tokenizer, "bench", thinkTokenId: -1, endThinkTokenId: -1);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = NewTokens };
        const string prompt = "Explain in two sentences what a hash map is.";

        async Task<(TimeSpan Elapsed, string Text)> Run()
        {
            var text = new System.Text.StringBuilder();
            var clock = Stopwatch.StartNew();
            await foreach (var chunk in engine.GenerateChunksAsync(prompt, sp)) if (chunk.Kind == GenerateChunkKind.Text) text.Append(chunk.Text);
            return (clock.Elapsed, text.ToString());
        }

        long measurements = 0, activities = 0;
        (MeterListener Meters, ActivityListener Spans) AttachListeners()
        {
            var meters = new MeterListener();
            meters.InstrumentPublished = (i, l) => { if (i.Meter.Name == EngineTelemetry.MeterName) l.EnableMeasurementEvents(i); };
            meters.SetMeasurementEventCallback<long>((i, v, t, s) => Interlocked.Increment(ref measurements));
            meters.SetMeasurementEventCallback<double>((i, v, t, s) => Interlocked.Increment(ref measurements));
            meters.SetMeasurementEventCallback<int>((i, v, t, s) => Interlocked.Increment(ref measurements));
            meters.Start();
            var spans = new ActivityListener
            {
                ShouldListenTo = s => s.Name == EngineTelemetry.ActivitySourceName,
                Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllData,
                ActivityStopped = _ => Interlocked.Increment(ref activities),
            };
            ActivitySource.AddActivityListener(spans);
            return (meters, spans);
        }

        // Warm up with and without a listener before anything is timed.
        for (int i = 0; i < 3; i++) await Run();
        var (wm, ws) = AttachListeners(); for (int i = 0; i < 3; i++) await Run(); wm.Dispose(); ws.Dispose();

        string reference = (await Run()).Text;
        Assert.False(string.IsNullOrWhiteSpace(reference));
        var off = new List<double>(); var on = new List<double>();
        measurements = 0; activities = 0;
        for (int round = 0; round < Rounds; round++)
        {
            foreach (bool listener in round % 2 == 0 ? new[] { false, true } : new[] { true, false })   // alternate which goes first
            {
                (MeterListener Meters, ActivityListener Spans)? attached = listener ? AttachListeners() : null;
                var (elapsed, text) = await Run();
                attached?.Meters.Dispose(); attached?.Spans.Dispose();
                Assert.Equal(reference, text);
                (listener ? on : off).Add(elapsed.TotalMilliseconds);
            }
        }

        double Median(List<double> v) { var s = v.OrderBy(x => x).ToList(); return (s[(s.Count - 1) / 2] + s[s.Count / 2]) / 2; }
        double Pct(List<double> v, double p) => v.OrderBy(x => x).ElementAt((int)Math.Floor((v.Count - 1) * p));
        Console.Error.WriteLine($"[TelemetryListenerOverheadBench] {modelFile}, {NewTokens} greedy tokens, {Rounds} interleaved rounds, same engine and binary");
        Console.Error.WriteLine($"  no listener : median {Median(off):F1} ms, p25 {Pct(off, 0.25):F1}, min {off.Min():F1}, max {off.Max():F1}");
        Console.Error.WriteLine($"  listener on : median {Median(on):F1} ms, p25 {Pct(on, 0.25):F1}, min {on.Min():F1}, max {on.Max():F1}");
        Console.Error.WriteLine($"  listener / none: median {Median(on) / Median(off):F4}x, p25 {Pct(on, 0.25) / Pct(off, 0.25):F4}x, min {on.Min() / off.Min():F4}x; listener saw {measurements} measurements and {activities} activities");
        Assert.True(measurements > 0 && activities > 0, "the listener received nothing: the 'on' runs measured no listener cost");
    }
}
