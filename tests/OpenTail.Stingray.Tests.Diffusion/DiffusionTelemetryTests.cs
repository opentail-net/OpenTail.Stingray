using System.Diagnostics.Metrics;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>Denoising-loop instruments: counts, step timing, no change to the numerical result.</summary>
[Collection("DiffusionTelemetry")]
public sealed class DiffusionTelemetryTests
{
    private static float[] Run(EulerFlowScheduler scheduler, List<(int, int)>? progress = null) =>
        scheduler.Denoise(new float[] { 1f, 2f, 3f, 4f }, (x, t) => x.Select(v => v * 0.5f + t).ToArray(),
            progress is null ? null : (s, n) => progress.Add((s, n)));

    [Fact]
    public void Denoise_RecordsStepsAndDurations_AndKeepsTheResultAndCallbacksIdentical()
    {
        var scheduler = EulerFlowScheduler.Linear(4);
        float[] baseline = Run(scheduler);   // no listener attached: the plain path

        long steps = 0; int stepSamples = 0, loops = 0; string? outcome = null;
        using var listener = new MeterListener();
        listener.InstrumentPublished = (i, l) => { if (i.Meter.Name == DiffusionTelemetry.MeterName) l.EnableMeasurementEvents(i); };
        listener.SetMeasurementEventCallback<long>((i, v, t, s) =>
        {
            if (i.Name != "opentail.diffusion.steps") return;
            Interlocked.Add(ref steps, v);
            foreach (var kv in t) if (kv.Key == "outcome") outcome = (string?)kv.Value;
        });
        listener.SetMeasurementEventCallback<double>((i, v, t, s) =>
        {
            Assert.True(v >= 0);
            if (i.Name == "opentail.diffusion.step.duration") Interlocked.Increment(ref stepSamples);
            if (i.Name == "opentail.diffusion.denoise.duration") Interlocked.Increment(ref loops);
        });
        listener.Start();

        var progress = new List<(int, int)>();
        float[] observed = Run(scheduler, progress);
        listener.Dispose();

        Assert.Equal(baseline, observed);                                   // telemetry never changes the numbers
        Assert.Equal(new[] { (1, 4), (2, 4), (3, 4), (4, 4) }, progress);   // the caller's progress callback still sees every step
        Assert.Equal(4, Interlocked.Read(ref steps));
        Assert.Equal(4, stepSamples);
        Assert.Equal(1, loops);
        Assert.Equal("success", outcome);
    }

    [Fact]
    public void Denoise_WhenTheModelThrows_RecordsTheStepsRunAndAnErrorOutcome()
    {
        var scheduler = EulerFlowScheduler.Linear(4);
        long steps = 0; string? outcome = null;
        using var listener = new MeterListener();
        listener.InstrumentPublished = (i, l) => { if (i.Meter.Name == DiffusionTelemetry.MeterName) l.EnableMeasurementEvents(i); };
        listener.SetMeasurementEventCallback<long>((i, v, t, s) =>
        {
            if (i.Name != "opentail.diffusion.steps") return;
            Interlocked.Add(ref steps, v);
            foreach (var kv in t) if (kv.Key == "outcome") outcome = (string?)kv.Value;
        });
        listener.Start();

        int calls = 0;
        Assert.Throws<InvalidOperationException>(() => scheduler.Denoise(new float[] { 1f }, (x, t) =>
            ++calls == 3 ? throw new InvalidOperationException("boom") : x));
        listener.Dispose();

        Assert.Equal(2, Interlocked.Read(ref steps));   // two steps completed before the third failed
        Assert.Equal("error", outcome);
    }

    [Fact]
    public void DiffusionMeterInstruments_MatchTheCheckedInTable()
    {
        string? table = null;
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null && table is null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "docs", "reference", "telemetry-instruments.md");
            if (File.Exists(candidate)) table = candidate;
        }
        Assert.SkipWhen(table is null, "repo docs not found from the test output folder");
        var rows = new Dictionary<string, (string Type, string Unit)>();
        foreach (string line in File.ReadAllLines(table!))
        {
            if (!line.StartsWith("| `" + DiffusionTelemetry.MeterName + "`", StringComparison.Ordinal)) continue;
            var cells = line.Trim().Trim('|').Split('|').Select(c => c.Trim().Trim('`')).ToArray();
            rows[cells[1]] = (cells[2], cells[3]);
        }

        // Touch a static member so the instruments exist, then enumerate them.
        EulerFlowScheduler.Linear(1).Denoise(new float[] { 1f }, (x, t) => x);
        var published = new Dictionary<string, Instrument>();
        using var listener = new MeterListener();
        listener.InstrumentPublished = (i, l) => { if (i.Meter.Name == DiffusionTelemetry.MeterName) lock (published) published[i.Name] = i; };
        listener.Start();
        lock (published)
        {
            Assert.Equal(rows.Keys.Order(), published.Keys.Order());
            foreach (var (name, i) in published)
            {
                string type = i.GetType().Name switch { "Counter`1" => "counter", "Histogram`1" => "histogram", var o => o };
                Assert.Equal(rows[name].Type, type);
                Assert.Equal(rows[name].Unit, i.Unit);
            }
        }
    }
}

[CollectionDefinition("DiffusionTelemetry", DisableParallelization = true)]
public sealed class DiffusionTelemetryCollection;
