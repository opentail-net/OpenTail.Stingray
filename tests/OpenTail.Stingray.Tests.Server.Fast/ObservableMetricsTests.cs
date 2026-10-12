using System.Diagnostics.Metrics;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>Observable gauges and counters over state the server already keeps (Phase 4).</summary>
public sealed class ObservableMetricsTests
{
    [Fact]
    public void ObservableInstruments_ArePublished_AndFaultsYieldNoMeasurementInsteadOfThrowing()
    {
        using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(new FakeInferenceEngine("m"))));
        using var client = factory.CreateClient(); // starts the host, which creates the observable instruments

        var published = new HashSet<string>();
        var measured = new Dictionary<string, long>();
        using var listener = new MeterListener();
        listener.InstrumentPublished = (i, l) =>
        {
            if (i.Meter.Name != "OpenTail.Stingray.Server" || !i.Name.StartsWith("opentail.")) return;
            lock (published) published.Add(i.Name);
            l.EnableMeasurementEvents(i);
        };
        listener.SetMeasurementEventCallback<long>((i, v, t, s) => { lock (measured) measured[i.Name] = v; });
        listener.Start();
        listener.RecordObservableInstruments();   // must not throw even if a source is unavailable

        string[] expected =
        [
            "opentail.models.resident", "opentail.models.pending_loads", "opentail.models.estimated_bytes",
            "opentail.admission.queued", "opentail.models.loads", "opentail.models.load_failures",
            "opentail.models.evictions", "opentail.models.admission_rejections",
            "opentail.engine.queue.depth", "opentail.engine.sequences.active",
            "opentail.engine.kv_cache.committed_tokens", "opentail.engine.kv_cache.budget_tokens", "opentail.engine.prefix_cache.entries",
            "opentail.engine.prefix_cache.estimated_bytes", "opentail.engine.prefix_cache.hits",
            "opentail.engine.prefix_cache.misses", "opentail.engine.prefix_cache.evictions",
        ];
        lock (published) foreach (var n in expected) Assert.Contains(n, published);
        // The fake engine reports a queue depth, so at least that engine gauge produced a value.
        lock (measured) Assert.True(measured.ContainsKey("opentail.engine.queue.depth"));
    }
}
