using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Globalization;
using System.Text;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// STINGRAY_TELEMETRY=1: an in-process listener over the engine's instruments and spans that prints a short report at
/// exit (stderr). No collector, package or native dependency; counts and durations only. It is also the proof that the
/// published NativeAOT binary delivers <c>Meter</c> and <c>ActivitySource</c> data to an in-process listener.
/// </summary>
internal sealed class TelemetrySummary : IDisposable
{
    private sealed class Agg
    {
        public long Count;
        public double Sum, Min = double.MaxValue, Max = double.MinValue;
        public string Unit = "";
        public void Add(double v) { Count++; Sum += v; if (v < Min) Min = v; if (v > Max) Max = v; }
    }

    private readonly object _gate = new();
    private readonly SortedDictionary<string, Agg> _metrics = new(StringComparer.Ordinal);
    private readonly SortedDictionary<string, int> _spans = new(StringComparer.Ordinal);
    private readonly MeterListener _meters = new();
    private readonly ActivityListener _activities;
    private readonly IDisposable _bridge;

    public TelemetrySummary()
    {
        _bridge = OpenTail.Stingray.Core.EngineEvents.Subscribe(new EngineEventsTelemetryBridge("cli"));
        _meters.InstrumentPublished = (instrument, listener) =>
        {
            if (instrument.Meter.Name == EngineTelemetry.MeterName) listener.EnableMeasurementEvents(instrument);
        };
        _meters.SetMeasurementEventCallback<long>((i, v, t, s) => Record(i, v));
        _meters.SetMeasurementEventCallback<double>((i, v, t, s) => Record(i, v));
        _meters.Start();
        _activities = new ActivityListener
        {
            ShouldListenTo = s => s.Name == EngineTelemetry.ActivitySourceName,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllData,
            ActivityStopped = a => { lock (_gate) _spans[a.OperationName] = _spans.GetValueOrDefault(a.OperationName) + 1; },
        };
        ActivitySource.AddActivityListener(_activities);
    }

    private void Record(Instrument instrument, double value)
    {
        lock (_gate)
        {
            if (!_metrics.TryGetValue(instrument.Name, out var agg))
                _metrics[instrument.Name] = agg = new Agg { Unit = instrument.Unit ?? "" };
            agg.Add(value);
        }
    }

    public string Render()
    {
        var sb = new StringBuilder("[telemetry] engine instruments seen by the in-process listener\n");
        lock (_gate)
        {
            if (_metrics.Count == 0 && _spans.Count == 0) sb.Append("  (nothing recorded)\n");
            foreach (var (name, a) in _metrics)
                sb.Append("  ").Append(name).Append(" [").Append(a.Unit).Append("] n=").Append(a.Count.ToString(CultureInfo.InvariantCulture))
                  .Append(" sum=").Append(a.Sum.ToString("0.###", CultureInfo.InvariantCulture))
                  .Append(" min=").Append(a.Min.ToString("0.###", CultureInfo.InvariantCulture))
                  .Append(" max=").Append(a.Max.ToString("0.###", CultureInfo.InvariantCulture)).Append('\n');
            foreach (var (name, n) in _spans)
                sb.Append("  span ").Append(name).Append(" x").Append(n.ToString(CultureInfo.InvariantCulture)).Append('\n');
        }
        return sb.ToString();
    }

    public void Dispose()
    {
        _bridge.Dispose();
        _meters.Dispose();
        _activities.Dispose();
    }
}
