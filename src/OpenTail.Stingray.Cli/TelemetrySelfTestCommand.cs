using System.Diagnostics;
using System.Diagnostics.Metrics;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// Usage: <c>stingray telemetry-selftest</c>. Needs no model: it emits synthetic engine events and a synthetic span through
/// the real instruments and checks that an in-process listener receives them, then exits 0 (ok) or 1 (a failure is named).
/// It exists so a CI job can publish the NativeAOT binary and prove that <c>Meter</c> and <c>ActivitySource</c> data actually
/// reach a listener in the published executable (a trimmed or switched-off build delivers nothing, and compiling is not proof).
/// </summary>
public sealed class TelemetrySelfTestCommand : Command<TelemetrySelfTestCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
    }

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        int spans = 0;
        using var meters = new MeterListener();
        meters.InstrumentPublished = (instrument, listener) =>
        {
            if (instrument.Meter.Name == EngineTelemetry.MeterName) listener.EnableMeasurementEvents(instrument);
        };
        meters.SetMeasurementEventCallback<long>((i, v, t, s) => seen.Add(i.Name));
        meters.SetMeasurementEventCallback<double>((i, v, t, s) => seen.Add(i.Name));
        meters.Start();
        using var activities = new ActivityListener
        {
            ShouldListenTo = s => s.Name == EngineTelemetry.ActivitySourceName,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllData,
            ActivityStopped = _ => spans++,
        };
        ActivitySource.AddActivityListener(activities);

        // Synthetic traffic through the same path the CLI run command uses.
        using (EngineEvents.Subscribe(new EngineEventsTelemetryBridge("selftest")))
        {
            EngineEvents.Emit(new(EngineEventKind.PrefillCompleted, A: 10, Ms: 5));
            EngineEvents.Emit(new(EngineEventKind.DecodeCompleted, A: 4, B: 4, Ms: 2));
        }
        EngineTelemetry.RunSelfTestSpan();

        string[] expected =
        [
            "opentail.engine.prefill.duration", "opentail.engine.prefill.tokens",
            "opentail.engine.decode.duration", "opentail.engine.decode.tokens", "opentail.engine.requests",
        ];
        var missing = expected.Where(n => !seen.Contains(n)).ToList();
        if (spans == 0) missing.Add("span opentail.engine.selftest");

        if (missing.Count == 0)
        {
            Console.WriteLine($"telemetry-selftest: ok ({seen.Count} instruments and {spans} span received by an in-process listener)");
            return 0;
        }
        Console.Error.WriteLine("telemetry-selftest: FAILED, nothing received for: " + string.Join(", ", missing)
            + ". In a published binary this usually means MetricsSupport is false or the instruments were trimmed.");
        return 1;
    }
}
