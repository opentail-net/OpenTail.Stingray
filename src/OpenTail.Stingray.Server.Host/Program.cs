using OpenTail.Stingray.Core;
using OpenTail.Stingray.Server;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;

foreach (string unknown in KnownEnvironmentVariables.FindUnknown())
{
    string? suggestion = KnownEnvironmentVariables.SuggestClosest(unknown);
    Console.Error.WriteLine(suggestion is null
        ? $"warning: {unknown} is set but is not read by this build — it will have no effect."
        : $"warning: {unknown} is set but is not read by this build — did you mean {suggestion}?");
}

var builder = WebApplication.CreateSlimBuilder(args);
var environmentOverrides = new ServerEnvironmentOverrideReceipt();
builder.Services.AddSingleton(environmentOverrides);

// Per-developer overrides (not committed) layer on top of appsettings.json.
builder.Configuration.AddJsonFile("appsettings.Local.json", optional: true, reloadOnChange: true);

// The server configuration section binds first. Valid legacy STINGRAY_* values then override it;
// the resolver is shared and testable, rather than letting precedence hide in this host file.
builder.Services.AddOpenTailStingray(builder.Configuration, opts =>
    environmentOverrides.Record(ServerEnvironmentOverrides.Apply(opts, Environment.GetEnvironmentVariable)));

// OpenTelemetry export: opt-in, configured by the standard OTEL_* variables, off when none is set.
//   OTEL_EXPORTER_OTLP_ENDPOINT (or the _TRACES_/_METRICS_ variants)  -> OTLP export
//   OTEL_TRACES_EXPORTER=console / OTEL_METRICS_EXPORTER=console     -> print to the console (local development)
// Metrics export every 30 s unless OTEL_METRIC_EXPORT_INTERVAL says otherwise: on a CPU-bound run the exporter's
// background work competes with the compute threads, so the default is deliberately long.
static string? Env(string name) => Environment.GetEnvironmentVariable(name);
static bool IsConsole(string name) => string.Equals(Env(name)?.Trim(), "console", StringComparison.OrdinalIgnoreCase);
bool otlp = !string.IsNullOrWhiteSpace(Env("OTEL_EXPORTER_OTLP_ENDPOINT"))
    || !string.IsNullOrWhiteSpace(Env("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))
    || !string.IsNullOrWhiteSpace(Env("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"));
bool consoleTraces = IsConsole("OTEL_TRACES_EXPORTER");
bool consoleMetrics = IsConsole("OTEL_METRICS_EXPORTER");
if (otlp || consoleTraces || consoleMetrics)
{
    var otel = builder.Services.AddOpenTelemetry();
    if (string.IsNullOrWhiteSpace(Env("OTEL_SERVICE_NAME")))
        otel.ConfigureResource(r => r.AddService("stingray"));
    otel.WithTracing(t =>
    {
        t.AddAspNetCoreInstrumentation()
         .AddSource(ServerMetrics.ActivitySourceName, "OpenTail.Stingray.Engine");
        if (otlp) t.AddOtlpExporter();
        if (consoleTraces) t.AddConsoleExporter();
    });
    otel.WithMetrics(m =>
    {
        m.AddAspNetCoreInstrumentation()
         .AddMeter("OpenTail.Stingray.Server", "OpenTail.Stingray.Engine");
        bool defaultInterval = string.IsNullOrWhiteSpace(Env("OTEL_METRIC_EXPORT_INTERVAL"));
        if (otlp)
            m.AddOtlpExporter((_, reader) =>
            {
                if (defaultInterval) reader.PeriodicExportingMetricReaderOptions.ExportIntervalMilliseconds = 30_000;
            });
        if (consoleMetrics)
            m.AddConsoleExporter((_, reader) =>
            {
                if (defaultInterval) reader.PeriodicExportingMetricReaderOptions.ExportIntervalMilliseconds = 30_000;
            });
    });
}

var app = builder.Build();
app.MapOpenTailStingray();
app.Run();

// Required for WebApplicationFactory<Program> in integration tests.
public partial class Program { }
