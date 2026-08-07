using OpenTail.Stingray.Core;
using OpenTail.Stingray.Server;

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

var app = builder.Build();
app.MapOpenTailStingray();
app.Run();

// Required for WebApplicationFactory<Program> in integration tests.
public partial class Program { }
