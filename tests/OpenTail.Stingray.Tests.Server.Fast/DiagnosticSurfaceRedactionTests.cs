using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Server;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Standalone cross-surface privacy suite (§8 Phase 2 item 5 of the QoL plan). The individual
/// diagnostic endpoints each carry their own narrow redaction check
/// (<c>StatusDocumentTests.Status_PublishesNoFilesystemPaths</c>,
/// <c>DoctorBundleTests</c>'s planted-secret cases); this file is the one place that walks every
/// read-only HTTP diagnostic surface — <c>/status</c>, <c>/capabilities</c>, <c>/health</c> — against
/// the same configured secret in one sweep, so a new endpoint added later has to be added here too
/// rather than silently sitting outside redaction coverage.
/// </summary>
public sealed class DiagnosticSurfaceRedactionTests : IDisposable
{
    private readonly List<WebApplicationFactory<Program>> _factories = new();

    public void Dispose()
    {
        foreach (var factory in _factories) factory.Dispose();
    }

    // Every path-shaped setting OpenTailStingrayServerOptions exposes. A path discloses a username, a
    // project name, or a private/unreleased model name — the thing this suite exists to catch.
    private HttpClient CreateClientWithConfiguredPaths()
    {
        var factory = new WebApplicationFactory<Program>()
            .WithWebHostBuilder(builder => builder.ConfigureServices(services =>
            {
                services.Configure<OpenTailStingrayServerOptions>(o =>
                {
                    o.ModelPath = @"C:\Users\dmitri\private-models\unreleased-finetune-sk-9137.gguf";
                    o.MmprojPath = @"C:\Users\dmitri\private-models\mmproj-sk-9137.gguf";
                    o.DSparkModelPath = @"C:\Users\dmitri\private-models\dspark-sk-9137";
                    o.ExpertStatsPath = @"C:\Users\dmitri\private-models\expert-stats-sk-9137.json";
                    // Newest path-shaped option, and the one the /status bound-configuration block
                    // touches: it reports SessionPersistenceConfigured as a BOOLEAN derived from
                    // this. Setting it here is what keeps that a deliberate choice — swap the bool
                    // for the directory itself and this sweep fails, which is the whole point.
                    o.SessionStorageDirectory = @"C:\Users\dmitri\private-models\sessions-sk-9137";
                });
                services.AddSingleton<IInferenceEngine>(new FakeInferenceEngine("test-model"));
            }));
        _factories.Add(factory);
        return factory.CreateClient();
    }

    public static IEnumerable<object[]> DiagnosticGetEndpoints =>
    [
        ["/status"],
        ["/capabilities"],
        ["/health"],
    ];

    [Theory]
    [MemberData(nameof(DiagnosticGetEndpoints))]
    public async Task DiagnosticEndpoint_NeverLeaksConfiguredFilesystemPaths(string path)
    {
        var client = CreateClientWithConfiguredPaths();
        var response = await client.GetAsync(path);
        string body = await response.Content.ReadAsStringAsync();

        Assert.DoesNotContain("private-models", body, StringComparison.Ordinal);
        Assert.DoesNotContain("unreleased-finetune", body, StringComparison.Ordinal);
        Assert.DoesNotContain("sk-9137", body, StringComparison.Ordinal);
        Assert.DoesNotContain("dmitri", body, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(@"C:\", body, StringComparison.Ordinal);
        Assert.DoesNotContain(@"\", body, StringComparison.Ordinal);
    }

    /// <summary>
    /// The opt-in <c>opentail_timing</c> response extension (§8 Phase 2 item 3) sits on the
    /// generation endpoints rather than the read-only diagnostic ones, but it is new surface added
    /// in the same slice as this suite, so it gets the same sweep: timing numbers only, no path or
    /// prompt content leaking in alongside them.
    /// </summary>
    [Fact]
    public async Task TimingExtension_NeverLeaksConfiguredFilesystemPaths()
    {
        var client = CreateClientWithConfiguredPaths();
        var response = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "hi" } },
            stream = false,
            opentail_timing = true,
        });
        string body = await response.Content.ReadAsStringAsync();

        Assert.DoesNotContain("private-models", body, StringComparison.Ordinal);
        Assert.DoesNotContain("sk-9137", body, StringComparison.Ordinal);
        Assert.DoesNotContain(@"C:\", body, StringComparison.Ordinal);
    }
}
