using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.TestHost;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Pins the <c>AllowedHosts</c> value that the demo host ships with.
///
/// <para><b>Why this exists.</b> The demo host listens on loopback with no authentication. A web page
/// can still reach a loopback server through DNS rebinding: a name the attacker controls resolves to
/// <c>127.0.0.1</c>, so the browser sends that name in the <c>Host</c> header. The host blocks that by
/// refusing any <c>Host</c> it does not list. A wildcard (<c>*</c>) admits every name, so this test
/// fails if the shipped value is widened again.</para>
/// </summary>
public sealed class HostFilteringTests
{
    /// <summary>
    /// Walks up from the test assembly to the repo directory containing both <c>src/</c> and
    /// <c>tests/</c>. Returns null for a packaged run, so the test can skip rather than fail.
    /// </summary>
    private static string? FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "src")) && Directory.Exists(Path.Combine(dir.FullName, "tests")))
                return dir.FullName;
            dir = dir.Parent;
        }
        return null;
    }

    [Fact]
    public async Task ShippedAllowedHosts_RefusesForeignHostHeaders_AndAdmitsLoopback()
    {
        string? root = FindRepoRoot();
        Assert.SkipWhen(root is null, "repo layout not found — shipped appsettings.json not available here");

        string appsettings = Path.Combine(root!, "src", "OpenTail.Stingray.Server.Host", "appsettings.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(appsettings));
        string allowedHosts = doc.RootElement.GetProperty("AllowedHosts").GetString()
            ?? throw new InvalidOperationException("appsettings.json: AllowedHosts is not a string");

        // The same host-filtering pipeline as the demo host, with the shipped AllowedHosts value and a
        // route that always answers, so only the filter decides the status code.
        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.UseTestServer();
        builder.Configuration["AllowedHosts"] = allowedHosts;
        await using WebApplication app = builder.Build();
        app.MapGet("/probe", () => "ok");
        await app.StartAsync();
        HttpClient client = app.GetTestClient();

        // TestServer routes by the request URI's host, so the foreign name goes in the URI itself.
        HttpResponseMessage rebound = await client.GetAsync("http://evil.example:8080/probe");
        Assert.Equal(HttpStatusCode.BadRequest, rebound.StatusCode);

        foreach (string loopback in new[] { "http://localhost/probe", "http://127.0.0.1:8080/probe", "http://[::1]:8080/probe" })
        {
            HttpResponseMessage ok = await client.GetAsync(loopback);
            Assert.True(ok.StatusCode == HttpStatusCode.OK,
                $"{loopback} should be admitted by the shipped AllowedHosts '{allowedHosts}', got {(int)ok.StatusCode}");
        }
    }
}
