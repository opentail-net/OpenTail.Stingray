using OpenTail.Stingray.Cli.Scout;
using OpenTail.Stingray.Core.Net;
using OpenTail.Stingray.Engine.Scout;
using static OpenTail.Stingray.Tests.Cli.HubFixtures;

namespace OpenTail.Stingray.Tests.Cli;

public sealed class BacklogTriageTests
{
    private static BacklogReport Report(params BacklogEntry[] entries) =>
        new(BacklogReport.CurrentSchemaVersion, "t", 10, 1, entries, [], entries, new NetworkUse(0, [], 0));

    [Fact]
    public async Task Triage_inspects_the_smallest_quant_of_each_unregistered_family_and_skips_admitted_ones()
    {
        var hub = new FakeHub("o/new", [new RepoFile("m-q4.gguf", Gguf("llama", 1)), new RepoFile("m-q8.gguf", Gguf("llama", 1, dataBytes: 1 << 20))]);
        using var http = new ExternalHttpClient(inner: hub, env: _ => null);
        var report = Report(
            new BacklogEntry("newarch", "unregistered", 1000, 1, 0, ["o/new"]),
            new BacklogEntry("llama", "admitted", 5000, 1, 0, ["o/other"]));

        var rows = await BacklogTriage.RunAsync(http, report, 5, new ScoutOptions("t"), default);

        var row = Assert.Single(rows);
        Assert.Equal("newarch", row.Architecture);
        Assert.Equal("inspected", row.Outcome);
        Assert.Contains("m-q4.gguf", row.Detail);
        Assert.DoesNotContain(hub.Requests, q => q.Contains("o/other"));
    }

    [Fact]
    public async Task Triage_respects_the_family_count_and_the_external_access_policy()
    {
        var hub = new FakeHub("o/new", [new RepoFile("m.gguf", Gguf("llama", 1))]);
        var report = Report(new BacklogEntry("a", "unregistered", 2, 1, 0, ["o/new"]), new BacklogEntry("b", "unregistered", 1, 1, 0, ["o/new"]));
        using var http = new ExternalHttpClient(inner: hub, env: _ => null);
        Assert.Single(await BacklogTriage.RunAsync(http, report, 1, new ScoutOptions("t"), default));

        using var denied = new ExternalHttpClient(inner: hub, env: n => n == "STINGRAY_ALLOW_EXTERNAL" ? "0" : null);
        await Assert.ThrowsAsync<ExternalAccessDeniedException>(() => BacklogTriage.RunAsync(denied, report, 1, new ScoutOptions("t"), default));
    }
}
