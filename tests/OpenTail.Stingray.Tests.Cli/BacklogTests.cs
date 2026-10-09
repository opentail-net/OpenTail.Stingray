using System.Net;
using System.Text;
using OpenTail.Stingray.Cli.Scout;
using OpenTail.Stingray.Core.Net;
using Xunit;

namespace OpenTail.Stingray.Tests.Cli;

public sealed class BacklogTests
{
    [Fact]
    public void Grouping_CombinesRepositoriesByArchitecture()
    {
        string json = """
            [
              { "id": "org/repo-1", "downloads": 1000, "gated": false, "gguf": { "architecture": "arch_test", "totalFileSize": 500 } },
              { "id": "org/repo-2", "downloads": 2500, "gated": true, "gguf": { "architecture": "arch_test", "totalFileSize": 700 } },
              { "id": "org/repo-3", "downloads": 800, "gated": false, "gguf": { "architecture": "other_arch", "totalFileSize": 300 } }
            ]
            """;

        var report = Backlog.Parse(json, limit: 50, buildId: "test", new NetworkUse(1, ["huggingface.co"], 1024));

        Assert.Equal(3, report.TotalReposInspected);

        // arch_test should be grouped with 2 repos and 3500 downloads
        var groupTest = report.All.First(e => e.Architecture == "arch_test");
        Assert.Equal(2, groupTest.RepoCount);
        Assert.Equal(3500, groupTest.Downloads30Days);
        Assert.Equal(1, groupTest.GatedCount);
        Assert.Equal(2, groupTest.Repos.Count);
        // Order inside Repos should be by downloads descending (repo-2 has 2500, repo-1 has 1000)
        Assert.Equal("org/repo-2", groupTest.Repos[0]);
        Assert.Equal("org/repo-1", groupTest.Repos[1]);

        // other_arch
        var groupOther = report.All.First(e => e.Architecture == "other_arch");
        Assert.Equal(1, groupOther.RepoCount);
        Assert.Equal(800, groupOther.Downloads30Days);
        Assert.Equal(0, groupOther.GatedCount);
    }

    [Fact]
    public void RankingOrder_RanksArchitecturesByDownloadsDescending()
    {
        string json = """
            [
              { "id": "org/low", "downloads": 100, "gguf": { "architecture": "low_arch" } },
              { "id": "org/high", "downloads": 99999, "gguf": { "architecture": "high_arch" } },
              { "id": "org/mid", "downloads": 5000, "gguf": { "architecture": "mid_arch" } }
            ]
            """;

        var report = Backlog.Parse(json, limit: 50, buildId: "test", new NetworkUse(1, ["huggingface.co"], 512));

        Assert.Equal(3, report.All.Count);
        Assert.Equal("high_arch", report.All[0].Architecture);
        Assert.Equal(99999, report.All[0].Downloads30Days);

        Assert.Equal("mid_arch", report.All[1].Architecture);
        Assert.Equal(5000, report.All[1].Downloads30Days);

        Assert.Equal("low_arch", report.All[2].Architecture);
        Assert.Equal(100, report.All[2].Downloads30Days);
    }

    [Fact]
    public async Task LimitCap_CapsAtHardLimitOf200()
    {
        string? requestedUrl = null;
        var handler = new FakeListHandler(req =>
        {
            requestedUrl = req.RequestUri!.ToString();
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("[]", Encoding.UTF8, "application/json")
            };
        });

        using var http = new ExternalHttpClient(userAgent: "test", inner: handler, env: key => key == "STINGRAY_ALLOW_EXTERNAL" ? "1" : null);

        // Ask for 500, must be clamped to 200
        var report = await Backlog.RunAsync(http, limit: 500, buildId: "test", CancellationToken.None);

        Assert.NotNull(requestedUrl);
        Assert.Contains("limit=200", requestedUrl);
        Assert.Equal(200, report.Limit);
    }

    [Fact]
    public void GatedFlag_HandlesBooleanAndStringFormats()
    {
        string json = """
            [
              { "id": "org/gated-bool", "downloads": 10, "gated": true, "gguf": { "architecture": "arch_gated" } },
              { "id": "org/gated-auto", "downloads": 20, "gated": "auto", "gguf": { "architecture": "arch_gated" } },
              { "id": "org/gated-manual", "downloads": 30, "gated": "manual", "gguf": { "architecture": "arch_gated" } },
              { "id": "org/not-gated-false", "downloads": 40, "gated": false, "gguf": { "architecture": "arch_gated" } },
              { "id": "org/not-gated-null", "downloads": 50, "gated": null, "gguf": { "architecture": "arch_gated" } }
            ]
            """;

        var report = Backlog.Parse(json, limit: 50, buildId: "test", new NetworkUse(1, ["huggingface.co"], 100));
        var entry = report.All.First(e => e.Architecture == "arch_gated");

        Assert.Equal(5, entry.RepoCount);
        Assert.Equal(3, entry.GatedCount);
    }

    [Fact]
    public void MalformedJsonEntries_AreSkippedGracefully()
    {
        string json = """
            [
              { "id": "valid/model1", "downloads": 100, "gguf": { "architecture": "arch1" } },
              "not an object",
              12345,
              { "bad": "no id property" },
              { "id": "valid/model2", "downloads": "not a number", "gguf": { "architecture": "arch2" } },
              { "id": "valid/model3", "downloads": 300, "gguf": "not an object" }
            ]
            """;

        var report = Backlog.Parse(json, limit: 50, buildId: "test", new NetworkUse(1, ["huggingface.co"], 500));

        // valid/model1 and valid/model3 are parsed (model3 defaults arch to (none))
        Assert.True(report.TotalReposInspected >= 2);
        Assert.Contains(report.All, e => e.Architecture == "arch1");
    }

    [Fact]
    public async Task PolicyOff_ThrowsExternalAccessDeniedException()
    {
        var handler = new FakeListHandler(_ => new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent("[]") });

        // Environment denies external access
        using var http = new ExternalHttpClient(userAgent: "test", inner: handler, env: key =>
        {
            if (key == "STINGRAY_ALLOW_EXTERNAL") return "0";
            return null;
        });

        await Assert.ThrowsAsync<ExternalAccessDeniedException>(async () =>
        {
            await Backlog.RunAsync(http, limit: 50, buildId: "test", CancellationToken.None);
        });
    }

    [Fact]
    public void LimitValidation_EnforcesPositiveLimit()
    {
        var settingsZero = new ScoutCommand.Settings { Backlog = true, Limit = 0 };
        Assert.Equal("--limit must be positive.", settingsZero.Validate());

        var settingsNegative = new ScoutCommand.Settings { Backlog = true, Limit = -10 };
        Assert.Equal("--limit must be positive.", settingsNegative.Validate());

        var settingsValid = new ScoutCommand.Settings { Backlog = true, Limit = 50 };
        Assert.Null(settingsValid.Validate());

        var settingsLimitWithoutBacklog = new ScoutCommand.Settings { ModelPath = "model.gguf", Limit = 50 };
        Assert.Equal("--limit only applies with --backlog.", settingsLimitWithoutBacklog.Validate());

        var settingsBacklogWithModel = new ScoutCommand.Settings { Backlog = true, ModelPath = "model.gguf" };
        Assert.Equal("--backlog queries the Hugging Face model list; drop -m and -r.", settingsBacklogWithModel.Validate());

        var settingsBacklogWithRepo = new ScoutCommand.Settings { Backlog = true, Repo = "org/repo" };
        Assert.Equal("--backlog queries the Hugging Face model list; drop -m and -r.", settingsBacklogWithRepo.Validate());
    }

    [Fact]
    public void ArchitectureRegistryMarking_AdmittedAndUnregisteredRules()
    {
        // 1. Admitted architecture -> 'admitted'
        Assert.Equal("admitted", Backlog.ResolveStatus("llama"));
        Assert.Equal("admitted", Backlog.ResolveStatus("qwen2"));

        // 2. Completely unknown architecture -> 'unregistered'
        Assert.Equal("unregistered", Backlog.ResolveStatus("unregistered_mystery_arch_999"));
        Assert.Equal("unregistered", Backlog.ResolveStatus(null));

        // 3. Ported-but-unverified architecture (CLAUDE.md rule 14: NotAdmitted and Experimental must NOT be printed as 'not admitted'; show as 'unregistered')
        // deepseek4 is registered in ArchitectureRegistry as AdmissionStatus.NotAdmitted
        Assert.Equal("unregistered", Backlog.ResolveStatus("deepseek4"));

        var exp = OpenTail.Stingray.Engine.ArchitectureRegistry.All.FirstOrDefault(d => d.Status == OpenTail.Stingray.Engine.AdmissionStatus.Experimental);
        if (exp is not null)
            Assert.Equal("unregistered", Backlog.ResolveStatus(exp.Id));
    }

    [Fact]
    public void ArchitectureLabeling_LabelsNoneAndClipCorrectly()
    {
        string json = """
            [
              { "id": "org/none-model", "downloads": 100, "gguf": { "architecture": "(none)" } },
              { "id": "org/missing-model", "downloads": 200, "gguf": {} },
              { "id": "org/clip-model", "downloads": 300, "gguf": { "architecture": "clip" } }
            ]
            """;

        var report = Backlog.Parse(json, limit: 50, buildId: "test", new NetworkUse(1, ["huggingface.co"], 500));

        var noArch = report.All.First(e => e.Architecture == "no architecture declared");
        Assert.Equal(2, noArch.RepoCount);
        Assert.Equal(300, noArch.Downloads30Days);

        var clip = report.All.First(e => e.Architecture == "projector");
        Assert.Equal(1, clip.RepoCount);
        Assert.Equal(300, clip.Downloads30Days);
    }

    [Fact]
    public async Task EndToEnd_BacklogRunAsync_FetchesAndParsesExpectedReport()
    {
        string json = """
            [
              { "id": "meta-llama/Llama-3-8B", "downloads": 500000, "gated": false, "gguf": { "architecture": "llama", "totalFileSize": 5000000000 } },
              { "id": "mystery/BrandNew-7B", "downloads": 200000, "gated": true, "gguf": { "architecture": "mystery_arch", "totalFileSize": 4000000000 } },
              { "id": "internal/DeepSeek4-Sample", "downloads": 100000, "gated": false, "gguf": { "architecture": "deepseek4", "totalFileSize": 3000000000 } }
            ]
            """;

        var handler = new FakeListHandler(req =>
        {
            Assert.Equal("huggingface.co", req.RequestUri!.Host);
            Assert.Equal("/api/models", req.RequestUri.AbsolutePath);
            Assert.Contains("filter=gguf", req.RequestUri.Query);
            Assert.Contains("pipeline_tag=text-generation", req.RequestUri.Query);
            Assert.Contains("sort=downloads", req.RequestUri.Query);

            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(json, Encoding.UTF8, "application/json")
            };
        });

        using var http = new ExternalHttpClient(userAgent: "test", inner: handler, env: key => key == "STINGRAY_ALLOW_EXTERNAL" ? "1" : null);

        var report = await Backlog.RunAsync(http, limit: 50, buildId: "1.0.0-test", CancellationToken.None);

        Assert.Equal(3, report.TotalReposInspected);

        // Backlog should contain mystery_arch and deepseek4 (both marked unregistered)
        Assert.Equal(2, report.Backlog.Count);
        Assert.Equal("mystery_arch", report.Backlog[0].Architecture);
        Assert.Equal("unregistered", report.Backlog[0].Status);
        Assert.Equal(200000, report.Backlog[0].Downloads30Days);

        Assert.Equal("deepseek4", report.Backlog[1].Architecture);
        Assert.Equal("unregistered", report.Backlog[1].Status);
        Assert.Equal(100000, report.Backlog[1].Downloads30Days);

        // Admitted should contain llama
        Assert.Single(report.Admitted);
        Assert.Equal("llama", report.Admitted[0].Architecture);
        Assert.Equal("admitted", report.Admitted[0].Status);
        Assert.Equal(500000, report.Admitted[0].Downloads30Days);

        // Text renderer should render without error
        BacklogTextRenderer.Write(report);
    }

    private sealed class FakeListHandler(Func<HttpRequestMessage, HttpResponseMessage> handle) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => Task.FromResult(handle(request));
    }
}
