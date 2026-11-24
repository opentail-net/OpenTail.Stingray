using System.Text.Json;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>The STINGRAY_STATS_JSON result file is a versioned contract that benchmark scripts parse.</summary>
public sealed class RunStatsJsonSinkTests
{
    [Fact]
    public void Render_ProducesTheVersionedContract_FromEngineEvents()
    {
        var sink = new RunStatsJsonSink("unused.json");
        sink.OnEvent(new(EngineEventKind.ModelLoaded, "llama", Ms: 600, Detail: "forwardPass=VulkanHybrid"));
        sink.OnEvent(new(EngineEventKind.PrefillCompleted, A: 100, Ms: 500));
        sink.OnEvent(new(EngineEventKind.DecodeCompleted, A: 20, B: 18, Ms: 400));
        sink.OnEvent(new(EngineEventKind.MtpStep, A: 5, B: 3, C: 2));
        sink.OnEvent(new(EngineEventKind.MtpStep, A: 8, B: 3, C: 3));

        using var doc = JsonDocument.Parse(sink.Render());
        var r = doc.RootElement;
        Assert.Equal(1, r.GetProperty("schema_version").GetInt32());
        Assert.Equal("llama", r.GetProperty("architecture").GetString());
        Assert.Equal("VulkanHybrid", r.GetProperty("forward_pass").GetString());
        Assert.Equal(600, r.GetProperty("load_ms").GetDouble());
        Assert.Equal(100, r.GetProperty("prompt_tokens").GetDouble());
        Assert.Equal(200, r.GetProperty("prefill_tokens_per_second").GetDouble());   // 100 tokens / 0.5 s
        Assert.Equal(20, r.GetProperty("decode_tokens").GetDouble());
        Assert.Equal(18, r.GetProperty("visible_tokens").GetDouble());
        Assert.Equal(50, r.GetProperty("decode_tokens_per_second").GetDouble());      // 20 tokens / 0.4 s
        Assert.Equal(2, r.GetProperty("speculative_steps").GetDouble());
        Assert.Equal(6, r.GetProperty("speculative_drafts_proposed").GetDouble());
        Assert.Equal(5, r.GetProperty("speculative_drafts_accepted").GetDouble());
        Assert.False(r.GetProperty("perturbed").GetBoolean());
    }

    [Fact]
    public void Render_OmitsFieldsWhoseEventNeverHappened_AndNeverContainsContentOrPaths()
    {
        var sink = new RunStatsJsonSink("C:\\secret\\place\\out.json");
        using var doc = JsonDocument.Parse(sink.Render());
        var names = doc.RootElement.EnumerateObject().Select(p => p.Name).Order().ToArray();
        Assert.Equal(new[] { "perturbed", "schema_version" }, names);
        Assert.DoesNotContain("secret", sink.Render());
    }

    [Fact]
    public void Render_ReportsPerturbed_WhenAnyDiagnosticSwitchChangesWhatIsMeasured()
    {
        // A benchmark script refuses a run whose result says perturbed=true. Any diagnostic switch can add syncs, prints or dumps,
        // so each one must flag the run: this one (STINGRAY_PROFILE_DECODE) was not checked before, so the flag was false.
        const string name = "STINGRAY_PROFILE_DECODE";
        string? before = Environment.GetEnvironmentVariable(name);
        try
        {
            Environment.SetEnvironmentVariable(name, "1");
            using var doc = JsonDocument.Parse(new RunStatsJsonSink("unused.json").Render());
            Assert.True(doc.RootElement.GetProperty("perturbed").GetBoolean(), $"{name}=1 must mark the run perturbed");
        }
        finally { Environment.SetEnvironmentVariable(name, before); }
    }

    [Fact]
    public void Render_DoesNotReportPerturbed_ForAnOutputPathOrAZeroValue()
    {
        // STINGRAY_STATS_JSON only names the result file, and a switch set to "0" is off.
        const string stats = "STINGRAY_STATS_JSON";
        const string profile = "STINGRAY_PROFILE_PREFILL";
        string? statsBefore = Environment.GetEnvironmentVariable(stats);
        string? profileBefore = Environment.GetEnvironmentVariable(profile);
        try
        {
            Environment.SetEnvironmentVariable(stats, "unused.json");
            Environment.SetEnvironmentVariable(profile, "0");
            using var doc = JsonDocument.Parse(new RunStatsJsonSink("unused.json").Render());
            Assert.False(doc.RootElement.GetProperty("perturbed").GetBoolean());
        }
        finally
        {
            Environment.SetEnvironmentVariable(stats, statsBefore);
            Environment.SetEnvironmentVariable(profile, profileBefore);
        }
    }

    [Fact]
    public void Render_CarriesTheQwenCounters_OnlyWhenAForwardCountersEventArrived()
    {
        var sink = new RunStatsJsonSink("unused.json");
        using (var before = JsonDocument.Parse(sink.Render()))
            Assert.False(before.RootElement.TryGetProperty("ple_gather_rows", out _));

        sink.OnEvent(new(EngineEventKind.ForwardCounters, "qwen4exp", A: 42, B: 67_584, C: 28_800_138_240));

        using var doc = JsonDocument.Parse(sink.Render());
        var r = doc.RootElement;
        Assert.Equal(42, r.GetProperty("ple_gather_rows").GetDouble());
        Assert.Equal(67_584, r.GetProperty("ple_gather_bytes").GetDouble());
        Assert.Equal(28_800_138_240, r.GetProperty("ple_table_bytes").GetDouble());
    }
}
