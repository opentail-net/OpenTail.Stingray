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
}
