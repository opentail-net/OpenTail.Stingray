using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Server;
using Xunit;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Plan 2 P5, server half. Sends unique canary text through the chat endpoints (system message, user prompt,
/// tool definition, completion, and a malformed body) and captures EVERY metric, span, span event and span link
/// the server emits. Two assertions:
/// <list type="number">
/// <item>No canary text appears in any captured string, and no captured attribute key is outside
/// <see cref="TelemetryAttributePolicy"/> (plan section 7).</item>
/// <item>Not vacuous: the capture holds the server's request metrics and its chat spans.</item>
/// </list>
/// <para>Limit: the fake engine emits text, not tool calls, so tool-call ARGUMENTS are not produced here. The tool
/// canary sits in the tool DEFINITION, which the server does echo into request handling. A tool-call output canary
/// needs an engine that emits tool calls, and is a follow-up.</para>
/// </summary>
public sealed class ServerTelemetryCanaryTests
{
    private const string SystemCanary = "CANARY-SYSTEM-7f3a91";
    private const string PromptCanary = "CANARY-PROMPT-7f3a91";
    private const string ToolCanary = "CANARY-TOOL-7f3a91";
    private const string CompletionCanary = "CANARY-COMPLETION-7f3a91";
    private const string PathCanary = @"C:\Users\canary-user\secret-project\model.gguf";
    private const string Model = "test-model";

    private static void Collect(List<string> sink, string where, string? value)
    {
        if (value is not null) lock (sink) sink.Add($"{where}={value}");
    }

    [Fact]
    public async Task CanaryTextAndUnlistedKeys_NeverAppearInAnyServerTelemetry()
    {
        var captured = new List<string>();
        using var meters = ListenToServerMeters(captured);
        using var spans = ListenToServerSpans(captured);

        using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(
                new FakeInferenceEngine(Model, [(GenerateChunkKind.Text, CompletionCanary)]))));
        using var client = factory.CreateClient();

        // Streaming and non-streaming, each carrying every canary.
        using (var streamed = await client.PostAsJsonAsync("/v1/chat/completions", ChatWithCanaries(stream: true)))
            Assert.Equal(HttpStatusCode.OK, streamed.StatusCode);   // drains: the SSE body must finish for the span to stop
        using (var plain = await client.PostAsJsonAsync("/v1/chat/completions", ChatWithCanaries(stream: false)))
            Assert.Equal(HttpStatusCode.OK, plain.StatusCode);

        // A malformed body that still carries the prompt canary, so the error path is covered too.
        string malformed = "{\"model\":\"" + Model + "\",\"messages\":[{\"role\":\"user\",\"content\":\"" + PromptCanary + "\"";
        using (var content = new StringContent(malformed, System.Text.Encoding.UTF8, "application/json"))
        using (await client.PostAsync("/v1/chat/completions", content)) { }

        string[] seen;
        lock (captured) seen = [.. captured];

        // Not vacuous: the server really emitted its request telemetry and its chat spans.
        Assert.True(seen.Length > 20, $"only {seen.Length} telemetry strings captured");
        Assert.Contains(seen, s => s.StartsWith("span name=chat ", StringComparison.Ordinal));
        Assert.Contains(seen, s => s.StartsWith("metric tag key=", StringComparison.Ordinal));

        // Privacy: no canary text, no paths, in any captured string, whatever field carried it.
        foreach (string canary in new[] { SystemCanary, PromptCanary, ToolCanary, CompletionCanary })
            Assert.DoesNotContain(seen, s => s.Contains(canary, StringComparison.OrdinalIgnoreCase));
        Assert.DoesNotContain(seen, s => s.Contains("canary", StringComparison.OrdinalIgnoreCase));
        // Unique tokens only. Other tests in this process emit model file names (for example "smol.gguf", a fake
        // engine id), so a generic ".gguf" or "C:\" match would fail on unrelated telemetry.
        Assert.DoesNotContain(seen, s => s.Contains(PathCanary, StringComparison.OrdinalIgnoreCase));

        // Section 7 names: every attribute key is on the allow-list.
        string[] keys = seen.Select(AttributeKeyOf).Where(k => k is not null).Select(k => k!).Distinct(StringComparer.Ordinal).ToArray();
        string[] unexpected = keys.Where(k => !TelemetryAttributePolicy.AllowedAttributeKeys.Contains(k)).Order(StringComparer.Ordinal).ToArray();
        Assert.True(unexpected.Length == 0,
            "attribute keys outside the section 7 allow-list (review, then add to TelemetryAttributePolicy): " + string.Join(", ", unexpected));
    }

    private static object ChatWithCanaries(bool stream) => new
    {
        model = Model,
        stream,
        max_tokens = 16,
        messages = new object[]
        {
            new { role = "system", content = SystemCanary },
            new { role = "user", content = PromptCanary },
        },
        tools = new object[]
        {
            new
            {
                type = "function",
                function = new
                {
                    name = "lookup",
                    description = ToolCanary,
                    parameters = new { type = "object", properties = new { q = new { type = "string" } } },
                },
            },
        },
    };

    private static MeterListener ListenToServerMeters(List<string> sink)
    {
        var l = new MeterListener();
        l.InstrumentPublished = (i, ml) =>
        {
            if (i.Meter.Name != "OpenTail.Stingray.Server") return;
            Collect(sink, "instrument", i.Name);
            Collect(sink, "unit", i.Unit);
            ml.EnableMeasurementEvents(i);
        };
        void Tags(ReadOnlySpan<KeyValuePair<string, object?>> t)
        {
            foreach (var kv in t) { Collect(sink, "metric tag key", kv.Key); Collect(sink, "metric tag value", kv.Value?.ToString()); }
        }
        l.SetMeasurementEventCallback<long>((i, v, t, s) => Tags(t));
        l.SetMeasurementEventCallback<double>((i, v, t, s) => Tags(t));
        l.SetMeasurementEventCallback<int>((i, v, t, s) => Tags(t));
        l.Start();
        return l;
    }

    private static ActivityListener ListenToServerSpans(List<string> sink)
    {
        void Dump(Activity a)
        {
            Collect(sink, "span name", a.OperationName);
            Collect(sink, "span display name", a.DisplayName);
            Collect(sink, "span status", a.StatusDescription);
            foreach (var t in a.TagObjects) { Collect(sink, "span tag key", t.Key); Collect(sink, "span tag value", t.Value?.ToString()); }
            foreach (var e in a.Events)
            {
                Collect(sink, "span event", e.Name);
                foreach (var t in e.Tags) { Collect(sink, "span event tag key", t.Key); Collect(sink, "span event tag value", t.Value?.ToString()); }
            }
            foreach (var link in a.Links) foreach (var t in link.Tags ?? []) Collect(sink, "span link tag", $"{t.Key}={t.Value}");
            foreach (var b in a.Baggage) Collect(sink, "baggage", $"{b.Key}={b.Value}");
        }
        var al = new ActivityListener
        {
            ShouldListenTo = s => s.Name == ServerMetrics.ActivitySourceName,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllData,
            ActivityStopped = Dump,
        };
        ActivitySource.AddActivityListener(al);
        return al;
    }

    /// <summary>The attribute key of a captured <c>metric tag key=</c>, <c>span tag key=</c>, <c>span event tag key=</c> or <c>span link tag=</c> string; null for any other string.</summary>
    private static string? AttributeKeyOf(string captured)
    {
        string? rest =
            captured.StartsWith("metric tag key=", StringComparison.Ordinal) ? captured["metric tag key=".Length..] :
            captured.StartsWith("span tag key=", StringComparison.Ordinal) ? captured["span tag key=".Length..] :
            captured.StartsWith("span event tag key=", StringComparison.Ordinal) ? captured["span event tag key=".Length..] :
            captured.StartsWith("span link tag=", StringComparison.Ordinal) ? captured["span link tag=".Length..] :
            null;
        if (rest is null) return null;
        int eq = rest.IndexOf('=');
        return eq >= 0 ? rest[..eq] : rest;
    }
}
