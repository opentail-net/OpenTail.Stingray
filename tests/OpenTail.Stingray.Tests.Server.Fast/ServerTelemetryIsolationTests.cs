using System.Collections.Concurrent;
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
/// Plan 5 C2, OpenTelemetry plan section 12: concurrent requests must not cross-contaminate trace context or measurements,
/// and one chat request is one connected trace. Eight chat requests are sent at once, each with its own W3C
/// <c>traceparent</c>. Each trace id must end up with exactly one <c>chat</c> span, parented to that request's own HTTP span,
/// and the request-duration histogram must hold exactly one successful sample per request.
/// Runs in the non-parallel telemetry collection: the meter and activity listeners are process-wide.
/// </summary>
[Collection(ModelLoadWaitTelemetryCollection.Name)]
public sealed class ServerTelemetryIsolationTests
{
    private const string Model = "isolation-model";
    private const int Requests = 8;

    [Fact]
    public async Task ConcurrentChatRequests_EachGetOneConnectedTrace_AndExactlyOneMeasurement()
    {
        var stopped = new ConcurrentBag<Activity>();
        using var spans = new ActivityListener
        {
            ShouldListenTo = s => s.Name == ServerMetrics.ActivitySourceName || s.Name == "Microsoft.AspNetCore",
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllData,
            ActivityStopped = stopped.Add,
        };
        ActivitySource.AddActivityListener(spans);

        var durations = new ConcurrentBag<string?>();
        using var meters = new MeterListener();
        meters.InstrumentPublished = (i, l) =>
        {
            if (i.Meter.Name == "OpenTail.Stingray.Server" && i.Name == "gen_ai.server.request.duration") l.EnableMeasurementEvents(i);
        };
        meters.SetMeasurementEventCallback<double>((i, v, tags, s) =>
        {
            string? outcome = null;
            foreach (var kv in tags) if (kv.Key == "outcome") outcome = kv.Value as string;
            durations.Add(outcome);
        });
        meters.Start();

        using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(
                new FakeInferenceEngine(Model, [(GenerateChunkKind.Text, "ok")]))));
        using var client = factory.CreateClient();

        var traceIds = Enumerable.Range(0, Requests).Select(_ => ActivityTraceId.CreateRandom()).ToArray();
        var responses = await Task.WhenAll(traceIds.Select(async id =>
        {
            using var request = new HttpRequestMessage(HttpMethod.Post, "/v1/chat/completions")
            {
                Content = JsonContent.Create(new { model = Model, max_tokens = 4, messages = new[] { new { role = "user", content = "hi" } } }),
            };
            request.Headers.Add("traceparent", $"00-{id}-{ActivitySpanId.CreateRandom()}-01");
            using var response = await client.SendAsync(request);
            return response.StatusCode;
        }));
        Assert.All(responses, code => Assert.Equal(HttpStatusCode.OK, code));

        var all = stopped.ToArray();
        foreach (var id in traceIds)
        {
            var chat = Assert.Single(all, a => a.Source.Name == ServerMetrics.ActivitySourceName
                && a.OperationName.StartsWith("chat", StringComparison.Ordinal) && a.TraceId == id);
            var parent = Assert.Single(all, a => a.Source.Name == "Microsoft.AspNetCore" && a.Id == chat.ParentId);
            Assert.Equal(id, parent.TraceId);
        }

        Assert.Equal(Requests, durations.Count);
        Assert.All(durations, outcome => Assert.Equal("success", outcome));
    }
}
