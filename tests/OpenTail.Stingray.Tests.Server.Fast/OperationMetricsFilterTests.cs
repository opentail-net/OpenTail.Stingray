using System.Diagnostics.Metrics;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using OpenTail.Stingray.Server.Endpoints;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>The non-chat operation wrapper: audio seconds and real-time factor, and error.type as a type name only.</summary>
public sealed class OperationMetricsFilterTests
{
    private static async Task<HttpClient> StartAsync(IHost[] hostOut)
    {
        var host = await new HostBuilder().ConfigureWebHost(web => web
            .UseTestServer()
            .ConfigureServices(s => { s.AddRouting(); s.AddSingleton<ServerMetrics>(); })
            .Configure(app =>
            {
                app.UseRouting();
                app.UseEndpoints(e =>
                {
                    e.MapPost("/audio", async (HttpContext c) =>
                    {
                        await Task.Delay(20);
                        c.Items[OperationMetricsFilterExtensions.AudioSecondsItem] = 4.0;
                        await c.Response.WriteAsync("ok");
                    }).WithOperationMetrics("speech", "test.filter_audio");
                    e.MapPost("/boom", (HttpContext c) => throw new InvalidOperationException("secret prompt text")).WithOperationMetrics("embeddings", "test.filter_boom");
                });
            })).StartAsync();
        hostOut[0] = host;
        return host.GetTestClient();
    }

    [Fact]
    public async Task Audio_RecordsDurationAndRealtimeFactor_AndFailuresRecordOnlyTheTypeName()
    {
        var samples = new List<(string Name, double Value, Dictionary<string, object?> Tags)>();
        using var listener = new MeterListener();
        listener.InstrumentPublished = (i, l) => { if (i.Meter.Name == "OpenTail.Stingray.Server") l.EnableMeasurementEvents(i); };
        listener.SetMeasurementEventCallback<double>((i, v, t, s) =>
        {
            var tags = new Dictionary<string, object?>();
            foreach (var kv in t) tags[kv.Key] = kv.Value;
            if (tags.TryGetValue("opentail.endpoint", out var ep) && ep is string e && e.StartsWith("test.filter_", StringComparison.Ordinal))
                lock (samples) samples.Add((i.Name, v, tags));
        });
        listener.Start();

        var holder = new IHost[1];
        var client = await StartAsync(holder);
        using var host = holder[0];
        Assert.Equal(System.Net.HttpStatusCode.OK, (await client.PostAsync("/audio", new StringContent(""))).StatusCode);
        await Assert.ThrowsAsync<InvalidOperationException>(() => client.PostAsync("/boom", new StringContent("")));
        listener.Dispose();

        lock (samples)
        {
            var audio = Assert.Single(samples, s => s.Name == "opentail.audio.duration");
            Assert.Equal(4.0, audio.Value);
            var rtf = Assert.Single(samples, s => s.Name == "opentail.audio.realtime_factor");
            Assert.InRange(rtf.Value, 0.004, 0.5);   // about 0.02 s of work for 4 s of audio
            var failure = Assert.Single(samples, s => s.Name == "gen_ai.server.request.duration" && (string?)s.Tags["opentail.endpoint"] == "test.filter_boom");
            Assert.Equal("error", failure.Tags["outcome"]);
            Assert.Equal("InvalidOperationException", failure.Tags["error.type"]);
            Assert.DoesNotContain(samples, s => s.Tags.Values.Any(v => v is string str && str.Contains("secret", StringComparison.Ordinal)));
            var ok = Assert.Single(samples, s => s.Name == "gen_ai.server.request.duration" && (string?)s.Tags["opentail.endpoint"] == "test.filter_audio");
            Assert.False(ok.Tags.ContainsKey("error.type"));
        }
    }
}
