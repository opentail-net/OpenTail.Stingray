using System.Collections.Concurrent;
using System.Diagnostics.Metrics;
using System.Globalization;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Plan 5 C2, OpenTelemetry plan section 12: <c>/metrics</c>, the <c>Meter</c> instruments and <c>/status</c> agree on request counts AND latency.
/// <c>ServingRequestTimingTests.MeterInstrumentsAndMetricsSummaries_AgreeOnCounts</c> covers counts at the recording level; this one goes through
/// the HTTP routes. Real chat requests are served, then the same quantities are read three ways: the Meter histogram measurements, the rendered
/// <c>/metrics</c> text, and the <c>/status</c> JSON. Runs in the non-parallel telemetry collection: the Meter is process-wide, so measurements
/// from tests running at the same time would be counted as extra requests.
/// </summary>
[Collection(ModelLoadWaitTelemetryCollection.Name)]
public sealed class ServerMetricsAgreementTests
{
    private const int Requests = 6;

    // An absolute tolerance, not Assert.Equal(precision): that rounds both sides first, and two values half a digit apart can land either side of a rounding boundary.
    private static void AssertWithin(double expected, double actual, double tolerance) =>
        Assert.True(Math.Abs(expected - actual) <= tolerance, $"expected {expected:F9}, got {actual:F9} (tolerance {tolerance})");

    [Fact]
    public async Task MetricsText_StatusJson_AndMeterMeasurements_AgreeOnCountsAndLatency()
    {
        var meterDurations = new ConcurrentBag<double>();
        var meterFirstTokens = new ConcurrentBag<double>();
        using var listener = new MeterListener();
        listener.InstrumentPublished = (i, l) =>
        {
            if (i.Meter.Name == "OpenTail.Stingray.Server"
                && i.Name is "gen_ai.server.request.duration" or "gen_ai.server.time_to_first_token") l.EnableMeasurementEvents(i);
        };
        listener.SetMeasurementEventCallback<double>((i, v, t, s) =>
        {
            if (i.Name == "gen_ai.server.request.duration") meterDurations.Add(v);
            else meterFirstTokens.Add(v);
        });
        listener.Start();

        using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(
                new FakeInferenceEngine("test-model", [(GenerateChunkKind.Text, "one two three")]))));
        using var client = factory.CreateClient();
        for (int i = 0; i < Requests; i++)
        {
            using var response = await client.PostAsJsonAsync("/v1/chat/completions", new
            {
                model = "test-model", max_tokens = 8, messages = new[] { new { role = "user", content = "hi" } },
            });
            Assert.True(response.IsSuccessStatusCode);
        }

        string metrics = await client.GetStringAsync("/metrics");
        using var status = JsonDocument.Parse(await client.GetStringAsync("/status"));
        var latency = status.RootElement.GetProperty("latency");

        (long Count, double Seconds) Prometheus(string name) => (
            long.Parse(Regex.Match(metrics, "^" + name + "_bucket\\{le=\"\\+Inf\"\\} (\\d+)$", RegexOptions.Multiline).Groups[1].Value, CultureInfo.InvariantCulture),
            double.Parse(Regex.Match(metrics, "^" + name + "_sum ([0-9.]+)$", RegexOptions.Multiline).Groups[1].Value, CultureInfo.InvariantCulture));
        (long Count, double Seconds) Status(string section) => (
            latency.GetProperty(section).GetProperty("count").GetInt64(),
            latency.GetProperty(section).GetProperty("total_seconds").GetDouble());

        // Request duration: the Meter, /metrics and /status are one measurement read three ways.
        var requestMetrics = Prometheus("opentail_llm_request_generation_duration_seconds");
        var requestStatus = Status("request");
        Assert.Equal(Requests, meterDurations.Count);
        Assert.Equal(Requests, requestMetrics.Count);
        Assert.Equal(Requests, requestStatus.Count);
        AssertWithin(meterDurations.Sum(), requestStatus.Seconds, 1e-9);
        AssertWithin(requestStatus.Seconds, requestMetrics.Seconds, 1e-6);   // /metrics prints six decimals: within half of the last digit

        // Time to first token, the same way.
        var ttftMetrics = Prometheus("opentail_llm_time_to_first_token_seconds");
        var ttftStatus = Status("time_to_first_token");
        Assert.Equal(Requests, meterFirstTokens.Count);
        Assert.Equal(Requests, ttftMetrics.Count);
        Assert.Equal(Requests, ttftStatus.Count);
        AssertWithin(meterFirstTokens.Sum(), ttftStatus.Seconds, 1e-9);
        AssertWithin(ttftStatus.Seconds, ttftMetrics.Seconds, 1e-6);

        // Not vacuous: time really was measured.
        Assert.True(requestStatus.Seconds > 0);
    }
}
