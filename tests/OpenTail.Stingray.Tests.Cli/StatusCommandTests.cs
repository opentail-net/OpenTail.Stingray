using OpenTail.Stingray.Cli;

namespace OpenTail.Stingray.Tests.Cli;

public sealed class StatusCommandTests
{
    [Fact]
    public void OfflineServer_ReturnsOfflineSnapshot()
    {
        using var client = new HttpClient();
        var snapshot = StatusCommand.QueryStatus(client, "http://127.0.0.1:59999", CancellationToken.None);

        Assert.Equal("offline", snapshot.Status);
        Assert.NotNull(snapshot.Error);
        Assert.Contains("59999", snapshot.Error);
    }

    [Fact]
    public void OfflineServer_ReturnsNonZeroExitCode()
    {
        var command = (OpenTail.Stingray.Cli.CommandLine.ICommand)new StatusCommand();
        int exit = command.Run(["--url", "http://127.0.0.1:59999"], CancellationToken.None);
        Assert.Equal(1, exit);
    }

    [Fact]
    public void ParseMetrics_CorrectlyParsesRealServerPrometheusPayload()
    {
        string serverMetricsPayload = """
            # HELP opentail-llm_requests_total Total inference requests served
            # TYPE opentail-llm_requests_total counter
            opentail-llm_requests_total 42
            # HELP opentail-llm_tokens_generated_total Total tokens generated
            # TYPE opentail-llm_tokens_generated_total counter
            opentail-llm_tokens_generated_total 1337
            # HELP opentail-llm_uptime_seconds Server uptime in seconds
            # TYPE opentail-llm_uptime_seconds gauge
            opentail-llm_uptime_seconds 3600
            # HELP opentail-llm_tokens_per_second Lifetime-average tokens generated per second
            # TYPE opentail-llm_tokens_per_second gauge
            opentail-llm_tokens_per_second 28.5
            # HELP opentail-llm_queue_depth Number of requests waiting to start generation
            # TYPE opentail-llm_queue_depth gauge
            opentail-llm_queue_depth 3
            # HELP opentail-llm_active_requests Number of requests currently generating tokens
            # TYPE opentail-llm_active_requests gauge
            opentail-llm_active_requests 5
            # HELP opentail_llm_batch_kv_committed_tokens KV tokens reserved by admitted requests
            # TYPE opentail_llm_batch_kv_committed_tokens gauge
            opentail_llm_batch_kv_committed_tokens 4096
            # HELP opentail_llm_batch_kv_token_budget KV-token admission budget
            # TYPE opentail_llm_batch_kv_token_budget gauge
            opentail_llm_batch_kv_token_budget 16384
            """;

        double requests = 0, tokens = 0, tokSec = 0, activeSeq = 0, queued = 0, kvPct = 0;
        StatusCommand.ParseMetrics(serverMetricsPayload, ref requests, ref tokens, ref tokSec, ref activeSeq, ref queued, ref kvPct);

        Assert.Equal(42, requests);
        Assert.Equal(1337, tokens);
        Assert.Equal(28.5, tokSec);
        Assert.Equal(5, activeSeq);
        Assert.Equal(3, queued);
        Assert.Equal(25.0, kvPct); // 4096 / 16384 * 100 = 25%
    }

    /// <summary>
    /// The /health payload is serialised from `record HealthStatus(string Status, string Model,
    /// long UptimeSeconds)`, so ASP.NET emits camelCase `uptimeSeconds`. An earlier version read
    /// `uptime_seconds` and silently reported 0 uptime against a healthy server. This pins the
    /// real casing; the snake_case case is accepted too, so a future serializer change either way
    /// keeps working.
    /// </summary>
    [Theory]
    [InlineData("{\"status\":\"ok\",\"model\":\"smol.gguf\",\"uptimeSeconds\":3600}")]
    [InlineData("{\"status\":\"ok\",\"model\":\"smol.gguf\",\"uptime_seconds\":3600}")]
    [InlineData("{\"Status\":\"ok\",\"Model\":\"smol.gguf\",\"UptimeSeconds\":3600}")]
    public void HealthPayload_UptimeIsParsedRegardlessOfCasingStyle(string payload)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(payload);
        var root = doc.RootElement;

        Assert.True(StatusCommand.TryGetPropertyCaseInsensitive(root, "status", out var status));
        Assert.Equal("ok", status.GetString());
        Assert.True(StatusCommand.TryGetPropertyCaseInsensitive(root, "model", out var model));
        Assert.Equal("smol.gguf", model.GetString());

        bool found = StatusCommand.TryGetPropertyCaseInsensitive(root, "uptimeSeconds", out var uptime)
                     || StatusCommand.TryGetPropertyCaseInsensitive(root, "uptime_seconds", out uptime);
        Assert.True(found, "uptime was not found in the health payload");
        Assert.Equal(3600d, uptime.GetDouble());
    }

    /// <summary>
    /// A non-numeric uptime must not throw — the ValueKind guard exists for exactly this, and a
    /// malformed field should degrade the display rather than kill the command.
    /// </summary>
    [Fact]
    public void HealthPayload_NonNumericUptimeIsIgnoredRatherThanThrowing()
    {
        using var doc = System.Text.Json.JsonDocument.Parse("{\"uptimeSeconds\":\"soon\"}");
        Assert.True(StatusCommand.TryGetPropertyCaseInsensitive(doc.RootElement, "uptimeSeconds", out var v));
        Assert.NotEqual(System.Text.Json.JsonValueKind.Number, v.ValueKind);
    }

    /// <summary>
    /// A representative <c>/status</c> document as the server emits it (snake_case, nulls
    /// omitted). <c>StatusDocumentTests.Status_ContainsEveryFieldTheCliStatusRendererReads</c>
    /// on the server side guards these names against drift.
    /// </summary>
    private const string StatusDocument = """
        {
          "schema_version": 1,
          "status": "ok",
          "model": "smol.gguf",
          "architecture": "qwen3",
          "uptime_seconds": 3600,
          "placement": {
            "backend": "cuda",
            "gpu_layers": -1,
            "context_size": 8192,
            "max_batch_size": 4,
            "continuous_batching": true,
            "spec_type": "none"
          },
          "traffic": {
            "requests_total": 42,
            "tokens_generated_total": 1337,
            "tokens_per_second": 28.5,
            "active_requests": 5,
            "queued_requests": 3,
            "admission_capacity": 20,
            "overload_rejections_total": 0
          },
          "latency": {
            "queue": { "count": 42, "total_seconds": 0.42, "mean_ms": 10, "p50_ms": 10, "p95_ms": 25 },
            "time_to_first_token": { "count": 42, "total_seconds": 8.4, "mean_ms": 200, "p50_ms": 250, "p95_ms": 500 },
            "generation": { "count": 42, "total_seconds": 40, "mean_ms": 952 },
            "request": { "count": 42, "total_seconds": 42, "mean_ms": 1000 },
            "inter_token_mean_ms": 30.9
          },
          "cache": {
            "prefix_cache_enabled": true,
            "prefill_tokens_reused_total": 12345,
            "hits": 3,
            "misses": 1,
            "hit_rate_percent": 75.0
          },
          "memory": {
            "managed_heap_bytes": 52428800,
            "gc_committed_bytes": 104857600,
            "working_set_bytes": 8589934592,
            "host_available_bytes": 68719476736
          },
          "batching": {
            "prefill_chunk_tokens": 256,
            "committed_kv_tokens": 4096,
            "kv_token_budget": 16384,
            "kv_occupancy_percent": 25.0
          },
          "warnings": ["KV admission budget is 25% committed."]
        }
        """;

    [Fact]
    public void StatusDocument_SuppliesEverythingTheMetricsScrapeCannot()
    {
        Assert.True(StatusCommand.TryParseStatusDocument(StatusDocument, "http://host:8080", out var s));

        Assert.Equal(1, s.SchemaVersion);
        Assert.Equal("ok", s.Status);
        Assert.Equal("smol.gguf", s.Model);
        Assert.Equal("http://host:8080", s.Url);
        Assert.Equal(3600, s.UptimeSeconds);
        Assert.Equal(42, s.RequestsTotal);
        Assert.Equal(1337, s.TokensGeneratedTotal);
        Assert.Equal(28.5, s.TokensPerSecond);
        Assert.Equal(5, s.ActiveSequences);
        Assert.Equal(3, s.QueuedRequests);
        Assert.Equal(25.0, s.KvOccupancyPercent);

        // The rows /health + /metrics could never fill in.
        Assert.Equal("cuda", s.Backend);
        Assert.Equal(8192, s.ContextSize);
        Assert.Equal(250, s.TimeToFirstTokenMs);
        Assert.Equal(30.9, s.InterTokenLatencyMs);
        Assert.True(s.PrefixCacheEnabled);
        Assert.Equal(12345, s.PrefillTokensReused);
        Assert.Equal(75.0, s.PrefixCacheHitRatePercent);
        Assert.Equal(8589934592d, s.WorkingSetBytes);
        Assert.Equal(52428800d, s.ManagedHeapBytes);
        Assert.Equal(["KV admission budget is 25% committed."], s.Warnings ?? []);
        Assert.Null(s.Error);
    }

    /// <summary>
    /// An unknown schema version must be rejected outright rather than parsed optimistically:
    /// the caller then falls back to /health + /metrics, which is strictly better than rendering
    /// fields whose meaning may have changed.
    /// </summary>
    [Theory]
    [InlineData("""{"schema_version": 2, "status": "ok", "model": "x"}""")]
    [InlineData("""{"status": "ok", "model": "x"}""")]
    [InlineData("""{"schema_version": "1"}""")]
    [InlineData("[1,2,3]")]
    [InlineData("not json at all")]
    public void StatusDocument_UnrecognisedPayloadIsRejectedSoTheScrapeFallbackRuns(string payload)
    {
        Assert.False(StatusCommand.TryParseStatusDocument(payload, "http://host:8080", out _));
    }

    /// <summary>
    /// The server omits null properties, so a field the host could not determine is simply
    /// absent. Those must stay null rather than render as a confident zero — an unknown
    /// time-to-first-token displayed as "0 ms" reads as an instantaneous server.
    /// </summary>
    [Fact]
    public void StatusDocument_AbsentOptionalFieldsRemainUnknown()
    {
        string minimal = """
            {
              "schema_version": 1,
              "status": "ok",
              "model": "smol.gguf",
              "uptime_seconds": 5,
              "traffic": { "requests_total": 0, "tokens_generated_total": 0 },
              "latency": { "queue": { "count": 0, "mean_ms": 0 } },
              "cache": { "prefix_cache_enabled": false, "prefill_tokens_reused_total": 0 },
              "memory": { "managed_heap_bytes": 1024 }
            }
            """;

        Assert.True(StatusCommand.TryParseStatusDocument(minimal, "http://host:8080", out var s));

        Assert.Null(s.TimeToFirstTokenMs);
        Assert.Null(s.InterTokenLatencyMs);
        Assert.Null(s.Backend);
        Assert.Null(s.ContextSize);
        Assert.Null(s.PrefixCacheHitRatePercent);
        Assert.Null(s.WorkingSetBytes);
        Assert.Null(s.Warnings);
        Assert.False(s.PrefixCacheEnabled);
        Assert.Equal(0, s.KvOccupancyPercent);
        Assert.Equal(1024d, s.ManagedHeapBytes);
    }

    /// <summary>
    /// A server that reports latency counts but no p50 (every sample above the largest histogram
    /// bucket) still has a usable mean. Falling back to it beats showing nothing.
    /// </summary>
    [Fact]
    public void StatusDocument_TimeToFirstTokenFallsBackToTheMeanWhenNoPercentileExists()
    {
        string payload = """
            {
              "schema_version": 1,
              "status": "ok",
              "model": "smol.gguf",
              "latency": { "time_to_first_token": { "count": 3, "mean_ms": 92000 } }
            }
            """;

        Assert.True(StatusCommand.TryParseStatusDocument(payload, "http://host:8080", out var s));
        Assert.Equal(92000, s.TimeToFirstTokenMs);
    }

    /// <summary>
    /// The offline path predates the status document and must keep working unchanged: it is what
    /// reports a connection failure. Its document-only fields stay null.
    /// </summary>
    [Fact]
    public void OfflineServer_LeavesStatusDocumentFieldsUnknown()
    {
        using var client = new HttpClient();
        var snapshot = StatusCommand.QueryStatus(client, "http://127.0.0.1:59999", CancellationToken.None);

        Assert.Null(snapshot.SchemaVersion);
        Assert.Null(snapshot.Backend);
        Assert.Null(snapshot.TimeToFirstTokenMs);
        Assert.Null(snapshot.PrefixCacheEnabled);
        Assert.NotNull(snapshot.Error);
    }
}
