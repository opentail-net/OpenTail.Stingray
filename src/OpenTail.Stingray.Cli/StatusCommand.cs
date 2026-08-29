
namespace OpenTail.Stingray.Cli;

/// <summary>
/// Diagnostic status command (§6.5 &amp; §7.4 of QoL plan):
/// <c>opentail-llm-cli status [--url &lt;URL&gt;] [--watch] [--json]</c>
///
/// Prefers the server's versioned <c>/status</c> document, which carries placement, serving
/// latency (TTFT and inter-token), prefix-cache reuse, memory and derived warnings in one
/// request. Falls back to scraping <c>/health</c> plus <c>/metrics</c> when the host does not
/// map <c>/status</c> — an older server, or one that mapped only a subset of OpenTail's routes.
/// </summary>
public sealed class StatusCommand : Command<StatusCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-u|--url <URL>")]
        [Description("Server URL (default: http://127.0.0.1:8080)")]
        public string Url { get; init; } = "http://127.0.0.1:8080";

        [CommandOption("-w|--watch")]
        [Description("Continuously refresh status every second")]
        public bool Watch { get; init; }

        [CommandOption("--json")]
        [Description("Write machine-readable JSON snapshot to stdout")]
        public bool Json { get; init; }
    }

    public sealed record StatusSnapshot(
        [property: JsonPropertyName("status")] string Status,
        [property: JsonPropertyName("url")] string Url,
        [property: JsonPropertyName("model")] string Model,
        [property: JsonPropertyName("uptime_seconds")] double UptimeSeconds,
        [property: JsonPropertyName("requests_total")] double RequestsTotal,
        [property: JsonPropertyName("tokens_generated_total")] double TokensGeneratedTotal,
        [property: JsonPropertyName("tokens_per_second")] double TokensPerSecond,
        [property: JsonPropertyName("active_sequences")] double ActiveSequences,
        [property: JsonPropertyName("queued_requests")] double QueuedRequests,
        [property: JsonPropertyName("kv_occupancy_percent")] double KvOccupancyPercent,
        [property: JsonPropertyName("error")] string? Error = null,
        // Everything below comes from the versioned /status document only. They stay null on the
        // /health + /metrics fallback path, which is how the renderer knows to omit those rows
        // rather than print a confident zero for something the server never reported.
        [property: JsonPropertyName("schema_version")] int? SchemaVersion = null,
        [property: JsonPropertyName("backend")] string? Backend = null,
        [property: JsonPropertyName("context_size")] int? ContextSize = null,
        [property: JsonPropertyName("time_to_first_token_ms")] double? TimeToFirstTokenMs = null,
        [property: JsonPropertyName("inter_token_latency_ms")] double? InterTokenLatencyMs = null,
        [property: JsonPropertyName("prefix_cache_enabled")] bool? PrefixCacheEnabled = null,
        [property: JsonPropertyName("prefill_tokens_reused")] double? PrefillTokensReused = null,
        [property: JsonPropertyName("prefix_cache_hit_rate_percent")] double? PrefixCacheHitRatePercent = null,
        [property: JsonPropertyName("working_set_bytes")] double? WorkingSetBytes = null,
        [property: JsonPropertyName("managed_heap_bytes")] double? ManagedHeapBytes = null,
        [property: JsonPropertyName("warnings")] string[]? Warnings = null
    );

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(3) };

        while (!cancellation.IsCancellationRequested)
        {
            StatusSnapshot snapshot = QueryStatus(client, settings.Url, cancellation);

            if (settings.Watch && !settings.Json && !Console.IsOutputRedirected)
            {
                try { Console.Clear(); } catch { }
            }

            if (settings.Json)
            {
                Console.WriteLine(JsonSerializer.Serialize(snapshot, StatusJsonContext.Default.StatusSnapshot));
            }
            else
            {
                RenderDashboard(snapshot);
            }

            if (!settings.Watch)
                return snapshot.Error != null || snapshot.Status == "offline" ? 1 : 0;

            // Wait on the cancellation handle rather than sleeping: Thread.Sleep cannot observe
            // cancellation, so Ctrl-C was only honoured on the next loop check — up to a second
            // late, every time. WaitOne returns true the moment cancellation is signalled.
            if (cancellation.WaitHandle.WaitOne(1000))
                break;
        }

        return 0;
    }

    internal static StatusSnapshot QueryStatus(HttpClient client, string baseUrl, CancellationToken cancellation)
    {
        string baseUri = baseUrl.TrimEnd('/');

        // The versioned document answers everything in one round trip. A transport failure here
        // is not reported directly: the /health probe below produces the canonical offline
        // snapshot with a connection message, so failing over keeps one error path.
        try
        {
            var statusResp = client.GetAsync($"{baseUri}/status", cancellation).GetAwaiter().GetResult();
            if (statusResp.IsSuccessStatusCode)
            {
                string body = statusResp.Content.ReadAsStringAsync(cancellation).GetAwaiter().GetResult();
                if (TryParseStatusDocument(body, baseUri, out StatusSnapshot fromDocument))
                    return fromDocument;
            }
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Fall through to the legacy scrape.
        }

        return QueryLegacyStatus(client, baseUri, cancellation);
    }

    /// <summary>
    /// Parses the server's versioned <c>/status</c> document. Returns false — rather than a
    /// half-populated snapshot — when the payload is not a status document of a schema version
    /// this build understands, so an unrelated 200 response cannot masquerade as server status.
    /// </summary>
    internal static bool TryParseStatusDocument(string json, string baseUri, out StatusSnapshot snapshot)
    {
        snapshot = null!;
        JsonDocument doc;
        try { doc = JsonDocument.Parse(json); }
        catch (JsonException) { return false; }

        using (doc)
        {
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object) return false;
            if (!TryGetPropertyCaseInsensitive(root, "schema_version", out var versionElement) ||
                versionElement.ValueKind != JsonValueKind.Number)
                return false;

            int schemaVersion = versionElement.GetInt32();
            if (schemaVersion != StatusDocumentSchemaVersion) return false;

            var placement = Section(root, "placement");
            var traffic = Section(root, "traffic");
            var latency = Section(root, "latency");
            var cache = Section(root, "cache");
            var memory = Section(root, "memory");
            var batching = Section(root, "batching");

            double kvPct = Number(batching, "kv_occupancy_percent") ?? 0;

            snapshot = new StatusSnapshot(
                Status: Text(root, "status") ?? "ok",
                Url: baseUri,
                Model: Text(root, "model") ?? "unknown",
                UptimeSeconds: Number(root, "uptime_seconds") ?? 0,
                RequestsTotal: Number(traffic, "requests_total") ?? 0,
                TokensGeneratedTotal: Number(traffic, "tokens_generated_total") ?? 0,
                TokensPerSecond: Number(traffic, "tokens_per_second") ?? 0,
                ActiveSequences: Number(traffic, "active_requests") ?? 0,
                QueuedRequests: Number(traffic, "queued_requests") ?? 0,
                KvOccupancyPercent: kvPct,
                Error: null,
                SchemaVersion: schemaVersion,
                Backend: Text(placement, "backend"),
                ContextSize: (int?)Number(placement, "context_size"),
                TimeToFirstTokenMs: Number(Section(latency, "time_to_first_token"), "p50_ms")
                                    ?? Number(Section(latency, "time_to_first_token"), "mean_ms"),
                InterTokenLatencyMs: Number(latency, "inter_token_mean_ms"),
                PrefixCacheEnabled: Flag(cache, "prefix_cache_enabled"),
                PrefillTokensReused: Number(cache, "prefill_tokens_reused_total"),
                PrefixCacheHitRatePercent: Number(cache, "hit_rate_percent"),
                WorkingSetBytes: Number(memory, "working_set_bytes"),
                ManagedHeapBytes: Number(memory, "managed_heap_bytes"),
                Warnings: Strings(root, "warnings"));
            return true;
        }
    }

    /// <summary>Schema version of the <c>/status</c> document this build understands.</summary>
    internal const int StatusDocumentSchemaVersion = 1;

    private static JsonElement? Section(JsonElement? parent, string name) =>
        parent is { } p && TryGetPropertyCaseInsensitive(p, name, out var v) && v.ValueKind == JsonValueKind.Object
            ? v
            : null;

    private static double? Number(JsonElement? parent, string name) =>
        parent is { } p && TryGetPropertyCaseInsensitive(p, name, out var v) && v.ValueKind == JsonValueKind.Number
            ? v.GetDouble()
            : null;

    private static string? Text(JsonElement? parent, string name) =>
        parent is { } p && TryGetPropertyCaseInsensitive(p, name, out var v) && v.ValueKind == JsonValueKind.String
            ? v.GetString()
            : null;

    private static bool? Flag(JsonElement? parent, string name) =>
        parent is { } p && TryGetPropertyCaseInsensitive(p, name, out var v)
            ? v.ValueKind switch
            {
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                _ => null,
            }
            : null;

    private static string[]? Strings(JsonElement? parent, string name)
    {
        if (parent is not { } p ||
            !TryGetPropertyCaseInsensitive(p, name, out var v) ||
            v.ValueKind != JsonValueKind.Array)
            return null;

        var items = new List<string>(v.GetArrayLength());
        foreach (var element in v.EnumerateArray())
            if (element.ValueKind == JsonValueKind.String && element.GetString() is { } s)
                items.Add(s);
        return items.Count == 0 ? null : [.. items];
    }

    private static StatusSnapshot QueryLegacyStatus(HttpClient client, string baseUri, CancellationToken cancellation)
    {
        string healthUrl = $"{baseUri}/health";
        string metricsUrl = $"{baseUri}/metrics";

        string model = "unknown";
        double uptime = 0;
        string status = "offline";
        string? error = null;

        try
        {
            var healthResp = client.GetAsync(healthUrl, cancellation).GetAwaiter().GetResult();
            if (healthResp.IsSuccessStatusCode)
            {
                string json = healthResp.Content.ReadAsStringAsync(cancellation).GetAwaiter().GetResult();
                using var doc = JsonDocument.Parse(json);
                var root = doc.RootElement;
                if (TryGetPropertyCaseInsensitive(root, "status", out var s)) status = s.GetString() ?? "ok";
                if (TryGetPropertyCaseInsensitive(root, "model", out var m)) model = m.GetString() ?? "unknown";
                if (TryGetPropertyCaseInsensitive(root, "uptimeSeconds", out var u) ||
                    TryGetPropertyCaseInsensitive(root, "uptime_seconds", out u))
                {
                    if (u.ValueKind == JsonValueKind.Number) uptime = u.GetDouble();
                }
            }
            else
            {
                error = $"HTTP {(int)healthResp.StatusCode} ({healthResp.ReasonPhrase})";
                return new StatusSnapshot(
                    Status: "offline",
                    Url: baseUri,
                    Model: "none",
                    UptimeSeconds: 0,
                    RequestsTotal: 0,
                    TokensGeneratedTotal: 0,
                    TokensPerSecond: 0,
                    ActiveSequences: 0,
                    QueuedRequests: 0,
                    KvOccupancyPercent: 0,
                    Error: error
                );
            }
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            error = $"Failed connecting to {baseUri}: {ex.Message}";
            return new StatusSnapshot(
                Status: "offline",
                Url: baseUri,
                Model: "none",
                UptimeSeconds: 0,
                RequestsTotal: 0,
                TokensGeneratedTotal: 0,
                TokensPerSecond: 0,
                ActiveSequences: 0,
                QueuedRequests: 0,
                KvOccupancyPercent: 0,
                Error: error
            );
        }

        double requests = 0, tokens = 0, tokSec = 0, activeSeq = 0, queued = 0, kvPct = 0;
        try
        {
            var metricsResp = client.GetAsync(metricsUrl, cancellation).GetAwaiter().GetResult();
            if (metricsResp.IsSuccessStatusCode)
            {
                string metricsText = metricsResp.Content.ReadAsStringAsync(cancellation).GetAwaiter().GetResult();
                ParseMetrics(metricsText, ref requests, ref tokens, ref tokSec, ref activeSeq, ref queued, ref kvPct);
            }
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Ignore metrics endpoint failure if health succeeded
        }

        return new StatusSnapshot(
            Status: status,
            Url: baseUri,
            Model: model,
            UptimeSeconds: uptime,
            RequestsTotal: requests,
            TokensGeneratedTotal: tokens,
            TokensPerSecond: tokSec,
            ActiveSequences: activeSeq,
            QueuedRequests: queued,
            KvOccupancyPercent: kvPct,
            Error: null
        );
    }

    internal static void ParseMetrics(string text, ref double requests, ref double tokens, ref double tokSec, ref double activeSeq, ref double queued, ref double kvPct)
    {
        double committedKv = -1;
        double kvBudget = -1;

        using var reader = new StringReader(text);
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            if (line.StartsWith('#') || string.IsNullOrWhiteSpace(line)) continue;
            string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2) continue;

            string rawKey = parts[0];
            if (!double.TryParse(parts[1], out double val)) continue;

            // Strip metric prefixes: "opentail-llm_", "opentail_llm_", "opentail_"
            string key = rawKey;
            if (key.StartsWith("opentail-llm_")) key = key.Substring("opentail-llm_".Length);
            else if (key.StartsWith("opentail_llm_")) key = key.Substring("opentail_llm_".Length);
            else if (key.StartsWith("opentail_")) key = key.Substring("opentail_".Length);

            if (key == "requests_total") requests = val;
            else if (key == "tokens_generated_total") tokens = val;
            else if (key == "tokens_per_second") tokSec = val;
            else if (key == "active_requests" || key == "batching_active_sequences") activeSeq = val;
            else if (key == "queue_depth" || key == "batching_queued_requests") queued = val;
            else if (key == "batch_kv_committed_tokens") committedKv = val;
            else if (key == "batch_kv_token_budget") kvBudget = val;
            else if (key == "kv_occupancy_percent") kvPct = val;
        }

        // The server reports long.MaxValue as "no KV admission budget configured"; a percentage of
        // an unlimited budget is meaningless, so leave kvPct at whatever an explicit
        // kv_occupancy_percent metric supplied (0 if none).
        if (committedKv >= 0 && kvBudget > 0 && kvBudget < long.MaxValue)
        {
            kvPct = (committedKv / kvBudget) * 100.0;
        }
    }

    internal static bool TryGetPropertyCaseInsensitive(JsonElement element, string propertyName, out JsonElement value)
    {
        foreach (var prop in element.EnumerateObject())
        {
            if (string.Equals(prop.Name, propertyName, StringComparison.OrdinalIgnoreCase))
            {
                value = prop.Value;
                return true;
            }
        }
        value = default;
        return false;
    }

    private static void RenderDashboard(StatusSnapshot s)
    {
        Console.WriteLine($"OpenTail Server Status: {s.Url}");
        Console.WriteLine(new string('-', 50));
        if (s.Error is { } err)
        {
            Console.WriteLine($"Status:               OFFLINE ({err})");
            return;
        }

        TimeSpan up = TimeSpan.FromSeconds(s.UptimeSeconds);
        Console.WriteLine($"Status:               {s.Status.ToUpperInvariant()}");
        Console.WriteLine($"Active Model:         {s.Model}");
        if (s.Backend is { Length: > 0 } backend)
        {
            string context = s.ContextSize is > 0 ? $", {s.ContextSize:N0} ctx" : string.Empty;
            Console.WriteLine($"Placement:            {backend}{context}");
        }
        Console.WriteLine($"Server Uptime:        {up.Hours:D2}h {up.Minutes:D2}m {up.Seconds:D2}s");
        Console.WriteLine($"Generation Rate:      {s.TokensPerSecond:F1} tok/s");
        Console.WriteLine($"Tokens Generated:     {s.TokensGeneratedTotal:F0}");
        Console.WriteLine($"Total Requests:       {s.RequestsTotal:F0}");
        Console.WriteLine($"Active Sequences:     {s.ActiveSequences:F0}");
        Console.WriteLine($"Queued Requests:      {s.QueuedRequests:F0}");
        Console.WriteLine($"KV Cache Occupancy:   {s.KvOccupancyPercent:F1}%");

        // Rows below need the versioned /status document. On the /health + /metrics fallback the
        // values are unknown, and an unknown latency printed as 0 ms reads as a very fast server.
        if (s.TimeToFirstTokenMs is { } ttft)
            Console.WriteLine($"Time To First Token:  {ttft:F0} ms (p50)");
        if (s.InterTokenLatencyMs is { } itl)
            Console.WriteLine($"Inter-Token Latency:  {itl:F1} ms (mean)");
        if (s.PrefixCacheEnabled is { } prefixEnabled)
        {
            string reuse = s.PrefillTokensReused is { } reused
                ? $"{reused:N0} prompt tokens reused"
                : "no reuse counter";
            string hitRate = s.PrefixCacheHitRatePercent is { } rate ? $", {rate:F0}% hit rate" : string.Empty;
            Console.WriteLine($"Prefix Cache:         {(prefixEnabled ? "enabled" : "unavailable")} — {reuse}{hitRate}");
        }
        if (s.WorkingSetBytes is { } workingSet)
        {
            string heap = s.ManagedHeapBytes is { } managed ? $" ({Mib(managed)} managed heap)" : string.Empty;
            Console.WriteLine($"Host Memory:          {Mib(workingSet)} working set{heap}");
        }

        if (s.Warnings is { Length: > 0 } warnings)
        {
            Console.WriteLine();
            foreach (string warning in warnings)
                Console.WriteLine($"  ! {warning}");
        }
    }

    private static string Mib(double bytes) => $"{bytes / (1024 * 1024):N0} MiB";
}

[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(StatusCommand.StatusSnapshot))]
internal partial class StatusJsonContext : JsonSerializerContext
{
}
