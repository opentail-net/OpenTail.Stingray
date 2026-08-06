using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Server;

namespace OpenTail.Stingray.Tests.Server;

/// <summary>
/// Wire contract for the versioned <c>/status</c> document (§7.4 of the QoL plan). These pin the
/// three things a status document has to get right: it is versioned, it never publishes an
/// internal sentinel or a path as if it were data, and its warnings describe conditions the
/// operator can act on.
/// </summary>
public sealed class StatusDocumentTests
{
    private static HttpClient CreateClient(
        IInferenceEngine engine,
        Action<OpenTailStingrayServerOptions>? configure = null) =>
        new WebApplicationFactory<Program>()
            .WithWebHostBuilder(builder => builder.ConfigureServices(services =>
            {
                if (configure is not null)
                    services.Configure(configure);
                services.AddSingleton<IInferenceEngine>(engine);
            }))
            .CreateClient();

    private static async Task<JsonDocument> GetStatusAsync(HttpClient client)
    {
        var response = await client.GetAsync("/status");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        return JsonDocument.Parse(await response.Content.ReadAsStringAsync());
    }

    /// <summary>
    /// The server's shared JSON context omits null properties, so "unknown" is an absent field
    /// rather than an explicit <c>null</c>. Both spellings mean the same thing to a client, and
    /// asserting on the meaning rather than the encoding keeps these tests from breaking if the
    /// host ever changes its ignore condition.
    /// </summary>
    private static void AssertUnknown(JsonElement parent, string property)
    {
        if (parent.TryGetProperty(property, out var value))
            Assert.Equal(JsonValueKind.Null, value.ValueKind);
    }

    [Fact]
    public async Task Status_IsVersionedAndDescribesModelPlacementAndTraffic()
    {
        var client = CreateClient(new StatusFakeEngine("smol.gguf"), options =>
        {
            options.Architecture = "qwen3";
            options.Backend = ServerBackend.Cuda;
            options.NGpuLayers = -1;
            options.ContextSize = 8192;
            options.MaxBatchSize = 1;
            options.SpecType = ServerSpecType.None;
        });

        using var document = await GetStatusAsync(client);
        var root = document.RootElement;

        Assert.Equal(ServerStatusSnapshot.CurrentSchemaVersion, root.GetProperty("schema_version").GetInt32());
        Assert.Equal("ok", root.GetProperty("status").GetString());
        Assert.Equal("smol.gguf", root.GetProperty("model").GetString());
        Assert.Equal("qwen3", root.GetProperty("architecture").GetString());
        Assert.True(root.GetProperty("uptime_seconds").GetInt64() >= 0);

        var placement = root.GetProperty("placement");
        Assert.Equal("cuda", placement.GetProperty("backend").GetString());
        Assert.Equal(-1, placement.GetProperty("gpu_layers").GetInt32());
        Assert.Equal(8192, placement.GetProperty("context_size").GetInt32());
        Assert.False(placement.GetProperty("continuous_batching").GetBoolean());

        var traffic = root.GetProperty("traffic");
        Assert.Equal(0, traffic.GetProperty("requests_total").GetInt64());
        Assert.Equal(0, traffic.GetProperty("overload_rejections_total").GetInt64());

        // A single-user engine has no batching section at all, rather than a section of zeroes
        // that would read as "continuous batching is on and idle".
        AssertUnknown(root, "batching");
    }

    [Fact]
    public async Task Status_LatencySummariesAreEmptyRatherThanZeroBeforeAnyRequest()
    {
        var client = CreateClient(new StatusFakeEngine("smol.gguf"));

        using var document = await GetStatusAsync(client);
        var latency = document.RootElement.GetProperty("latency");

        foreach (string section in (string[])["queue", "time_to_first_token", "generation", "request"])
        {
            var summary = latency.GetProperty(section);
            Assert.Equal(0, summary.GetProperty("count").GetInt64());
            // No samples means no percentile. Publishing 0 ms would claim an instantaneous server.
            AssertUnknown(summary, "p50_ms");
            AssertUnknown(summary, "p95_ms");
        }

        AssertUnknown(latency, "inter_token_mean_ms");
    }

    [Fact]
    public async Task Status_LatencyPercentilesAppearAfterASampledRequest()
    {
        var engine = new StatusFakeEngine("smol.gguf");
        var client = CreateClient(engine);

        var completion = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "smol.gguf",
            messages = new[] { new { role = "user", content = "hi" } },
            max_tokens = 4,
            stream = false,
        });
        Assert.Equal(HttpStatusCode.OK, completion.StatusCode);

        using var document = await GetStatusAsync(client);
        var latency = document.RootElement.GetProperty("latency");

        var ttft = latency.GetProperty("time_to_first_token");
        Assert.Equal(1, ttft.GetProperty("count").GetInt64());
        Assert.True(ttft.GetProperty("mean_ms").GetDouble() >= 0);

        Assert.Equal(1, document.RootElement.GetProperty("traffic").GetProperty("requests_total").GetInt64());
    }

    [Fact]
    public async Task Status_UnlimitedBudgetsArePublishedAsNullNotAsMaxValue()
    {
        // long.MaxValue is the engine's "no budget configured" sentinel. Serialised literally it
        // reads as a real 9.2-quintillion-token budget, and any occupancy percentage computed
        // from it would be silently meaningless.
        var engine = new StatusFakeEngine("smol.gguf")
        {
            ContinuousBatching = true,
            KvTokenBudget = long.MaxValue,
            PrefixCacheBudgetBytes = long.MaxValue,
            CommittedKvTokens = 4096,
        };
        var client = CreateClient(engine, options => options.MaxBatchSize = 4);

        using var document = await GetStatusAsync(client);
        var root = document.RootElement;

        var batching = root.GetProperty("batching");
        AssertUnknown(batching, "kv_token_budget");
        AssertUnknown(batching, "kv_occupancy_percent");
        Assert.Equal(4096, batching.GetProperty("committed_kv_tokens").GetInt64());
        AssertUnknown(root.GetProperty("cache"), "capacity_bytes");
        Assert.DoesNotContain(long.MaxValue.ToString(System.Globalization.CultureInfo.InvariantCulture),
            root.GetRawText(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task Status_ReportsKvOccupancyAndCacheReuseWhenBudgetsAreBounded()
    {
        var engine = new StatusFakeEngine("smol.gguf")
        {
            ContinuousBatching = true,
            KvTokenBudget = 16384,
            CommittedKvTokens = 4096,
            PrefillTokensReused = 12_345,
            PrefixCacheHits = 3,
            PrefixCacheMisses = 1,
            PrefixCacheBudgetBytes = 1024,
            PrefixCacheUsedBytes = 256,
        };
        var client = CreateClient(engine, options => options.MaxBatchSize = 4);

        using var document = await GetStatusAsync(client);
        var root = document.RootElement;

        Assert.Equal(25.0, root.GetProperty("batching").GetProperty("kv_occupancy_percent").GetDouble(), 3);

        var cache = root.GetProperty("cache");
        Assert.True(cache.GetProperty("prefix_cache_enabled").GetBoolean());
        Assert.Equal(12_345, cache.GetProperty("prefill_tokens_reused_total").GetInt64());
        Assert.Equal(75.0, cache.GetProperty("hit_rate_percent").GetDouble(), 3);
    }

    [Fact]
    public async Task Status_WarnsWhenPrefixReuseIsUnavailableForTheModelFamily()
    {
        // GDN hybrids cannot partially rewind their KV cache, so every turn re-prefills. That is
        // the single most common "why did multi-turn get slow?" answer and it is invisible in
        // /metrics, which only exposes it as a 0/1 gauge.
        var client = CreateClient(new StatusFakeEngine("qwen35moe.gguf") { PrefixCacheEnabled = false });

        using var document = await GetStatusAsync(client);
        string[] warnings = [.. document.RootElement.GetProperty("warnings")
            .EnumerateArray().Select(w => w.GetString() ?? string.Empty)];

        Assert.Contains(warnings, w => w.Contains("Prefix-cache reuse is unavailable", StringComparison.Ordinal));
        Assert.False(document.RootElement.GetProperty("cache").GetProperty("prefix_cache_enabled").GetBoolean());
    }

    [Fact]
    public async Task Status_SaturatedGateIsReportedAsStatusAndWarning()
    {
        var engine = new StatusFakeEngine("smol.gguf") { ActiveRequests = 1, QueueDepth = 1 };
        var client = CreateClient(engine, options =>
        {
            options.MaxBatchSize = 1;
            options.MaxQueuedRequests = 1; // capacity = batch(1) + queued(1) = 2
        });

        using var document = await GetStatusAsync(client);
        var root = document.RootElement;

        Assert.Equal("saturated", root.GetProperty("status").GetString());
        Assert.Equal(2, root.GetProperty("traffic").GetProperty("admission_capacity").GetInt32());
        Assert.Contains(root.GetProperty("warnings").EnumerateArray(),
            w => (w.GetString() ?? string.Empty).Contains("Admission capacity reached", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Status_ContainsEveryFieldTheCliStatusRendererReads()
    {
        // OpenTail.Stingray.Cli's `status` command parses these exact paths over HTTP. It is a
        // separate project with no reference to this one, so renaming a field here would
        // otherwise degrade the CLI to silently blank rows instead of failing a build.
        var engine = new StatusFakeEngine("smol.gguf")
        {
            ContinuousBatching = true,
            KvTokenBudget = 16384,
            CommittedKvTokens = 4096,
            PrefillTokensReused = 128,
            PrefixCacheHits = 1,
            PrefixCacheMisses = 1,
        };
        var client = CreateClient(engine, options => options.MaxBatchSize = 4);
        await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "smol.gguf",
            messages = new[] { new { role = "user", content = "hi" } },
            stream = false,
        });

        using var document = await GetStatusAsync(client);

        string[] paths =
        [
            "schema_version", "status", "model", "uptime_seconds", "warnings",
            "placement.backend", "placement.context_size",
            "traffic.requests_total", "traffic.tokens_generated_total", "traffic.tokens_per_second",
            "traffic.active_requests", "traffic.queued_requests",
            "latency.inter_token_mean_ms",
            "latency.time_to_first_token.mean_ms", "latency.time_to_first_token.p50_ms",
            "cache.prefix_cache_enabled", "cache.prefill_tokens_reused_total", "cache.hit_rate_percent",
            "memory.working_set_bytes", "memory.managed_heap_bytes",
            "batching.kv_occupancy_percent",
        ];

        foreach (string path in paths)
        {
            var element = document.RootElement;
            foreach (string segment in path.Split('.'))
            {
                Assert.True(element.TryGetProperty(segment, out element),
                    $"/status is missing '{path}', which the CLI status renderer reads.");
            }
        }
    }

    [Fact]
    public async Task Status_PublishesNoFilesystemPaths()
    {
        // The configured model path is the one genuinely sensitive string reachable from the
        // options the snapshot is built from — it discloses a username, a project name, or a
        // private model name. /status reports the engine's model *id*, never its location.
        var client = CreateClient(new StatusFakeEngine("smol.gguf"), options =>
            options.ModelPath = @"C:\secret-models\corporate-finetune-sk-live-1234.gguf");

        var response = await client.GetAsync("/status");
        string body = await response.Content.ReadAsStringAsync();

        Assert.DoesNotContain("sk-live-1234", body, StringComparison.Ordinal);
        Assert.DoesNotContain("secret-models", body, StringComparison.Ordinal);
        Assert.DoesNotContain("corporate-finetune", body, StringComparison.Ordinal);
        Assert.DoesNotContain(":\\", body, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Status_IsCheapEnoughToPollAtOneHertz()
    {
        // §7.4 requires a document cheap enough to poll once a second. Twenty sequential polls
        // well under a second in total leaves ample headroom; this fails loudly if the snapshot
        // ever starts doing real work (a forced GC, a device query, a file read).
        var client = CreateClient(new StatusFakeEngine("smol.gguf"));
        await client.GetAsync("/status"); // discard first-request JIT/route warmup

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < 20; i++)
            (await client.GetAsync("/status")).EnsureSuccessStatusCode();
        stopwatch.Stop();

        Assert.True(stopwatch.ElapsedMilliseconds < 1000,
            $"20 status polls took {stopwatch.ElapsedMilliseconds} ms");
    }
}

/// <summary>
/// A status-focused engine double. <see cref="FakeInferenceEngine"/> hard-codes its
/// observability properties, and the status document is largely a projection of exactly those
/// values, so it needs a double whose counters can be set.
/// </summary>
internal sealed class StatusFakeEngine(string modelId) : IInferenceEngine, IContinuousBatchingObservability
{
    public string ModelId { get; } = modelId;
    public int QueueDepth { get; init; }
    public int ActiveRequests { get; init; }
    public bool PrefixCacheEnabled { get; init; } = true;
    public long PrefillTokensReused { get; init; }

    public bool ContinuousBatching { get; init; }
    bool IContinuousBatchingObservability.IsContinuousBatching => ContinuousBatching;
    public int PrefillChunkTokens { get; init; } = 256;
    public long KvTokenBudget { get; init; } = long.MaxValue;
    public long CommittedKvTokens { get; init; }
    public long PrefixCacheBudgetBytes { get; init; } = long.MaxValue;
    public long PrefixCacheUsedBytes { get; init; }
    public int PrefixCacheEntries { get; init; }
    public long PrefixCacheHits { get; init; }
    public long PrefixCacheMisses { get; init; }
    public long PrefixCacheEvictions { get; init; }
    public long BatchedArgmaxSteps { get; init; }
    public long BatchedFullLogitsSteps { get; init; }
    public long BatchedArgmaxSequences { get; init; }
    public long BatchedFullLogitsSequences { get; init; }

    public async IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
        string prompt,
        SamplingParams sp,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default,
        string? canonicalHistoryPrefix = null)
    {
        await Task.Yield();
        // More than one chunk on purpose: mean inter-token latency is only defined once a
        // request has produced a second token, so a single-chunk double could never exercise it.
        yield return new GenerateChunk(GenerateChunkKind.Text, "ok");
        yield return new GenerateChunk(GenerateChunkKind.Text, " then");
        yield return new GenerateChunk(GenerateChunkKind.Text, " more");
    }
}
