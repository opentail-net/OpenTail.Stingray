
namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Diagnostics in a multi-model deployment (<see cref="OpenTailStingrayServerOptions.Models"/> set). Before the
/// fix, <c>/health</c>, <c>/metrics</c>, <c>/status</c> and <c>/capabilities</c> resolved the singleton
/// <see cref="IInferenceEngine"/>: with no top-level ModelPath that threw (500), and with one that matched an entry
/// it loaded and pinned that model, which in SingleSlot residency blocked every other model for ever.
/// </summary>
public sealed class MultiModelDiagnosticsTests : IDisposable
{
    private readonly List<WebApplicationFactory<Program>> _factories = new();

    public void Dispose()
    {
        foreach (var factory in _factories) factory.Dispose();
    }

    private sealed class LoadCounter
    {
        public int Sidekick;
        public int Reasoner;
    }

    private (HttpClient Client, IServiceProvider Services) CreateMultiModel(
        IInferenceEngine sidekick, IInferenceEngine reasoner, LoadCounter loads,
        Action<OpenTailStingrayServerOptions>? configure = null)
    {
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
            builder.ConfigureServices(services => services.Configure<OpenTailStingrayServerOptions>(o =>
            {
                o.Architecture = "llama";
                o.Models =
                [
                    new NamedModelOptions
                    {
                        Alias = "sidekick",
                        ModelPath = "/models/sidekick.gguf",
                        Architecture = "qwen2",
                        EngineFactory = _ =>
                        {
                            Interlocked.Increment(ref loads.Sidekick);
                            return new LoadedEngine(sidekick, "qwen3", null,
                                RuntimeResolution: new ServerRuntimeResolution("cpu", "cpu-dense", "gguf", 4096, 0));
                        },
                    },
                    new NamedModelOptions
                    {
                        Alias = "reasoner",
                        ModelPath = "/models/reasoner.gguf",
                        EngineFactory = _ =>
                        {
                            Interlocked.Increment(ref loads.Reasoner);
                            return new LoadedEngine(reasoner, "qwen2", null);
                        },
                    },
                ];
                configure?.Invoke(o);
            })));
        _factories.Add(factory);
        return (factory.CreateClient(), factory.Services);
    }

    private static Task<HttpResponseMessage> Chat(HttpClient client, string model, CancellationToken ct = default) =>
        client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model,
            messages = new[] { new { role = "user", content = "hi" } },
            stream = false,
        }, ct);

    private static async Task<JsonDocument> GetJson(HttpClient client, string route)
    {
        var response = await client.GetAsync(route);
        string body = await response.Content.ReadAsStringAsync();
        Assert.True(response.StatusCode == HttpStatusCode.OK, $"{route} returned {(int)response.StatusCode}: {body}");
        return JsonDocument.Parse(body);
    }

    private static string[] Aliases(JsonElement root) =>
        [.. root.GetProperty("models").EnumerateArray().Select(e => e.GetString() ?? "")];

    [Fact]
    public async Task Diagnostics_AnswerWithoutLoadingAnyModel()
    {
        var loads = new LoadCounter();
        var (client, _) = CreateMultiModel(new FakeInferenceEngine("a"), new FakeInferenceEngine("b"), loads);

        using var health = await GetJson(client, "/health");
        using var status = await GetJson(client, "/status");
        using var capabilities = await GetJson(client, "/capabilities");
        var metrics = await client.GetAsync("/metrics");

        Assert.Equal(HttpStatusCode.OK, metrics.StatusCode);
        Assert.Contains("opentail_llm_active_requests 0", await metrics.Content.ReadAsStringAsync(), StringComparison.Ordinal);
        Assert.Equal(0, loads.Sidekick);
        Assert.Equal(0, loads.Reasoner);

        foreach (var root in new[] { health.RootElement, status.RootElement, capabilities.RootElement })
        {
            // model = the default alias (where a request without `model` goes); models = every alias.
            Assert.Equal("sidekick", root.GetProperty("model").GetString());
            Assert.Equal(["sidekick", "reasoner"], Aliases(root));
        }

        // Not loaded yet: the default entry's configured architecture, not the global hint.
        Assert.Equal("qwen2", status.RootElement.GetProperty("architecture").GetString());
        Assert.Equal("qwen2", capabilities.RootElement.GetProperty("architecture").GetString());
    }

    [Fact]
    public async Task Diagnostics_DescribeTheResidentDefaultModel_AndNeverPinIt()
    {
        var loads = new LoadCounter();
        var (client, services) = CreateMultiModel(new FakeInferenceEngine("a"), new FakeInferenceEngine("b"), loads);

        Assert.Equal(HttpStatusCode.OK, (await Chat(client, "sidekick")).StatusCode);
        using var status = await GetJson(client, "/status");
        _ = await GetJson(client, "/capabilities");
        _ = await GetJson(client, "/health");

        var root = status.RootElement;
        Assert.Equal("qwen3", root.GetProperty("architecture").GetString()); // resolved at load, not configured
        Assert.Equal("cpu-dense", root.GetProperty("configuration").GetProperty("resolved").GetProperty("forward_pass").GetString());
        Assert.Equal(1, loads.Sidekick);

        var manager = services.GetRequiredService<IModelRuntimeManager>();
        Assert.All(manager.Snapshot(), s => Assert.False(s.Pinned));
    }

    [Fact]
    public async Task SingleSlot_DiagnosticsDoNotBlockSwitchingModels_EvenWithAMatchingTopLevelModelPath()
    {
        // The old singleton factory would have loaded and pinned /models/sidekick.gguf on /health, and a pinned
        // resident model makes every SingleSlot acquisition of another model wait for ever.
        var loads = new LoadCounter();
        var (client, _) = CreateMultiModel(new FakeInferenceEngine("a"), new FakeInferenceEngine("b"), loads, o =>
        {
            o.ModelPath = "/models/sidekick.gguf";
            o.ModelResidencyMode = ModelResidencyMode.SingleSlot;
        });

        Assert.Equal(HttpStatusCode.OK, (await Chat(client, "sidekick")).StatusCode);
        _ = await GetJson(client, "/health");
        _ = await GetJson(client, "/status");

        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        var reasoner = await Chat(client, "reasoner", timeout.Token);
        Assert.Equal(HttpStatusCode.OK, reasoner.StatusCode);
        Assert.Equal(1, loads.Reasoner);
    }

    [Fact]
    public async Task Metrics_SumQueueAndActiveCountersOverResidentModels()
    {
        var loads = new LoadCounter();
        var (client, _) = CreateMultiModel(
            new StatusFakeEngine("a") { ActiveRequests = 2, QueueDepth = 1, PrefillTokensReused = 10 },
            new StatusFakeEngine("b") { ActiveRequests = 3, QueueDepth = 4, PrefillTokensReused = 5, PrefixCacheEnabled = false },
            loads);

        Assert.Equal(HttpStatusCode.OK, (await Chat(client, "sidekick")).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await Chat(client, "reasoner")).StatusCode);

        string metrics = await (await client.GetAsync("/metrics")).Content.ReadAsStringAsync();
        Assert.Contains("opentail_llm_active_requests 5\n", metrics, StringComparison.Ordinal);
        Assert.Contains("opentail_llm_queue_depth 5\n", metrics, StringComparison.Ordinal);
        Assert.Contains("opentail_llm_prefill_tokens_reused_total 15\n", metrics, StringComparison.Ordinal);
        Assert.Contains("opentail_llm_prefix_cache_enabled 0\n", metrics, StringComparison.Ordinal);

        using var status = await GetJson(client, "/status");
        var traffic = status.RootElement.GetProperty("traffic");
        Assert.Equal(5, traffic.GetProperty("active_requests").GetInt32());
        Assert.Equal(5, traffic.GetProperty("queued_requests").GetInt32());
    }

    [Fact]
    public void SingletonEngine_IsRefusedInMultiModelMode_WithoutLoadingOrPinning()
    {
        int loads = 0;
        var s = new ServiceCollection();
        s.AddOpenTailStingray(o =>
        {
            o.ModelPath = "/models/sidekick.gguf";
            o.Models =
            [
                new NamedModelOptions
                {
                    Alias = "sidekick",
                    ModelPath = "/models/sidekick.gguf",
                    EngineFactory = _ =>
                    {
                        loads++;
                        return new LoadedEngine(new FakeInferenceEngine("a"), "qwen2", null);
                    },
                },
            ];
        });
        using var sp = s.BuildServiceProvider();

        var error = Assert.Throws<InvalidOperationException>(() => sp.GetRequiredService<IInferenceEngine>());
        Assert.Contains("IInferenceService", error.Message, StringComparison.Ordinal);
        Assert.Equal(0, loads);
        Assert.Empty(sp.GetRequiredService<IModelRuntimeManager>().Snapshot());
    }

    [Fact]
    public async Task SingleModel_HealthOmitsTheModelsList()
    {
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
            builder.ConfigureServices(services => services.AddSingleton<IInferenceEngine>(new FakeInferenceEngine("solo"))));
        _factories.Add(factory);

        using var health = await GetJson(factory.CreateClient(), "/health");
        Assert.Equal("solo", health.RootElement.GetProperty("model").GetString());
        if (health.RootElement.TryGetProperty("models", out var models))
            Assert.Equal(JsonValueKind.Null, models.ValueKind);
    }
}
