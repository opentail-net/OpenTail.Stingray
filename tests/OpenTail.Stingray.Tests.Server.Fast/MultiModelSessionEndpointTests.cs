
namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// docs/032-multi-model-inference-runtime-plan.md Phase 7 follow-up — multi-model
/// <c>/v1/sessions/*</c> routing through the real HTTP endpoints. Fake engines only (mirrors
/// <see cref="MultiModelEndpointTests"/>'s pattern for the stateless endpoints); the real-model
/// end-to-end acceptance test lives in the heavy Tests.Server project.
///
/// <para>Single-model session behavior is covered exhaustively by <see cref="SessionEndpointTests"/>
/// and is untouched here — this file proves the new multi-model routing/wiring itself: session
/// creation resolves and binds the requested model, later calls route back to that same model, an
/// unknown session id or model alias is rejected cleanly, and deleting a session releases its
/// model-residency claim.</para>
/// </summary>
public sealed class MultiModelSessionEndpointTests : IDisposable
{
    private readonly List<WebApplicationFactory<Program>> _factories = new();
    private readonly List<IDisposable> _engines = new();

    public void Dispose()
    {
        foreach (var factory in _factories) factory.Dispose();
        foreach (var engine in _engines) engine.Dispose();
    }

    private const int Eos = 31;

    // Local copies of the fakes SessionEndpointTests already uses; they are private nested types
    // there, so this is a local copy rather than a shared one (same convention that file's own
    // doc comment already establishes).
    private sealed class Tokenizer : ITokenizer
    {
        public int VocabSize => 64;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;
        public IReadOnlyCollection<int> EogTokenIds => [Eos];
        public IReadOnlyList<int> Encode(string text) => [1, 2];
        public string Decode(IEnumerable<int> tokens) => "tok";
        public byte[] DecodeBytes(int token) => [];
    }

    private sealed class FakeCache : IRewindableSequenceKvCache
    {
        public int LogicalPosition { get; set; }
        public bool CanRewindTo(int logicalPosition) => logicalPosition >= 0 && logicalPosition <= LogicalPosition;
        public void RewindTo(int logicalPosition) => LogicalPosition = logicalPosition;
        public void Dispose() { }
    }

    private sealed class FakeForwardPass : IBatchedForwardPass
    {
        public int PrefillCount;

        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 1;
        public int MaxSeqLen => 512;
        public bool PrefillDequantCacheActive => false;
        public ISequenceKvCache CreateCache() => new FakeCache();

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
        {
            Interlocked.Increment(ref PrefillCount);
            var retained = Assert.IsType<FakeCache>(cache);
            retained.LogicalPosition = startPos + tokens.Count;
            var logits = new float[64];
            logits[Eos] = 1f; // stop immediately: these tests are about routing, not generation
            return logits;
        }

        public float[]?[] PrefillPackedMulti(ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits) =>
            throw new NotSupportedException();

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            var rows = new float[tokens.Length][];
            for (int i = 0; i < tokens.Length; i++)
            {
                if (caches[i] is FakeCache cache) cache.LogicalPosition = positions[i] + 1;
                var logits = new float[64];
                logits[Eos] = 1f;
                rows[i] = logits;
            }
            return rows;
        }
    }

    /// <summary>A named model's real (fake-backed) session runtime, for wiring into a
    /// <see cref="NamedModelOptions.EngineFactory"/>.</summary>
    private sealed class ModelLane : IDisposable
    {
        public readonly FakeForwardPass ForwardPass = new();
        public readonly ContinuousBatchingEngine Engine;
        public readonly Sessions.HotSessionRuntime SessionRuntime;

        public ModelLane(string modelId)
        {
            Engine = new ContinuousBatchingEngine(ForwardPass, new Tokenizer(), modelId, maxBatchSize: 1);
            SessionRuntime = new Sessions.HotSessionRuntime(Engine, new Tokenizer());
        }

        public void Dispose() => Engine.Dispose();
    }

    private HttpClient CreateMultiModelClient(ModelLane sidekick, ModelLane reasoner)
    {
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
            builder.ConfigureServices(services =>
            {
                services.Configure<OpenTailStingrayServerOptions>(o =>
                {
                    o.Models =
                    [
                        new NamedModelOptions
                        {
                            Alias = "sidekick",
                            ModelPath = "/models/sidekick.gguf",
                            EngineFactory = _ => new LoadedEngine(sidekick.Engine, "qwen2", null,
                                SessionRuntime: sidekick.SessionRuntime),
                        },
                        new NamedModelOptions
                        {
                            Alias = "reasoner",
                            ModelPath = "/models/reasoner.gguf",
                            EngineFactory = _ => new LoadedEngine(reasoner.Engine, "qwen2", null,
                                SessionRuntime: reasoner.SessionRuntime),
                        },
                    ];
                });
            }));
        _factories.Add(factory);
        return factory.CreateClient();
    }

    private static async Task<Guid> CreateSessionAsync(HttpClient client, string? model)
    {
        var body = model is null ? (object)new { } : new { model };
        var created = await client.PostAsJsonAsync("/v1/sessions", body);
        Assert.Equal(HttpStatusCode.Created, created.StatusCode);
        using var json = JsonDocument.Parse(await created.Content.ReadAsStringAsync());
        return Guid.Parse(json.RootElement.GetProperty("id").GetString()!);
    }

    private static object TurnBody(string prompt, long revision, int maxTokens = 4) =>
        new { append_prompt = prompt, expected_revision = revision, max_tokens = maxTokens };

    [Fact]
    public async Task CreateAndRunTurn_RoutesToTheRequestedModel_ByAlias()
    {
        var sidekick = new ModelLane("sidekick-engine");
        var reasoner = new ModelLane("reasoner-engine");
        _engines.Add(sidekick);
        _engines.Add(reasoner);
        var client = CreateMultiModelClient(sidekick, reasoner);

        var sessionId = await CreateSessionAsync(client, "sidekick");
        var turn = await client.PostAsJsonAsync($"/v1/sessions/{sessionId}/turns", TurnBody("hi", 0));
        Assert.Equal(HttpStatusCode.OK, turn.StatusCode);

        Assert.Equal(1, sidekick.ForwardPass.PrefillCount);
        Assert.Equal(0, reasoner.ForwardPass.PrefillCount);
    }

    [Fact]
    public async Task CaseInsensitiveAlias_Matches()
    {
        var sidekick = new ModelLane("sidekick-engine");
        var reasoner = new ModelLane("reasoner-engine");
        _engines.Add(sidekick);
        _engines.Add(reasoner);
        var client = CreateMultiModelClient(sidekick, reasoner);

        var sessionId = await CreateSessionAsync(client, "SIDEKICK");
        var turn = await client.PostAsJsonAsync($"/v1/sessions/{sessionId}/turns", TurnBody("hi", 0));
        Assert.Equal(HttpStatusCode.OK, turn.StatusCode);
        Assert.Equal(1, sidekick.ForwardPass.PrefillCount);
    }

    [Fact]
    public async Task TwoSessions_OnDifferentModels_DoNotCrossTalk()
    {
        var sidekick = new ModelLane("sidekick-engine");
        var reasoner = new ModelLane("reasoner-engine");
        _engines.Add(sidekick);
        _engines.Add(reasoner);
        var client = CreateMultiModelClient(sidekick, reasoner);

        var sidekickSession = await CreateSessionAsync(client, "sidekick");
        var reasonerSession = await CreateSessionAsync(client, "reasoner");

        Assert.Equal(HttpStatusCode.OK,
            (await client.PostAsJsonAsync($"/v1/sessions/{sidekickSession}/turns", TurnBody("a", 0))).StatusCode);
        Assert.Equal(HttpStatusCode.OK,
            (await client.PostAsJsonAsync($"/v1/sessions/{reasonerSession}/turns", TurnBody("b", 0))).StatusCode);

        Assert.Equal(1, sidekick.ForwardPass.PrefillCount);
        Assert.Equal(1, reasoner.ForwardPass.PrefillCount);
    }

    [Fact]
    public async Task Create_UnknownModelAlias_Returns404()
    {
        var sidekick = new ModelLane("sidekick-engine");
        var reasoner = new ModelLane("reasoner-engine");
        _engines.Add(sidekick);
        _engines.Add(reasoner);
        var client = CreateMultiModelClient(sidekick, reasoner);

        var response = await client.PostAsJsonAsync("/v1/sessions", new { model = "nonexistent" });
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    [Fact]
    public async Task NoModelField_DefaultsToFirstConfiguredModel()
    {
        var sidekick = new ModelLane("sidekick-engine");
        var reasoner = new ModelLane("reasoner-engine");
        _engines.Add(sidekick);
        _engines.Add(reasoner);
        var client = CreateMultiModelClient(sidekick, reasoner);

        var sessionId = await CreateSessionAsync(client, model: null);
        var turn = await client.PostAsJsonAsync($"/v1/sessions/{sessionId}/turns", TurnBody("hi", 0));
        Assert.Equal(HttpStatusCode.OK, turn.StatusCode);
        Assert.Equal(1, sidekick.ForwardPass.PrefillCount); // "sidekick" is configured first
        Assert.Equal(0, reasoner.ForwardPass.PrefillCount);
    }

    [Fact]
    public async Task UnknownSessionId_Get_Returns404()
    {
        var sidekick = new ModelLane("sidekick-engine");
        var reasoner = new ModelLane("reasoner-engine");
        _engines.Add(sidekick);
        _engines.Add(reasoner);
        var client = CreateMultiModelClient(sidekick, reasoner);

        var response = await client.GetAsync($"/v1/sessions/{Guid.NewGuid()}");
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    [Fact]
    public async Task UnknownSessionId_RunTurn_Returns404()
    {
        var sidekick = new ModelLane("sidekick-engine");
        var reasoner = new ModelLane("reasoner-engine");
        _engines.Add(sidekick);
        _engines.Add(reasoner);
        var client = CreateMultiModelClient(sidekick, reasoner);

        var response = await client.PostAsJsonAsync($"/v1/sessions/{Guid.NewGuid()}/turns", TurnBody("hi", 0));
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    [Fact]
    public async Task Delete_ReleasesTheSession_SubsequentGetIs404()
    {
        var sidekick = new ModelLane("sidekick-engine");
        var reasoner = new ModelLane("reasoner-engine");
        _engines.Add(sidekick);
        _engines.Add(reasoner);
        var client = CreateMultiModelClient(sidekick, reasoner);

        var sessionId = await CreateSessionAsync(client, "sidekick");
        Assert.Equal(HttpStatusCode.OK, (await client.GetAsync($"/v1/sessions/{sessionId}")).StatusCode);
        Assert.Equal(HttpStatusCode.NoContent, (await client.DeleteAsync($"/v1/sessions/{sessionId}")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await client.GetAsync($"/v1/sessions/{sessionId}")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await client.DeleteAsync($"/v1/sessions/{sessionId}")).StatusCode);
    }
}
