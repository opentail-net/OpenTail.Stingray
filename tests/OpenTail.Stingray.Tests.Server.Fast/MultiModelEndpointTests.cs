using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Server;
using OpenTail.Stingray.Server.Endpoints;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// docs/032-multi-model-inference-runtime-plan.md Phase 7 — per-request model routing through
/// the real HTTP endpoints (OpenAI chat completions, Anthropic messages, Responses, /v1/models).
/// Fake engines only (no real GGUF); the real-model end-to-end acceptance test (sidekick +
/// reasoner, concurrent cross-model users, session isolation) lives in the heavy Tests.Server
/// project since it needs actual models. This file proves the routing/wiring itself: the right
/// request reaches the right engine, an unknown model is rejected cleanly, and /v1/models lists
/// every configured alias.
/// </summary>
public sealed class MultiModelEndpointTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;

    public MultiModelEndpointTests(WebApplicationFactory<Program> factory) => _factory = factory;

    private HttpClient CreateMultiModelClient(FakeInferenceEngine sidekick, FakeInferenceEngine reasoner) =>
        _factory.WithWebHostBuilder(builder =>
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
                            EngineFactory = _ => new LoadedEngine(sidekick, "qwen2", null),
                        },
                        new NamedModelOptions
                        {
                            Alias = "reasoner",
                            ModelPath = "/models/reasoner.gguf",
                            EngineFactory = _ => new LoadedEngine(reasoner, "qwen2", null),
                        },
                    ];
                });
            })).CreateClient();

    [Fact]
    public async Task ChatCompletions_RoutesToTheRequestedModel_ByAlias()
    {
        var sidekick = new FakeInferenceEngine("sidekick-engine", [(GenerateChunkKind.Text, "from-sidekick")]);
        var reasoner = new FakeInferenceEngine("reasoner-engine", [(GenerateChunkKind.Text, "from-reasoner")]);
        var client = CreateMultiModelClient(sidekick, reasoner);

        var sidekickResp = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "sidekick",
            messages = new[] { new { role = "user", content = "hi" } },
            stream = false,
        });
        var reasonerResp = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "reasoner",
            messages = new[] { new { role = "user", content = "hi" } },
            stream = false,
        });

        Assert.Equal(HttpStatusCode.OK, sidekickResp.StatusCode);
        Assert.Equal(HttpStatusCode.OK, reasonerResp.StatusCode);
        var sidekickBody = await sidekickResp.Content.ReadAsStringAsync();
        var reasonerBody = await reasonerResp.Content.ReadAsStringAsync();
        Assert.Contains("from-sidekick", sidekickBody);
        Assert.Contains("sidekick-engine", sidekickBody); // response.model
        Assert.Contains("from-reasoner", reasonerBody);
        Assert.Contains("reasoner-engine", reasonerBody);
        Assert.DoesNotContain("from-reasoner", sidekickBody);
        Assert.DoesNotContain("from-sidekick", reasonerBody);
    }

    [Fact]
    public async Task ChatCompletions_CaseInsensitiveAlias_Matches()
    {
        var sidekick = new FakeInferenceEngine("sidekick-engine");
        var reasoner = new FakeInferenceEngine("reasoner-engine");
        var client = CreateMultiModelClient(sidekick, reasoner);

        var resp = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "SIDEKICK",
            messages = new[] { new { role = "user", content = "hi" } },
            stream = false,
        });

        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        Assert.Equal(1, sidekick.EnteredCount);
        Assert.Equal(0, reasoner.EnteredCount);
    }

    [Fact]
    public async Task ChatCompletions_UnknownModel_Returns404()
    {
        var client = CreateMultiModelClient(new FakeInferenceEngine("a"), new FakeInferenceEngine("b"));

        var resp = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "does-not-exist",
            messages = new[] { new { role = "user", content = "hi" } },
            stream = false,
        });

        Assert.Equal(HttpStatusCode.NotFound, resp.StatusCode);
        var body = await resp.Content.ReadAsStringAsync();
        Assert.Contains("does-not-exist", body);
        Assert.Contains("sidekick", body); // available models listed in the error
    }

    [Fact]
    public async Task AnthropicMessages_RoutesToTheRequestedModel_ByAlias()
    {
        var sidekick = new FakeInferenceEngine("sidekick-engine", [(GenerateChunkKind.Text, "from-sidekick")]);
        var reasoner = new FakeInferenceEngine("reasoner-engine", [(GenerateChunkKind.Text, "from-reasoner")]);
        var client = CreateMultiModelClient(sidekick, reasoner);

        var resp = await client.PostAsJsonAsync("/v1/messages", new
        {
            model = "reasoner",
            max_tokens = 32,
            messages = new[] { new { role = "user", content = "hi" } },
        });

        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        var body = await resp.Content.ReadAsStringAsync();
        Assert.Contains("from-reasoner", body);
        Assert.Equal(0, sidekick.EnteredCount);
        Assert.Equal(1, reasoner.EnteredCount);
    }

    [Fact]
    public async Task Responses_RoutesToTheRequestedModel_ByAlias()
    {
        var sidekick = new FakeInferenceEngine("sidekick-engine", [(GenerateChunkKind.Text, "from-sidekick")]);
        var reasoner = new FakeInferenceEngine("reasoner-engine", [(GenerateChunkKind.Text, "from-reasoner")]);
        var client = CreateMultiModelClient(sidekick, reasoner);

        var resp = await client.PostAsJsonAsync("/v1/responses", new
        {
            model = "sidekick",
            input = "hi",
            stream = false,
        });

        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        var body = await resp.Content.ReadAsStringAsync();
        Assert.Contains("from-sidekick", body);
        Assert.Equal(1, sidekick.EnteredCount);
        Assert.Equal(0, reasoner.EnteredCount);
    }

    [Fact]
    public async Task ListModels_MultiModelMode_ListsEveryConfiguredAlias()
    {
        var client = CreateMultiModelClient(new FakeInferenceEngine("a"), new FakeInferenceEngine("b"));

        var resp = await client.GetAsync("/v1/models");

        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        var body = await resp.Content.ReadAsStringAsync();
        Assert.Contains("\"id\":\"sidekick\"", body);
        Assert.Contains("\"id\":\"reasoner\"", body);
    }

    [Fact]
    public async Task ListModels_SingleModelMode_Unchanged()
    {
        // No Models configured — /v1/models must keep reporting the injected engine's own
        // ModelId, exactly like before Phase 7 (not IInferenceService.AvailableModelAliases'
        // canonicalized-path string, which is a different value).
        var client = _factory.WithWebHostBuilder(builder =>
            builder.ConfigureServices(services =>
                services.AddSingleton<IInferenceEngine>(new FakeInferenceEngine("solo-model"))))
            .CreateClient();

        var resp = await client.GetAsync("/v1/models");

        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        var body = await resp.Content.ReadAsStringAsync();
        Assert.Contains("\"id\":\"solo-model\"", body);
    }
}
