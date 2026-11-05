namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Plan 6 G9: an engine factory that refuses a checkpoint with <see cref="NotSupportedException"/> (the Gemma 4 mixture-of-experts file) must reach
/// the client as a structured 400 carrying the refusal text on every endpoint family that acquires a model, not as a bare 500.
/// </summary>
public sealed class ModelLoadRefusalEndpointTests : IClassFixture<WebApplicationFactory<Program>>
{
    private const string Refusal = "this checkpoint uses a layout the gemma4 path does not run";
    private readonly WebApplicationFactory<Program> _factory;

    public ModelLoadRefusalEndpointTests(WebApplicationFactory<Program> factory) => _factory = factory;

    private HttpClient CreateClient() =>
        _factory.WithWebHostBuilder(builder =>
            builder.ConfigureServices(services =>
                services.Configure<OpenTailStingrayServerOptions>(o =>
                    o.Models =
                    [
                        new NamedModelOptions { Alias = "refused", ModelPath = "/models/refused.gguf", EngineFactory = _ => throw new NotSupportedException(Refusal) },
                        new NamedModelOptions { Alias = "other", ModelPath = "/models/other.gguf", EngineFactory = _ => new LoadedEngine(new FakeInferenceEngine("other"), "qwen2", null) },
                    ]))).CreateClient();

    private static async Task AssertStructuredRefusal(HttpResponseMessage resp)
    {
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
        string body = await resp.Content.ReadAsStringAsync();
        using var json = JsonDocument.Parse(body);
        Assert.Contains(Refusal, body);
        Assert.Equal("invalid_request_error", json.RootElement.GetProperty("type").GetString());
    }

    [Fact]
    public async Task ChatCompletions_ReturnsStructured400() =>
        await AssertStructuredRefusal(await CreateClient().PostAsJsonAsync("/v1/chat/completions",
            new { model = "refused", messages = new[] { new { role = "user", content = "hi" } }, stream = false }));

    [Fact]
    public async Task AnthropicMessages_ReturnsStructured400() =>
        await AssertStructuredRefusal(await CreateClient().PostAsJsonAsync("/v1/messages",
            new { model = "refused", max_tokens = 8, messages = new[] { new { role = "user", content = "hi" } } }));

    [Fact]
    public async Task Responses_ReturnsStructured400() =>
        await AssertStructuredRefusal(await CreateClient().PostAsJsonAsync("/v1/responses",
            new { model = "refused", input = "hi", stream = false }));

    [Fact]
    public async Task SessionCreate_ReturnsStructured400() =>
        await AssertStructuredRefusal(await CreateClient().PostAsJsonAsync("/v1/sessions", new { model = "refused" }));

    [Fact]
    public async Task ARefusedModel_DoesNotBreakAnotherModel()
    {
        var client = CreateClient();
        await AssertStructuredRefusal(await client.PostAsJsonAsync("/v1/chat/completions",
            new { model = "refused", messages = new[] { new { role = "user", content = "hi" } }, stream = false }));
        var ok = await client.PostAsJsonAsync("/v1/chat/completions",
            new { model = "other", messages = new[] { new { role = "user", content = "hi" } }, stream = false });
        Assert.Equal(HttpStatusCode.OK, ok.StatusCode);
    }
}
