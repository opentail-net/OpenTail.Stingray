namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// The sampling fields each endpoint accepts must arrive at the engine with the value the client sent, not the server default
/// (docs/reference/request-parameter-audit-2026-10-31.md, which listed <c>temperature</c> and <c>top_p</c> as implemented but untested
/// as request fields, and Anthropic <c>top_k</c> likewise). The values are chosen to differ from every server default, so a field that
/// is read and then dropped, or never read, shows up as the default.
/// </summary>
public sealed class SamplingFieldsReachTheEngineTests
{
    private static (HttpClient Client, FakeInferenceEngine Fake, WebApplicationFactory<Program> Factory) Host()
    {
        var fake = new FakeInferenceEngine("test-model");
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(fake)));
        return (factory.CreateClient(), fake, factory);
    }

    [Fact]
    public async Task OpenAiChat_TemperatureAndTopP_ReachTheEngine()
    {
        var (client, fake, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "test-model", messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 5, stream = false, temperature = 0.31, top_p = 0.77, top_k = 23,
        });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var sp = fake.LastSamplingParams!;
        Assert.Equal(0.31f, sp.Temperature);
        Assert.Equal(0.77f, sp.TopP);
        Assert.Equal(23, sp.TopK);
        Assert.Equal(5, sp.MaxNewTokens);
    }

    [Fact]
    public async Task OpenAiChat_RequestValues_OverrideTheServerDefaults_AndOmittedValuesKeepThem()
    {
        var (client, fake, factory) = Host();
        using var _f = factory; using var _c = client;

        await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "test-model", messages = new[] { new { role = "user", content = "Hi" } }, max_tokens = 5, stream = false,
        });
        var defaults = fake.LastSamplingParams!;

        await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "test-model", messages = new[] { new { role = "user", content = "Hi" } }, max_tokens = 5, stream = false, temperature = 0.31,
        });
        var sp = fake.LastSamplingParams!;

        Assert.Equal(0.31f, sp.Temperature);
        Assert.NotEqual(0.31f, defaults.Temperature);   // positive control: the override is visible against the default
        Assert.Equal(defaults.TopP, sp.TopP);            // a field the request omits keeps the server default
        Assert.Equal(defaults.TopK, sp.TopK);
    }

    [Fact]
    public async Task Anthropic_TemperatureTopPTopKAndMaxTokens_ReachTheEngine()
    {
        var (client, fake, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await client.PostAsJsonAsync("/v1/messages", new
        {
            model = "test-model", messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 7, stream = false, temperature = 0.41, top_p = 0.66, top_k = 19,
        });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var sp = fake.LastSamplingParams!;
        Assert.Equal(0.41f, sp.Temperature);
        Assert.Equal(0.66f, sp.TopP);
        Assert.Equal(19, sp.TopK);
        Assert.Equal(7, sp.MaxNewTokens);
    }

    [Fact]
    public async Task Responses_TemperatureTopPAndMaxOutputTokens_ReachTheEngine()
    {
        var (client, fake, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await client.PostAsJsonAsync("/v1/responses", new
        {
            model = "test-model", input = "Hi", stream = false, temperature = 0.52, top_p = 0.61, max_output_tokens = 6,
        });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var sp = fake.LastSamplingParams!;
        Assert.Equal(0.52f, sp.Temperature);
        Assert.Equal(0.61f, sp.TopP);
        Assert.Equal(6, sp.MaxNewTokens);
    }

    [Fact]
    public async Task OpenAiChat_LogitBias_ReachesTheEngineAsTokenIds()
    {
        var (client, fake, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "test-model", messages = new[] { new { role = "user", content = "Hi" } }, max_tokens = 5, stream = false,
            logit_bias = new Dictionary<string, double> { ["42"] = -50.0 },
        });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var bias = fake.LastSamplingParams!.LogitBias;
        Assert.NotNull(bias);
        Assert.Equal(-50f, bias![42]);
    }
}
