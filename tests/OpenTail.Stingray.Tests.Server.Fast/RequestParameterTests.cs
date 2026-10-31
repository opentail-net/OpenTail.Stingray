using System.Text.Json;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Plan 2 P7 (docs/reference/request-parameter-audit-2026-10-31.md): sampling fields the server reads are range-checked and a
/// field it cannot honour is refused with a 400 that names it, while fields it can honour (<c>max_completion_tokens</c>,
/// <c>seed</c>, <c>top_k</c>) reach the engine. Unknown JSON properties keep being accepted: clients send vendor fields.
/// </summary>
public sealed class RequestParameterTests
{
    private static (HttpClient Client, FakeInferenceEngine Fake, WebApplicationFactory<Program> Factory) Host()
    {
        var fake = new FakeInferenceEngine("test-model");
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(fake)));
        return (factory.CreateClient(), fake, factory);
    }

    private static Dictionary<string, object?> ChatBody(params (string Key, object? Value)[] extra)
    {
        var body = new Dictionary<string, object?>
        {
            ["model"] = "test-model",
            ["messages"] = new[] { new { role = "user", content = "Hi" } },
            ["max_tokens"] = 5,
            ["stream"] = false,
        };
        foreach (var (k, v) in extra) body[k] = v;
        return body;
    }

    [Theory]
    [InlineData("max_tokens", 0)]
    [InlineData("max_tokens", -3)]
    [InlineData("max_completion_tokens", 0)]
    [InlineData("temperature", -0.1)]
    [InlineData("temperature", 2.5)]
    [InlineData("top_p", 1.5)]
    [InlineData("top_p", -0.1)]
    [InlineData("top_k", -1)]
    [InlineData("presence_penalty", 3.0)]
    [InlineData("frequency_penalty", -3.0)]
    [InlineData("n", 2)]
    [InlineData("n", 0)]
    public async Task OpenAi_OutOfRangeOrUnsupported_IsA400_NamingTheField_AndNeverReachesTheEngine(string field, double value)
    {
        var (client, fake, factory) = Host();
        using var _f = factory; using var _c = client;

        object sent = field is "max_tokens" or "max_completion_tokens" or "top_k" or "n" ? (int)value : value;
        var response = await client.PostAsJsonAsync("/v1/chat/completions", ChatBody((field, sent)));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("invalid_request_error", doc.RootElement.GetProperty("type").GetString());
        Assert.Contains(field, doc.RootElement.GetProperty("message").GetString());
        Assert.Equal(0, fake.EnteredCount);
    }

    [Theory]
    [InlineData("temperature", 0.0)]
    [InlineData("temperature", 2.0)]
    [InlineData("top_p", 0.0)]
    [InlineData("top_p", 1.0)]
    [InlineData("presence_penalty", -2.0)]
    [InlineData("frequency_penalty", 2.0)]
    public async Task OpenAi_BoundaryValues_AreAccepted(string field, double value)
    {
        var (client, _, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await client.PostAsJsonAsync("/v1/chat/completions", ChatBody((field, value)));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task OpenAi_NEqualToOne_IsAccepted()
    {
        var (client, _, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await client.PostAsJsonAsync("/v1/chat/completions", ChatBody(("n", 1)));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task OpenAi_MaxCompletionTokens_SetsTheBudget_AndWinsOverMaxTokens()
    {
        var (client, fake, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await client.PostAsJsonAsync("/v1/chat/completions",
            ChatBody(("max_tokens", 5), ("max_completion_tokens", 9)));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal(9, fake.LastSamplingParams!.MaxNewTokens);
    }

    [Fact]
    public async Task OpenAi_MaxCompletionTokensAlone_SetsTheBudget()
    {
        var (client, fake, factory) = Host();
        using var _f = factory; using var _c = client;
        var body = ChatBody(("max_completion_tokens", 11));
        body.Remove("max_tokens");

        var response = await client.PostAsJsonAsync("/v1/chat/completions", body);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal(11, fake.LastSamplingParams!.MaxNewTokens);
    }

    [Fact]
    public async Task OpenAi_SeedAndTopK_ReachSamplingParams()
    {
        var (client, fake, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await client.PostAsJsonAsync("/v1/chat/completions",
            ChatBody(("seed", 12345), ("top_k", 40)));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal(12345, fake.LastSamplingParams!.Seed);
        Assert.Equal(40, fake.LastSamplingParams.TopK);
    }

    [Fact]
    public async Task OpenAi_WithoutSeed_LeavesTheSeedUnset()
    {
        var (client, fake, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await client.PostAsJsonAsync("/v1/chat/completions", ChatBody());

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Null(fake.LastSamplingParams!.Seed);
    }

    [Fact]
    public async Task OpenAi_UnknownVendorFields_StillWork()
    {
        var (client, _, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await client.PostAsJsonAsync("/v1/chat/completions",
            ChatBody(("some_vendor_flag", true), ("another", new { nested = 1 }), ("parallel_tool_calls", false), ("user", "u-1")));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    private static Dictionary<string, object?> AnthropicBody(params (string Key, object? Value)[] extra)
    {
        var body = new Dictionary<string, object?>
        {
            ["model"] = "test-model",
            ["messages"] = new[] { new { role = "user", content = "Hi" } },
            ["max_tokens"] = 5,
        };
        foreach (var (k, v) in extra) body[k] = v;
        return body;
    }

    [Theory]
    [InlineData("max_tokens", 0)]
    [InlineData("temperature", 1.5)]
    [InlineData("temperature", -0.5)]
    [InlineData("top_p", 1.5)]
    [InlineData("top_k", -1)]
    public async Task Anthropic_OutOfRange_IsA400_NamingTheField_AndNeverReachesTheEngine(string field, double value)
    {
        var (client, fake, factory) = Host();
        using var _f = factory; using var _c = client;

        object sent = field is "max_tokens" or "top_k" ? (int)value : value;
        var response = await client.PostAsJsonAsync("/v1/messages", AnthropicBody((field, sent)));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("invalid_request_error", doc.RootElement.GetProperty("type").GetString());
        Assert.Contains(field, doc.RootElement.GetProperty("message").GetString());
        Assert.Equal(0, fake.EnteredCount);
    }

    [Fact]
    public async Task Anthropic_TemperatureOfOne_IsAccepted()
    {
        var (client, _, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await client.PostAsJsonAsync("/v1/messages", AnthropicBody(("temperature", 1.0)));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }
}
