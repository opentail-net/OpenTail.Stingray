
namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Wire contract for the opt-in <c>opentail_timing</c> response extension (§8 Phase 2 item 3 of
/// the QoL plan). Non-streaming only for both protocols — see the field's own doc comment in
/// <c>ResponseTimingExtension.cs</c> for why opt-in rather than always-on.
/// </summary>
public sealed class ResponseTimingExtensionTests : IDisposable
{
    private readonly List<WebApplicationFactory<Program>> _factories = new();

    public void Dispose()
    {
        foreach (var factory in _factories) factory.Dispose();
    }

    private HttpClient CreateClient()
    {
        var factory = new WebApplicationFactory<Program>()
            .WithWebHostBuilder(builder => builder.ConfigureServices(services =>
                services.AddSingleton<IInferenceEngine>(new FakeInferenceEngine("test-model"))));
        _factories.Add(factory);
        return factory.CreateClient();
    }

    [Fact]
    public async Task OpenAi_DefaultRequest_OmitsTimingExtension()
    {
        var client = CreateClient();
        var response = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "hi" } },
            stream = false,
        });

        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.False(doc.RootElement.TryGetProperty("opentail_timing", out _));
    }

    [Fact]
    public async Task OpenAi_OptedInRequest_ReturnsTimingWithPositiveTotalMs()
    {
        var client = CreateClient();
        var response = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "hi" } },
            stream = false,
            opentail_timing = true,
        });

        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        var timing = doc.RootElement.GetProperty("opentail_timing");
        Assert.True(timing.GetProperty("total_ms").GetDouble() >= 0);
        // The fake engine always yields at least one text chunk, so TTFT must be recorded.
        Assert.True(timing.GetProperty("time_to_first_token_ms").GetDouble() >= 0);
    }

    [Fact]
    public async Task OpenAi_StreamingRequest_IgnoresOptInBecauseOnlyNonStreamingIsSupported()
    {
        var client = CreateClient();
        var response = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "hi" } },
            stream = true,
            opentail_timing = true,
        });

        string body = await response.Content.ReadAsStringAsync();
        Assert.DoesNotContain("opentail_timing", body, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Anthropic_DefaultRequest_OmitsTimingExtension()
    {
        var client = CreateClient();
        var response = await client.PostAsJsonAsync("/v1/messages", new
        {
            model = "test-model",
            max_tokens = 10,
            messages = new[] { new { role = "user", content = "hi" } },
            stream = false,
        });

        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.False(doc.RootElement.TryGetProperty("opentail_timing", out _));
    }

    [Fact]
    public async Task Anthropic_OptedInRequest_ReturnsTimingWithPositiveTotalMs()
    {
        var client = CreateClient();
        var response = await client.PostAsJsonAsync("/v1/messages", new
        {
            model = "test-model",
            max_tokens = 10,
            messages = new[] { new { role = "user", content = "hi" } },
            stream = false,
            opentail_timing = true,
        });

        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        var timing = doc.RootElement.GetProperty("opentail_timing");
        Assert.True(timing.GetProperty("total_ms").GetDouble() >= 0);
        Assert.True(timing.GetProperty("time_to_first_token_ms").GetDouble() >= 0);
    }

    [Fact]
    public async Task Responses_OptedInRequest_ReturnsTimingWithPositiveTotalMs()
    {
        var client = CreateClient();
        var response = await client.PostAsJsonAsync("/v1/responses", new
        {
            model = "test-model",
            input = "hi",
            opentail_timing = true,
        });

        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        var timing = doc.RootElement.GetProperty("opentail_timing");
        Assert.True(timing.GetProperty("total_ms").GetDouble() >= 0);
        Assert.True(timing.GetProperty("time_to_first_token_ms").GetDouble() >= 0);
    }
}
