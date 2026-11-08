namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Owner decision D3: a tool call that does not match the request's declared tools is returned as an error (HTTP 422),
/// not as a tool call. A valid call still returns normally. Fake engine, no checkpoint.
/// </summary>
public sealed class ToolCallValidationEndpointTests : IDisposable
{
    private readonly List<WebApplicationFactory<Program>> _factories = new();

    public void Dispose()
    {
        foreach (var factory in _factories) factory.Dispose();
    }

    private HttpClient CreateClient(FakeInferenceEngine fake, string architecture)
    {
        var factory = new WebApplicationFactory<Program>()
            .WithWebHostBuilder(b => b.ConfigureServices(s =>
            {
                s.Configure<OpenTailStingrayServerOptions>(o => o.Architecture = architecture);
                s.AddSingleton<IInferenceEngine>(fake);
            }));
        _factories.Add(factory);
        return factory.CreateClient();
    }

    private static object OpenAiWeatherTool() => new
    {
        type = "function",
        function = new
        {
            name = "get_weather",
            description = "Get weather",
            parameters = new { type = "object", properties = new { city = new { type = "string" } }, required = new[] { "city" } }
        }
    };

    [Fact]
    public async Task OpenAi_InvalidArguments_ReturnTheCallAsAnErrorNot_AToolCall()
    {
        var fake = new FakeInferenceEngine("qwen2", [
            (GenerateChunkKind.Text, "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":5}}</tool_call>"),
        ]);
        var client = CreateClient(fake, "qwen2");

        var response = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "qwen2",
            messages = new[] { new { role = "user", content = "Weather?" } },
            tools = new[] { OpenAiWeatherTool() },
            stream = false,
        });

        Assert.Equal(HttpStatusCode.UnprocessableEntity, response.StatusCode);
        var body = await response.Content.ReadAsStringAsync();
        Assert.Contains("invalid_tool_call", body, StringComparison.Ordinal);
        Assert.DoesNotContain("tool_calls", body, StringComparison.Ordinal);
    }

    [Fact]
    public async Task OpenAi_ValidArguments_StillReturnAToolCall()
    {
        var fake = new FakeInferenceEngine("qwen2", [
            (GenerateChunkKind.Text, "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>"),
        ]);
        var client = CreateClient(fake, "qwen2");

        var response = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "qwen2",
            messages = new[] { new { role = "user", content = "Weather?" } },
            tools = new[] { OpenAiWeatherTool() },
            stream = false,
        });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var body = await response.Content.ReadAsStringAsync();
        Assert.Contains("tool_calls", body, StringComparison.Ordinal);
        Assert.Contains("get_weather", body, StringComparison.Ordinal);
    }

    [Fact]
    public async Task OpenAi_Streaming_InvalidArguments_EmitAnErrorEvent_NotAToolCallDelta()
    {
        var fake = new FakeInferenceEngine("qwen2", [
            (GenerateChunkKind.Text, "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":5}}</tool_call>"),
        ]);
        var client = CreateClient(fake, "qwen2");

        var response = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "qwen2",
            messages = new[] { new { role = "user", content = "Weather?" } },
            tools = new[] { OpenAiWeatherTool() },
            stream = true,
        });

        // The stream is already open with status 200, so the error arrives as an event in the body.
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var body = await response.Content.ReadAsStringAsync();
        Assert.Contains("invalid_tool_call", body, StringComparison.Ordinal);
        Assert.DoesNotContain("\"tool_calls\"", body, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Anthropic_UndeclaredTool_ReturnsAnError_NotAToolUseBlock()
    {
        var fake = new FakeInferenceEngine("qwen3-coder", [
            (GenerateChunkKind.Text, "<function=delete_file>"),
            (GenerateChunkKind.Text, "<parameter=path>/tmp/x</parameter>"),
            (GenerateChunkKind.Text, "</function>"),
        ]);
        var client = CreateClient(fake, "qwen3coder");

        var response = await client.PostAsJsonAsync("/v1/messages", new
        {
            model = "qwen3-coder",
            messages = new[] { new { role = "user", content = "Weather?" } },
            max_tokens = 50,
            stream = false,
            tools = new[] { new
            {
                name = "get_weather",
                description = "Get weather",
                input_schema = new { type = "object", properties = new { city = new { type = "string" } } }
            } }
        });

        Assert.Equal(HttpStatusCode.UnprocessableEntity, response.StatusCode);
        var body = await response.Content.ReadAsStringAsync();
        Assert.Contains("invalid_tool_call", body, StringComparison.Ordinal);
        Assert.DoesNotContain("tool_use", body, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Anthropic_Streaming_UndeclaredTool_EmitsAnErrorEvent_NotAToolUseBlock()
    {
        var fake = new FakeInferenceEngine("qwen3-coder", [
            (GenerateChunkKind.Text, "<function=delete_file>"),
            (GenerateChunkKind.Text, "<parameter=path>/tmp/x</parameter>"),
            (GenerateChunkKind.Text, "</function>"),
        ]);
        var client = CreateClient(fake, "qwen3coder");

        var response = await client.PostAsJsonAsync("/v1/messages", new
        {
            model = "qwen3-coder",
            messages = new[] { new { role = "user", content = "Weather?" } },
            max_tokens = 50,
            stream = true,
            tools = new[] { new
            {
                name = "get_weather",
                description = "Get weather",
                input_schema = new { type = "object", properties = new { city = new { type = "string" } } }
            } }
        });

        // The stream is already open with status 200, so the error arrives as an event in the body.
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var body = await response.Content.ReadAsStringAsync();
        Assert.Contains("event: error", body, StringComparison.Ordinal);
        Assert.Contains("invalid_tool_call", body, StringComparison.Ordinal);
        Assert.DoesNotContain("tool_use", body, StringComparison.Ordinal);
    }
}
