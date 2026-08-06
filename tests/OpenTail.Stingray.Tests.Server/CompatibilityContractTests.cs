using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Server;

namespace OpenTail.Stingray.Tests.Server;

/// <summary>
/// A compact external-client regression suite. Endpoint-specific tests retain detailed parsing
/// coverage elsewhere; these tests protect the cross-protocol wire contracts that must remain
/// true while server configuration and planning work evolves.
/// </summary>
public sealed class CompatibilityContractTests
{
    private static HttpClient CreateClient(
        FakeInferenceEngine engine,
        Action<OpenTailStingrayServerOptions>? configure = null) =>
        new WebApplicationFactory<Program>()
            .WithWebHostBuilder(builder => builder.ConfigureServices(services =>
            {
                if (configure is not null)
                    services.Configure(configure);
                services.AddSingleton<IInferenceEngine>(engine);
            }))
            .CreateClient();

    [Fact]
    public async Task Capabilities_DescribeMappedProtocolsAndEffectiveCompatibilityConfiguration()
    {
        var client = CreateClient(new FakeInferenceEngine("test-model"), options =>
        {
            options.Architecture = "qwen3coder";
            options.Backend = ServerBackend.Cuda;
            options.NGpuLayers = -1;
            options.ContextSize = 8192;
            options.MaxBatchSize = 1;
            options.DisableThinking = true;
            options.ToolGrammar = true;
            options.SpecType = ServerSpecType.None;
        });

        var response = await client.GetAsync("/capabilities");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var document = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        var root = document.RootElement;
        Assert.Equal(1, root.GetProperty("schema_version").GetInt32());
        Assert.Equal("test-model", root.GetProperty("model").GetString());
        Assert.Equal("qwen3coder", root.GetProperty("architecture").GetString());

        var api = root.GetProperty("api");
        Assert.True(api.GetProperty("open_ai_chat_completions").GetBoolean());
        Assert.True(api.GetProperty("open_ai_responses").GetBoolean());
        Assert.True(api.GetProperty("anthropic_messages").GetBoolean());
        Assert.True(api.GetProperty("open_ai_models").GetBoolean());

        var runtime = root.GetProperty("runtime");
        Assert.True(runtime.GetProperty("tool_grammar").GetProperty("available").GetBoolean());
        Assert.False(runtime.GetProperty("image_input").GetBoolean());

        var configuration = root.GetProperty("configuration");
        Assert.Equal("cuda", configuration.GetProperty("backend").GetString());
        Assert.Equal(-1, configuration.GetProperty("gpu_layers").GetInt32());
        Assert.Equal(8192, configuration.GetProperty("context_size").GetInt32());
        Assert.True(configuration.GetProperty("disable_thinking").GetBoolean());
        Assert.True(configuration.GetProperty("tool_grammar_requested").GetBoolean());
        Assert.Equal("none", configuration.GetProperty("spec_type").GetString());
        Assert.False(root.GetRawText().Contains("model_path", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Capabilities_ExplainThatToolGrammarIsUnavailableWithContinuousBatching()
    {
        var client = CreateClient(new FakeInferenceEngine("test-model"), options =>
        {
            options.MaxBatchSize = 2;
            options.ToolGrammar = true;
        });

        var response = await client.GetAsync("/capabilities");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var document = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        var toolGrammar = document.RootElement.GetProperty("runtime").GetProperty("tool_grammar");
        Assert.False(toolGrammar.GetProperty("available").GetBoolean());
        Assert.Contains("continuous batching", toolGrammar.GetProperty("detail").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task NonStreamingProtocols_KeepTheirRequiredTopLevelWireShapes()
    {
        var client = CreateClient(new FakeInferenceEngine("test-model") { PromptTokens = 3 });

        var chat = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hello" } },
            max_tokens = 8,
        });
        Assert.Equal(HttpStatusCode.OK, chat.StatusCode);
        using var chatJson = JsonDocument.Parse(await chat.Content.ReadAsStringAsync());
        Assert.Equal("chat.completion", chatJson.RootElement.GetProperty("object").GetString());
        Assert.Equal(JsonValueKind.Array, chatJson.RootElement.GetProperty("choices").ValueKind);
        Assert.Equal(JsonValueKind.String, chatJson.RootElement.GetProperty("choices")[0].GetProperty("finish_reason").ValueKind);
        Assert.Equal(JsonValueKind.Number, chatJson.RootElement.GetProperty("usage").GetProperty("prompt_tokens").ValueKind);

        var messages = await client.PostAsJsonAsync("/v1/messages", new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hello" } },
            max_tokens = 8,
        });
        Assert.Equal(HttpStatusCode.OK, messages.StatusCode);
        using var messagesJson = JsonDocument.Parse(await messages.Content.ReadAsStringAsync());
        Assert.Equal("message", messagesJson.RootElement.GetProperty("type").GetString());
        Assert.Equal("assistant", messagesJson.RootElement.GetProperty("role").GetString());
        Assert.Equal(JsonValueKind.Array, messagesJson.RootElement.GetProperty("content").ValueKind);
        Assert.Equal(JsonValueKind.Number, messagesJson.RootElement.GetProperty("usage").GetProperty("input_tokens").ValueKind);

        var responses = await client.PostAsJsonAsync("/v1/responses", new
        {
            model = "test-model",
            input = "Hello",
            max_output_tokens = 8,
        });
        Assert.Equal(HttpStatusCode.OK, responses.StatusCode);
        using var responsesJson = JsonDocument.Parse(await responses.Content.ReadAsStringAsync());
        Assert.Equal("response", responsesJson.RootElement.GetProperty("object").GetString());
        Assert.Equal("completed", responsesJson.RootElement.GetProperty("status").GetString());
        Assert.Equal(JsonValueKind.Array, responsesJson.RootElement.GetProperty("output").ValueKind);
        Assert.Equal(JsonValueKind.Number, responsesJson.RootElement.GetProperty("usage").GetProperty("input_tokens").ValueKind);
    }

    [Fact]
    public async Task StreamingProtocols_KeepTheirTerminalEventContracts()
    {
        var client = CreateClient(new FakeInferenceEngine("test-model"));

        var chat = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hello" } },
            max_tokens = 8,
            stream = true,
        });
        string chatBody = await chat.Content.ReadAsStringAsync();
        Assert.Equal("text/event-stream", chat.Content.Headers.ContentType?.MediaType);
        Assert.Contains("chat.completion.chunk", chatBody);
        Assert.True(chatBody.LastIndexOf("[DONE]", StringComparison.Ordinal) > chatBody.IndexOf("chat.completion.chunk", StringComparison.Ordinal));

        var messages = await client.PostAsJsonAsync("/v1/messages", new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hello" } },
            max_tokens = 8,
            stream = true,
        });
        string messagesBody = await messages.Content.ReadAsStringAsync();
        Assert.Equal("text/event-stream", messages.Content.Headers.ContentType?.MediaType);
        Assert.True(messagesBody.IndexOf("event: message_start", StringComparison.Ordinal) >= 0);
        Assert.True(messagesBody.LastIndexOf("event: message_stop", StringComparison.Ordinal) > messagesBody.IndexOf("event: message_start", StringComparison.Ordinal));

        var responses = await client.PostAsJsonAsync("/v1/responses", new
        {
            model = "test-model",
            input = "Hello",
            max_output_tokens = 8,
            stream = true,
        });
        string responsesBody = await responses.Content.ReadAsStringAsync();
        Assert.Equal("text/event-stream", responses.Content.Headers.ContentType?.MediaType);
        Assert.True(responsesBody.IndexOf("event: response.created", StringComparison.Ordinal) >= 0);
        Assert.True(responsesBody.LastIndexOf("event: response.completed", StringComparison.Ordinal) > responsesBody.IndexOf("event: response.created", StringComparison.Ordinal));
    }
}
