using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// A28: a prompt longer than the context window is the client's error. The chat, responses and messages endpoints
/// must answer 400 (not 500) and name the context length; the streaming chat endpoint must refuse before it commits
/// the 200 SSE header.
/// </summary>
public sealed class ContextLimitEndpointTests : IDisposable
{
    private const string Model = "context-limit-model";

    private readonly List<WebApplicationFactory<Program>> _factories = new();

    public void Dispose()
    {
        foreach (var factory in _factories) factory.Dispose();
    }

    private HttpClient CreateClient(IInferenceEngine engine)
    {
        var factory = new WebApplicationFactory<Program>()
            .WithWebHostBuilder(b => b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(engine)));
        _factories.Add(factory);
        return factory.CreateClient();
    }

    [Fact]
    public async Task NonStreamingChat_PromptOverTheContext_Is400_WithContextLengthExceededCode()
    {
        var client = CreateClient(new ContextRefusingEngine(Model, refuseAtPrecheck: false));

        using var response = await client.PostAsJsonAsync("/v1/chat/completions", ChatRequest(stream: false));

        string body = await response.Content.ReadAsStringAsync();
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        using var doc = JsonDocument.Parse(body);
        var error = doc.RootElement.GetProperty("error");
        Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
        Assert.Equal("context_length_exceeded", error.GetProperty("code").GetString());
        Assert.Contains(ContextRefusingEngine.Context.ToString(), error.GetProperty("message").GetString());
    }

    [Fact]
    public async Task StreamingChat_PromptOverTheContext_IsRefused_BeforeTheSseHeaderIsCommitted()
    {
        var client = CreateClient(new ContextRefusingEngine(Model, refuseAtPrecheck: true));

        using var response = await client.PostAsJsonAsync("/v1/chat/completions", ChatRequest(stream: true));

        string body = await response.Content.ReadAsStringAsync();
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal("application/json", response.Content.Headers.ContentType?.MediaType);
        using var doc = JsonDocument.Parse(body);
        Assert.Equal("context_length_exceeded", doc.RootElement.GetProperty("error").GetProperty("code").GetString());
    }

    [Fact]
    public async Task Responses_PromptOverTheContext_Is400_WithContextLengthExceededCode()
    {
        var client = CreateClient(new ContextRefusingEngine(Model, refuseAtPrecheck: false));

        using var response = await client.PostAsJsonAsync("/v1/responses",
            new { model = Model, input = "hi", stream = false });

        string body = await response.Content.ReadAsStringAsync();
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        using var doc = JsonDocument.Parse(body);
        Assert.Equal("context_length_exceeded", doc.RootElement.GetProperty("error").GetProperty("code").GetString());
    }

    [Fact]
    public async Task Messages_PromptOverTheContext_Is400_NamingTheContextLength()
    {
        var client = CreateClient(new ContextRefusingEngine(Model, refuseAtPrecheck: false));

        using var response = await client.PostAsJsonAsync("/v1/messages", new
        {
            model = Model,
            max_tokens = 16,
            messages = new[] { new { role = "user", content = "hi" } },
            stream = false,
        });

        string body = await response.Content.ReadAsStringAsync();
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        using var doc = JsonDocument.Parse(body);
        var error = doc.RootElement.GetProperty("error");
        Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
        Assert.Contains(ContextRefusingEngine.Context.ToString(), error.GetProperty("message").GetString());
    }

    // A28 open follow-up: the Responses and messages streaming paths run the same pre-check before their SSE headers.
    [Fact]
    public async Task StreamingResponses_PromptOverTheContext_IsRefused_BeforeTheSseHeaderIsCommitted()
    {
        var client = CreateClient(new ContextRefusingEngine(Model, refuseAtPrecheck: true));

        using var response = await client.PostAsJsonAsync("/v1/responses",
            new { model = Model, input = "hi", stream = true });

        string body = await response.Content.ReadAsStringAsync();
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal("application/json", response.Content.Headers.ContentType?.MediaType);
        using var doc = JsonDocument.Parse(body);
        Assert.Equal("context_length_exceeded", doc.RootElement.GetProperty("error").GetProperty("code").GetString());
    }

    [Fact]
    public async Task StreamingMessages_PromptOverTheContext_IsRefused_BeforeTheSseHeaderIsCommitted()
    {
        var client = CreateClient(new ContextRefusingEngine(Model, refuseAtPrecheck: true));

        using var response = await client.PostAsJsonAsync("/v1/messages", new
        {
            model = Model,
            max_tokens = 16,
            messages = new[] { new { role = "user", content = "hi" } },
            stream = true,
        });

        string body = await response.Content.ReadAsStringAsync();
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal("application/json", response.Content.Headers.ContentType?.MediaType);
        using var doc = JsonDocument.Parse(body);
        var error = doc.RootElement.GetProperty("error");
        Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
        Assert.Contains(ContextRefusingEngine.Context.ToString(), error.GetProperty("message").GetString());
    }

    private static object ChatRequest(bool stream) => new
    {
        model = Model,
        messages = new[] { new { role = "user", content = "hi" } },
        max_tokens = 16,
        stream,
    };

    /// <summary>
    /// An engine whose prompt does not fit the window. <paramref name="refuseAtPrecheck"/> refuses in the
    /// engine's pre-generation check (what the streaming endpoints call before committing the response); otherwise
    /// the refusal comes from generation itself, on the first step, as the continuous batcher's admission does.
    /// </summary>
    private sealed class ContextRefusingEngine(string modelId, bool refuseAtPrecheck) : IInferenceEngine
    {
        public const int Context = 4096;

        public string ModelId { get; } = modelId;
        public int QueueDepth => 0;
        public int ActiveRequests => 0;
        public bool PrefixCacheEnabled => false;
        public long PrefillTokensReused => 0;

        public void CheckPromptFitsContext(string prompt)
        {
            if (refuseAtPrecheck) throw new ContextLimitExceededException(0, Context + 1, Context);
        }

        public async IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
            string prompt,
            SamplingParams sp,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default,
            string? canonicalHistoryPrefix = null)
        {
            await Task.Yield();
            if (!refuseAtPrecheck) throw new ContextLimitExceededException(0, Context + 1, Context);
            yield return new GenerateChunk(GenerateChunkKind.Text, "unreachable in the refusal tests");
        }
    }
}
