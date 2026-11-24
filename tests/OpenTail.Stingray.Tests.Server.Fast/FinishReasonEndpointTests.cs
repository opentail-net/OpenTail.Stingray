using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// The finish reason a run reports when it stops at its token budget (A29 open follow-up). The engine marks such a stop
/// with <c>TruncatedByMaxTokens</c>; the chat endpoint must answer <c>finish_reason: length</c> and the messages endpoint
/// <c>stop_reason: max_tokens</c>. The responses endpoint is not covered here: it does not read the flag at all (see
/// docs/00-current-work.md, A29, open item on the Responses status).
/// </summary>
public sealed class FinishReasonEndpointTests : IDisposable
{
    private const string Model = "finish-reason-model";

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
    public async Task Chat_RunStoppedAtItsTokenBudget_ReportsFinishReasonLength()
    {
        var client = CreateClient(new TruncatingEngine(Model));

        using var response = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = Model,
            messages = new[] { new { role = "user", content = "hi" } },
            max_tokens = 1,
        });

        string body = await response.Content.ReadAsStringAsync();
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var doc = JsonDocument.Parse(body);
        var choice = doc.RootElement.GetProperty("choices")[0];
        Assert.Equal("length", choice.GetProperty("finish_reason").GetString());
    }

    [Fact]
    public async Task Messages_RunStoppedAtItsTokenBudget_ReportsStopReasonMaxTokens()
    {
        var client = CreateClient(new TruncatingEngine(Model));

        using var response = await client.PostAsJsonAsync("/v1/messages", new
        {
            model = Model,
            max_tokens = 1,
            messages = new[] { new { role = "user", content = "hi" } },
            stream = false,
        });

        string body = await response.Content.ReadAsStringAsync();
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var doc = JsonDocument.Parse(body);
        Assert.Equal("max_tokens", doc.RootElement.GetProperty("stop_reason").GetString());
    }

    /// <summary>Emits one token of text, then a stop marked as stopped by the token budget.</summary>
    private sealed class TruncatingEngine(string modelId) : IInferenceEngine
    {
        public string ModelId { get; } = modelId;
        public int QueueDepth => 0;
        public int ActiveRequests => 0;
        public bool PrefixCacheEnabled => false;
        public long PrefillTokensReused => 0;

        public void CheckPromptFitsContext(string prompt) { }

        public async IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
            string prompt,
            SamplingParams sp,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default,
            string? canonicalHistoryPrefix = null)
        {
            await Task.Yield();
            yield return new GenerateChunk(GenerateChunkKind.Text, "a");
            yield return new GenerateChunk(GenerateChunkKind.Stop, "", TruncatedByMaxTokens: true);
        }
    }
}
