using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Plan 2 P7c through the real endpoints: OpenAI <c>stop</c> and Anthropic <c>stop_sequences</c> cut the answer at the first match,
/// report the right finish/stop reason, and cancel the engine instead of letting it generate to the end.
/// </summary>
public sealed class StopSequenceEndpointTests
{
    /// <summary>Yields the given pieces as answer text, one per step, and records how far it was read and whether it was cancelled.</summary>
    private sealed class PiecesEngine(params string[] pieces) : IInferenceEngine
    {
        public string ModelId => "test-model";
        public int QueueDepth => 0;
        public int ActiveRequests => 0;
        public bool PrefixCacheEnabled => true;
        public long PrefillTokensReused => 0;

        public int Produced;
        public volatile bool CancelledSeen;
        public volatile bool RanToEnd;

        public async IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
            string prompt, SamplingParams sp, [EnumeratorCancellation] CancellationToken ct = default, string? canonicalHistoryPrefix = null)
        {
            try
            {
                foreach (string piece in pieces)
                {
                    ct.ThrowIfCancellationRequested();
                    await Task.Yield();
                    Interlocked.Increment(ref Produced);
                    yield return new GenerateChunk(GenerateChunkKind.Text, piece);
                }
                RanToEnd = true;
                yield return new GenerateChunk(GenerateChunkKind.Stop, "");
            }
            finally
            {
                if (ct.IsCancellationRequested) CancelledSeen = true;
            }
        }

        public IAsyncEnumerable<GenerateChunk> GenerateImageChunksAsync(
            string prompt, IReadOnlyList<byte[]> imageBytes, SamplingParams sp, CancellationToken ct = default)
            => GenerateChunksAsync(prompt, sp, ct);
    }

    private static readonly string[] Pieces = ["Hello ", "wor", "ld! END more text", "never 1", "never 2", "never 3"];

    private static (HttpClient Client, PiecesEngine Engine, WebApplicationFactory<Program> Factory) Host()
    {
        var engine = new PiecesEngine(Pieces);
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(engine)));
        return (factory.CreateClient(), engine, factory);
    }

    private static Dictionary<string, object?> ChatBody(bool stream, object? stop)
    {
        var body = new Dictionary<string, object?>
        {
            ["model"] = "test-model",
            ["messages"] = new[] { new { role = "user", content = "Hi" } },
            ["max_tokens"] = 50,
            ["stream"] = stream,
        };
        if (stop is not null) body["stop"] = stop;
        return body;
    }

    private static Dictionary<string, object?> MessagesBody(bool stream, string[]? stopSequences)
    {
        var body = new Dictionary<string, object?>
        {
            ["model"] = "test-model",
            ["messages"] = new[] { new { role = "user", content = "Hi" } },
            ["max_tokens"] = 50,
            ["stream"] = stream,
        };
        if (stopSequences is not null) body["stop_sequences"] = stopSequences;
        return body;
    }

    private static IEnumerable<JsonDocument> SseData(string body)
    {
        foreach (string line in body.Split('\n'))
            if (line.StartsWith("data: ", StringComparison.Ordinal) && !line.StartsWith("data: [DONE]", StringComparison.Ordinal))
                yield return JsonDocument.Parse(line["data: ".Length..]);
    }

    private static async Task<HttpResponseMessage> Post(HttpClient c, string path, object body)
        => await c.PostAsJsonAsync(path, body).WaitAsync(TimeSpan.FromSeconds(30));

    // ── OpenAI ────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(new[] { "END" }, true)]    // stop as a single string
    [InlineData(new[] { "END" }, false)]   // stop as a one-element array
    [InlineData(new[] { "zzz", "END" }, false)]
    public async Task OpenAi_NonStreaming_CutsAtTheStop_ReportsStop_AndCancelsTheEngine(string[] stops, bool asSingleString)
    {
        object stop = asSingleString ? stops[0] : stops;
        var (client, engine, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await Post(client, "/v1/chat/completions", ChatBody(false, stop));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        var choice = doc.RootElement.GetProperty("choices")[0];
        Assert.Equal("Hello world! ", choice.GetProperty("message").GetProperty("content").GetString());
        Assert.Equal("stop", choice.GetProperty("finish_reason").GetString());
        Assert.True(engine.CancelledSeen, "the engine's token was never cancelled, so it would have generated to the end");
        Assert.False(engine.RanToEnd);
        Assert.True(engine.Produced <= 3, $"engine produced {engine.Produced} chunks after the stop string appeared in the 3rd");
    }

    [Fact]
    public async Task OpenAi_NonStreaming_WithoutStop_RunsToTheEnd()
    {
        var (client, engine, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await Post(client, "/v1/chat/completions", ChatBody(false, null));

        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("Hello world! END more textnever 1never 2never 3",
            doc.RootElement.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString());
        Assert.True(engine.RanToEnd);
        Assert.False(engine.CancelledSeen);
    }

    [Fact]
    public async Task OpenAi_Streaming_NeverSendsTheStopOrAnythingAfterIt_AndCancelsTheEngine()
    {
        var (client, engine, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await Post(client, "/v1/chat/completions", ChatBody(true, "END"));
        string body = await response.Content.ReadAsStringAsync();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var text = new StringBuilder();
        string? finish = null;
        foreach (var doc in SseData(body))
        {
            using (doc)
            {
                var choice = doc.RootElement.GetProperty("choices")[0];
                if (choice.GetProperty("delta").TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String)
                    text.Append(c.GetString());
                if (choice.TryGetProperty("finish_reason", out var f) && f.ValueKind == JsonValueKind.String)
                    finish = f.GetString();
            }
        }
        Assert.Equal("Hello world! ", text.ToString());
        Assert.DoesNotContain("END", body);
        Assert.Equal("stop", finish);
        Assert.True(engine.CancelledSeen);
        Assert.True(engine.Produced <= 3);
    }

    [Theory]
    [InlineData("[\"a\",\"b\",\"c\",\"d\",\"e\"]", "at most 4")]
    [InlineData("\"\"", "empty string")]
    [InlineData("[\"ok\",\"\"]", "empty string")]
    [InlineData("5", "string or an array")]
    [InlineData("[1]", "string or an array")]
    [InlineData("{\"a\":1}", "string or an array")]
    public async Task OpenAi_InvalidStop_IsA400_AndNeverReachesTheEngine(string stopJson, string expectedFragment)
    {
        var (client, engine, factory) = Host();
        using var _f = factory; using var _c = client;
        string json = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":5,\"stop\":" + stopJson + "}";

        var response = await client.PostAsync("/v1/chat/completions",
            new StringContent(json, Encoding.UTF8, "application/json")).WaitAsync(TimeSpan.FromSeconds(30));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Contains(expectedFragment, doc.RootElement.GetProperty("message").GetString());
        Assert.Equal(0, engine.Produced);
    }

    [Fact]
    public async Task OpenAi_StopNull_IsTreatedAsAbsent()
    {
        var (client, engine, factory) = Host();
        using var _f = factory; using var _c = client;
        string json = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":5,\"stop\":null}";

        var response = await client.PostAsync("/v1/chat/completions",
            new StringContent(json, Encoding.UTF8, "application/json")).WaitAsync(TimeSpan.FromSeconds(30));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.True(engine.RanToEnd);
    }

    // ── Anthropic ─────────────────────────────────────────────────────────

    [Fact]
    public async Task Anthropic_NonStreaming_CutsAtTheStop_ReportsStopSequence_AndCancelsTheEngine()
    {
        var (client, engine, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await Post(client, "/v1/messages", MessagesBody(false, ["END"]));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("Hello world! ", doc.RootElement.GetProperty("content")[0].GetProperty("text").GetString());
        Assert.Equal("stop_sequence", doc.RootElement.GetProperty("stop_reason").GetString());
        Assert.Equal("END", doc.RootElement.GetProperty("stop_sequence").GetString());
        Assert.True(engine.CancelledSeen);
        Assert.True(engine.Produced <= 3);
    }

    [Fact]
    public async Task Anthropic_NonStreaming_WithoutStopSequences_EndsWithEndTurn()
    {
        var (client, engine, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await Post(client, "/v1/messages", MessagesBody(false, null));

        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("end_turn", doc.RootElement.GetProperty("stop_reason").GetString());
        Assert.True(engine.RanToEnd);
    }

    [Fact]
    public async Task Anthropic_Streaming_CutsAtTheStop_AndReportsItInMessageDelta()
    {
        var (client, engine, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await Post(client, "/v1/messages", MessagesBody(true, ["END"]));
        string body = await response.Content.ReadAsStringAsync();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var text = new StringBuilder();
        string? reason = null, matched = null;
        foreach (var doc in SseData(body))
        {
            using (doc)
            {
                string type = doc.RootElement.GetProperty("type").GetString()!;
                if (type == "content_block_delta" && doc.RootElement.GetProperty("delta").TryGetProperty("text", out var t))
                    text.Append(t.GetString());
                if (type == "message_delta")
                {
                    var delta = doc.RootElement.GetProperty("delta");
                    reason = delta.GetProperty("stop_reason").GetString();
                    if (delta.TryGetProperty("stop_sequence", out var s) && s.ValueKind == JsonValueKind.String) matched = s.GetString();
                }
            }
        }
        Assert.Equal("Hello world! ", text.ToString());
        Assert.DoesNotContain("END", body.Replace("\"stop_sequence\":\"END\"", ""));
        Assert.Equal("stop_sequence", reason);
        Assert.Equal("END", matched);
        Assert.True(engine.CancelledSeen);
        Assert.True(engine.Produced <= 3);
    }

    [Fact]
    public async Task Anthropic_EmptyStopSequence_IsA400()
    {
        var (client, engine, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await Post(client, "/v1/messages", MessagesBody(false, ["ok", ""]));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Contains("stop_sequences", doc.RootElement.GetProperty("message").GetString());
        Assert.Equal(0, engine.Produced);
    }
}
