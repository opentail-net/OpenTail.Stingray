using System.Text;
using System.Text.Json;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Plan 2 P7d and P7e: <c>stream_options.include_usage</c> adds the final usage chunk OpenAI SDKs expect, Anthropic
/// <c>tool_choice</c> is honoured where it can be and refused where it cannot, and error bodies carry the nested <c>error</c> object
/// real clients parse while keeping the flat <c>type</c>/<c>message</c> fields existing consumers read.
/// </summary>
public sealed class UsageAndErrorEnvelopeTests
{
    private static (HttpClient Client, FakeInferenceEngine Fake, WebApplicationFactory<Program> Factory) Host(int promptTokens = 0)
    {
        var fake = new FakeInferenceEngine("test-model") { PromptTokens = promptTokens };
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(fake)));
        return (factory.CreateClient(), fake, factory);
    }

    private static async Task<HttpResponseMessage> PostJson(HttpClient c, string path, string json)
        => await c.PostAsync(path, new StringContent(json, Encoding.UTF8, "application/json")).WaitAsync(TimeSpan.FromSeconds(30));

    private static List<JsonDocument> SseEvents(string body)
    {
        var list = new List<JsonDocument>();
        foreach (string line in body.Split('\n'))
            if (line.StartsWith("data: ", StringComparison.Ordinal) && !line.StartsWith("data: [DONE]", StringComparison.Ordinal))
                list.Add(JsonDocument.Parse(line["data: ".Length..]));
        return list;
    }

    private const string ChatPrefix = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":10,\"stream\":true";

    // ── P7d: stream_options.include_usage ──────────────────────────────────

    [Fact]
    public async Task Streaming_IncludeUsage_AddsAFinalUsageChunk_AfterTheFinishReasonChunk()
    {
        var (client, _, factory) = Host(promptTokens: 7);
        using var _f = factory; using var _c = client;

        var response = await PostJson(client, "/v1/chat/completions", ChatPrefix + ",\"stream_options\":{\"include_usage\":true}}");
        string body = await response.Content.ReadAsStringAsync();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var events = SseEvents(body);
        try
        {
            var last = events[^1].RootElement;
            Assert.Equal(0, last.GetProperty("choices").GetArrayLength());
            var usage = last.GetProperty("usage");
            Assert.Equal(7, usage.GetProperty("prompt_tokens").GetInt32());
            Assert.Equal(3, usage.GetProperty("completion_tokens").GetInt32());
            Assert.Equal(10, usage.GetProperty("total_tokens").GetInt32());

            var finish = events[^2].RootElement.GetProperty("choices")[0];
            Assert.Equal("stop", finish.GetProperty("finish_reason").GetString());
            Assert.EndsWith("data: [DONE]", body.TrimEnd(), StringComparison.Ordinal);
        }
        finally { foreach (var e in events) e.Dispose(); }
    }

    [Theory]
    [InlineData("")]
    [InlineData(",\"stream_options\":{\"include_usage\":false}")]
    [InlineData(",\"stream_options\":{}")]
    [InlineData(",\"stream_options\":null")]
    public async Task Streaming_WithoutIncludeUsage_HasNoUsageChunk(string extra)
    {
        var (client, _, factory) = Host(promptTokens: 7);
        using var _f = factory; using var _c = client;

        var response = await PostJson(client, "/v1/chat/completions", ChatPrefix + extra + "}");
        string body = await response.Content.ReadAsStringAsync();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var events = SseEvents(body);
        try
        {
            foreach (var e in events)
                Assert.False(e.RootElement.TryGetProperty("usage", out var u) && u.ValueKind == JsonValueKind.Object,
                    "a usage object appeared although include_usage was not requested");
            Assert.Equal("stop", events[^1].RootElement.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
        }
        finally { foreach (var e in events) e.Dispose(); }
    }

    // ── P7d: Anthropic tool_choice ─────────────────────────────────────────

    private const string AnthropicTools =
        "\"tools\":[{\"name\":\"get_weather_unique_name\",\"description\":\"Weather\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}]";

    private static string Messages(string extra) =>
        "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":10," + extra + "}";

    [Theory]
    [InlineData("{\"type\":\"auto\"}")]
    [InlineData("{\"type\":\"none\"}")]
    [InlineData("null")]
    public async Task Anthropic_ToolChoice_AutoNoneAndNull_AreAccepted(string toolChoice)
    {
        var (client, _, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await PostJson(client, "/v1/messages", Messages(AnthropicTools + ",\"tool_choice\":" + toolChoice));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public void Anthropic_ToolChoiceNone_DropsTheTools_AndOtherFormsKeepThem()
    {
        var tools = new AnthropicTool[1];   // only presence matters here
        static AnthropicMessageRequest Req(AnthropicTool[]? tools, string? toolChoiceJson) => new(
            "m", null, 5, null, null, null, null, null, Tools: tools,
            ToolChoice: toolChoiceJson is null ? null : System.Text.Json.JsonDocument.Parse(toolChoiceJson).RootElement);

        Assert.Null(RequestParameterValidation.ApplyAnthropicToolChoice(Req(tools, "{\"type\":\"none\"}")).Tools);
        Assert.Same(tools, RequestParameterValidation.ApplyAnthropicToolChoice(Req(tools, "{\"type\":\"auto\"}")).Tools);
        Assert.Same(tools, RequestParameterValidation.ApplyAnthropicToolChoice(Req(tools, null)).Tools);
    }

    [Theory]
    [InlineData("{\"type\":\"any\"}", "not supported")]
    [InlineData("{\"type\":\"tool\",\"name\":\"get_weather_unique_name\"}", "not supported")]
    [InlineData("{\"type\":\"bogus\"}", "not one of")]
    [InlineData("\"auto\"", "must be an object")]
    [InlineData("{}", "must be an object")]
    public async Task Anthropic_ToolChoice_ThatCannotBeHonoured_IsA400_NamingTheField(string toolChoice, string fragment)
    {
        var (client, fake, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await PostJson(client, "/v1/messages", Messages(AnthropicTools + ",\"tool_choice\":" + toolChoice));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        string message = doc.RootElement.GetProperty("message").GetString()!;
        Assert.Contains("tool_choice", message);
        Assert.Contains(fragment, message);
        Assert.Equal(0, fake.EnteredCount);
    }

    // ── P7e: error envelope ────────────────────────────────────────────────

    [Fact]
    public async Task OpenAi_ErrorBody_HasTheNestedErrorObject_AndKeepsTheFlatFields()
    {
        var (client, _, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await PostJson(client, "/v1/chat/completions", "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":0}");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        var root = doc.RootElement;
        Assert.Equal("invalid_request_error", root.GetProperty("error").GetProperty("type").GetString());
        string nested = root.GetProperty("error").GetProperty("message").GetString()!;
        Assert.Contains("max_tokens", nested);
        Assert.Equal("invalid_request_error", root.GetProperty("type").GetString());
        Assert.Equal(nested, root.GetProperty("message").GetString());
    }

    [Fact]
    public async Task Anthropic_ErrorBody_HasTheNestedErrorObject_AndKeepsTheFlatFields()
    {
        var (client, _, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await PostJson(client, "/v1/messages", Messages("\"temperature\":1.5"));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        var root = doc.RootElement;
        Assert.Equal("invalid_request_error", root.GetProperty("error").GetProperty("type").GetString());
        string nested = root.GetProperty("error").GetProperty("message").GetString()!;
        Assert.Contains("temperature", nested);
        Assert.Equal("invalid_request_error", root.GetProperty("type").GetString());
        Assert.Equal(nested, root.GetProperty("message").GetString());
    }
}
