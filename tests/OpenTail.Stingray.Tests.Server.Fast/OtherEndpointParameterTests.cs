using System.Text;
using System.Text.Json;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// The parameter audit of the endpoints other than chat and messages
/// (docs/reference/request-parameter-audit-other-endpoints-2026-11-01.md): <c>/v1/responses</c> refuses a value outside the range the API
/// defines and the two request features it does not implement (non-empty <c>tools</c>, <c>previous_response_id</c>) with a 400 whose body
/// names the field, and every 400 on it carries a body; <c>/v1/audio/speech</c> refuses a <c>response_format</c> other than WAV instead of
/// answering with WAV bytes. A refused request never reaches the engine.
/// </summary>
public sealed class OtherEndpointParameterTests
{
    private static (HttpClient Client, FakeInferenceEngine Fake, WebApplicationFactory<Program> Factory) Host()
    {
        var fake = new FakeInferenceEngine("test-model");
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(fake)));
        return (factory.CreateClient(), fake, factory);
    }

    private static Dictionary<string, object?> ResponsesBody(params (string Key, object? Value)[] extra)
    {
        var body = new Dictionary<string, object?> { ["model"] = "test-model", ["input"] = "Hi", ["max_output_tokens"] = 5, ["stream"] = false };
        foreach (var (k, v) in extra) body[k] = v;
        return body;
    }

    private static async Task AssertRefusedNamingAsync(HttpResponseMessage response, string field)
    {
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        string text = await response.Content.ReadAsStringAsync();
        Assert.False(string.IsNullOrWhiteSpace(text), "a 400 must carry a body that says why");
        using var doc = JsonDocument.Parse(text);
        Assert.Equal("invalid_request_error", doc.RootElement.GetProperty("type").GetString());
        Assert.Contains(field, doc.RootElement.GetProperty("message").GetString());
        Assert.Contains(field, doc.RootElement.GetProperty("error").GetProperty("message").GetString());   // the nested object clients read
    }

    [Theory]
    [InlineData("max_output_tokens", 0)]
    [InlineData("max_output_tokens", -3)]
    [InlineData("temperature", 2.5)]
    [InlineData("temperature", -0.1)]
    [InlineData("top_p", 1.5)]
    public async Task Responses_OutOfRangeSamplingValue_IsA400_NamingTheField_AndNeverReachesTheEngine(string field, double value)
    {
        var (client, fake, factory) = Host();
        using var _f = factory; using var _c = client;

        object sent = field == "max_output_tokens" ? (int)value : value;
        var response = await client.PostAsJsonAsync("/v1/responses", ResponsesBody((field, sent)));

        await AssertRefusedNamingAsync(response, field);
        Assert.Equal(0, fake.EnteredCount);
    }

    [Fact]
    public async Task Responses_NonEmptyTools_AndPreviousResponseId_AreRefused_ButAnEmptyToolsArrayIsAccepted()
    {
        var (client, fake, factory) = Host();
        using var _f = factory; using var _c = client;

        var tools = new[] { new { type = "function", name = "get_weather" } };
        await AssertRefusedNamingAsync(await client.PostAsJsonAsync("/v1/responses", ResponsesBody(("tools", tools))), "tools");
        await AssertRefusedNamingAsync(await client.PostAsJsonAsync("/v1/responses", ResponsesBody(("previous_response_id", "resp_123"))), "previous_response_id");
        Assert.Equal(0, fake.EnteredCount);

        // Positive control: the same request without the unsupported features, and with an empty tools array, is served by the engine.
        var ok = await client.PostAsJsonAsync("/v1/responses", ResponsesBody(("tools", Array.Empty<object>())));
        Assert.Equal(HttpStatusCode.OK, ok.StatusCode);
        Assert.Equal(1, fake.EnteredCount);
    }

    // ── /v1/responses usage: the prompt size comes from the engine's usage chunk, not a constant 0 ──

    [Fact]
    public async Task Responses_NonStreaming_ReportsThePromptTokensTheEngineCounted()
    {
        var fake = new FakeInferenceEngine("test-model") { PromptTokens = 17 };
        using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b => b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(fake)));
        using var client = factory.CreateClient();

        var response = await client.PostAsJsonAsync("/v1/responses", ResponsesBody());

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        var usage = doc.RootElement.GetProperty("usage");
        Assert.Equal(17, usage.GetProperty("input_tokens").GetInt32());
        Assert.Equal(3, usage.GetProperty("output_tokens").GetInt32());   // the fake streams three text chunks
        Assert.Equal(20, usage.GetProperty("total_tokens").GetInt32());
    }

    [Fact]
    public async Task Responses_Streaming_CompletedEventReportsThePromptTokensTheEngineCounted()
    {
        var fake = new FakeInferenceEngine("test-model") { PromptTokens = 17 };
        using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b => b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(fake)));
        using var client = factory.CreateClient();

        var response = await client.PostAsJsonAsync("/v1/responses", ResponsesBody(("stream", true)));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        string body = await response.Content.ReadAsStringAsync();
        string? completed = body.Split('\n').Where(l => l.StartsWith("data:", StringComparison.Ordinal)).Select(l => l[5..].Trim())
            .LastOrDefault(l => l.Contains("response.completed", StringComparison.Ordinal));
        Assert.NotNull(completed);
        using var doc = JsonDocument.Parse(completed!);
        var usage = doc.RootElement.GetProperty("response").GetProperty("usage");
        Assert.Equal(17, usage.GetProperty("input_tokens").GetInt32());
        Assert.Equal(3, usage.GetProperty("output_tokens").GetInt32());
        Assert.Equal(20, usage.GetProperty("total_tokens").GetInt32());
    }
    [Fact]
    public async Task Responses_MalformedJson_IsA400_WithABody()
    {
        var (client, fake, factory) = Host();
        using var _f = factory; using var _c = client;

        using var content = new StringContent("{ not json", Encoding.UTF8, "application/json");
        var response = await client.PostAsync("/v1/responses", content);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("invalid_request_error", doc.RootElement.GetProperty("type").GetString());
        Assert.Equal(0, fake.EnteredCount);
    }

    [Theory]
    [InlineData("mp3")]
    [InlineData("opus")]
    [InlineData("aac")]
    [InlineData("flac")]
    [InlineData("pcm")]
    public async Task Speech_ResponseFormatOtherThanWav_IsA400_NamingTheField(string format)
    {
        var (client, _, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await client.PostAsJsonAsync("/v1/audio/speech", new { model = "tts", input = "Hello", response_format = format });

        await AssertRefusedNamingAsync(response, "response_format");
    }

    [Theory]
    [InlineData(null)]
    [InlineData("wav")]
    [InlineData("WAV")]
    public async Task Speech_ResponseFormatWavOrOmitted_IsNotRefusedForItsFormat(string? format)
    {
        var (client, _, factory) = Host();
        using var _f = factory; using var _c = client;

        var body = new Dictionary<string, object?> { ["model"] = "tts", ["input"] = "Hello" };
        if (format is not null) body["response_format"] = format;
        var response = await client.PostAsJsonAsync("/v1/audio/speech", body);

        // No speech engine is configured in this host, so the request may fail for that reason, but never because of the format.
        string text = await response.Content.ReadAsStringAsync();
        Assert.DoesNotContain("response_format", text, StringComparison.Ordinal);
    }

    // ── /v1/images/*: not implemented, and they say so (they used to answer 200 with a prompt-seeded gradient) ──

    [Fact]
    public async Task ImageGenerations_ValidRequest_Is501_NotAFakeImage_AndPointsAtTheCommandLine()
    {
        var (client, _, factory) = Host();
        using var _f = factory; using var _c = client;

        var response = await client.PostAsJsonAsync("/v1/images/generations", new { model = "sd", prompt = "a red bicycle", n = 1, size = "512x512" });

        Assert.Equal(HttpStatusCode.NotImplemented, response.StatusCode);
        string text = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(text);
        Assert.Equal("not_implemented", doc.RootElement.GetProperty("type").GetString());
        Assert.Contains("stingray image", doc.RootElement.GetProperty("message").GetString());
        Assert.False(doc.RootElement.TryGetProperty("data", out _), "a 501 must not carry image data");
        Assert.DoesNotContain("b64_json", text, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ImageGenerations_MissingPrompt_AndBadJson_AreStill400()
    {
        var (client, _, factory) = Host();
        using var _f = factory; using var _c = client;

        await AssertRefusedNamingAsync(await client.PostAsJsonAsync("/v1/images/generations", new { model = "sd" }), "prompt");
        using var bad = new StringContent("{ not json", Encoding.UTF8, "application/json");
        Assert.Equal(HttpStatusCode.BadRequest, (await client.PostAsync("/v1/images/generations", bad)).StatusCode);
    }

    [Fact]
    public async Task ImageEdits_ValidMultipart_Is501_AndTheRequestChecksStay400()
    {
        var (client, _, factory) = Host();
        using var _f = factory; using var _c = client;

        using var withPrompt = new MultipartFormDataContent { { new StringContent("make it blue"), "prompt" } };
        var response = await client.PostAsync("/v1/images/edits", withPrompt);
        Assert.Equal(HttpStatusCode.NotImplemented, response.StatusCode);
        Assert.DoesNotContain("b64_json", await response.Content.ReadAsStringAsync(), StringComparison.Ordinal);

        using var noPrompt = new MultipartFormDataContent { { new StringContent("x"), "size" } };
        Assert.Equal(HttpStatusCode.BadRequest, (await client.PostAsync("/v1/images/edits", noPrompt)).StatusCode);
        using var notMultipart = new StringContent("{}", Encoding.UTF8, "application/json");
        Assert.Equal(HttpStatusCode.BadRequest, (await client.PostAsync("/v1/images/edits", notMultipart)).StatusCode);
    }

    // ── /v1/audio/transcriptions and /translations: client mistakes are 400s that name the problem (a non-WAV body used to be a 500) ──

    private static byte[] TruncatedRiff() => "RIFF"u8.ToArray().Concat(new byte[] { 0x24, 0, 0, 0 }).Concat("WAVEfmt "u8.ToArray()).ToArray();

    [Theory]
    [InlineData("/v1/audio/transcriptions")]
    [InlineData("/v1/audio/translations")]
    public async Task SpeechToText_NonWavBody_IsA400_NotA500(string route)
    {
        var (client, _, factory) = Host();
        using var _f = factory; using var _c = client;

        using var junk = new ByteArrayContent(Enumerable.Range(1, 64).Select(i => (byte)i).ToArray());
        junk.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("audio/wav");
        var direct = await client.PostAsync(route, junk);
        await AssertRefusedNamingAsync(direct, "WAV");

        using var form = new MultipartFormDataContent { { new ByteArrayContent(Enumerable.Range(1, 64).Select(i => (byte)i).ToArray()), "file", "a.wav" } };
        await AssertRefusedNamingAsync(await client.PostAsync(route, form), "WAV");
    }

    [Fact]
    public async Task SpeechToText_TruncatedRiffHeader_IsA400_NotA500()
    {
        var (client, _, factory) = Host();
        using var _f = factory; using var _c = client;

        using var form = new MultipartFormDataContent { { new ByteArrayContent(TruncatedRiff()), "file", "a.wav" } };
        await AssertRefusedNamingAsync(await client.PostAsync("/v1/audio/transcriptions", form), "WAV");
    }

    [Theory]
    [InlineData("response_format", "bogus", "response_format")]
    [InlineData("temperature", "abc", "temperature")]
    [InlineData("temperature", "2", "temperature")]
    [InlineData("temperature", "-0.1", "temperature")]
    [InlineData("stream", "maybe", "stream")]
    [InlineData("vad", "perhaps", "vad")]
    public async Task SpeechToText_BadParameter_IsA400_NamingTheField(string field, string value, string named)
    {
        var (client, _, factory) = Host();
        using var _f = factory; using var _c = client;

        using var form = new MultipartFormDataContent { { new StringContent(value), field } };
        await AssertRefusedNamingAsync(await client.PostAsync("/v1/audio/transcriptions", form), named);
    }

    [Fact]
    public async Task SpeechToText_ValidParameters_AreNotRefused_TheMissingAudioIsWhatIsReported()
    {
        var (client, _, factory) = Host();
        using var _f = factory; using var _c = client;

        using var form = new MultipartFormDataContent
        {
            { new StringContent("srt"), "response_format" }, { new StringContent("0.2"), "temperature" },
            { new StringContent("true"), "vad" }, { new StringContent("false"), "stream" },
        };
        var response = await client.PostAsync("/v1/audio/transcriptions", form);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        string text = await response.Content.ReadAsStringAsync();
        Assert.Contains("Audio file or payload is required", text, StringComparison.Ordinal);
        Assert.DoesNotContain("temperature", text, StringComparison.Ordinal);
        Assert.DoesNotContain("response_format", text, StringComparison.Ordinal);
    }
}