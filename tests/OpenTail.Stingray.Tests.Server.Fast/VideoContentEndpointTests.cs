namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Phase 5.3 (server path): a <c>video_url</c> content part is decoded at request time. A payload that is not a usable
/// base64 MP4 is refused with HTTP 400 before any engine call, whether or not ffmpeg is installed on the test host.
/// Fake engine, no checkpoint.
/// </summary>
public sealed class VideoContentEndpointTests : IDisposable
{
    private readonly List<WebApplicationFactory<Program>> _factories = new();

    public void Dispose()
    {
        foreach (var factory in _factories) factory.Dispose();
    }

    private HttpClient CreateClient(FakeInferenceEngine fake)
    {
        var factory = new WebApplicationFactory<Program>()
            .WithWebHostBuilder(b => b.ConfigureServices(s =>
            {
                s.Configure<OpenTailStingrayServerOptions>(o => o.Architecture = "qwen4exp");
                s.AddSingleton<IInferenceEngine>(fake);
            }));
        _factories.Add(factory);
        return factory.CreateClient();
    }

    private static async Task<HttpResponseMessage> PostVideo(HttpClient client, string url)
    {
        return await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "qwen4exp",
            messages = new object[]
            {
                new
                {
                    role = "user",
                    content = new object[]
                    {
                        new { type = "text", text = "What happens in this clip?" },
                        new { type = "video_url", video_url = new { url } },
                    },
                },
            },
            max_tokens = 20,
            stream = false,
        });
    }

    [Fact]
    public async Task VideoUrl_NotAnMp4DataUrl_IsRefusedWith400_BeforeAnyEngineCall()
    {
        var fake = new FakeInferenceEngine("qwen4exp", [(GenerateChunkKind.Text, "unreached")]);
        var client = CreateClient(fake);

        var response = await PostVideo(client, "data:image/png;base64,AAAA");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var body = await response.Content.ReadAsStringAsync();
        Assert.Contains("invalid_request_error", body, StringComparison.Ordinal);
        Assert.Contains("only base64-encoded MP4 video is supported", body, StringComparison.Ordinal);
    }

    [Fact]
    public async Task VideoUrl_BadBase64_IsRefusedWith400()
    {
        var fake = new FakeInferenceEngine("qwen4exp", [(GenerateChunkKind.Text, "unreached")]);
        var client = CreateClient(fake);

        var response = await PostVideo(client, "data:video/mp4;base64,%%%not-base64%%%");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var body = await response.Content.ReadAsStringAsync();
        Assert.Contains("video data is not valid base64", body, StringComparison.Ordinal);
    }

    [Fact]
    public async Task VideoUrl_GarbageMp4_IsRefusedWith400_WhetherOrNotFfmpegIsInstalled()
    {
        // Without ffmpeg the server says so; with it, the decode fails or finds no frames. Every path is a 400 about the video.
        var fake = new FakeInferenceEngine("qwen4exp", [(GenerateChunkKind.Text, "unreached")]);
        var client = CreateClient(fake);
        string payload = Convert.ToBase64String(new byte[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });

        var response = await PostVideo(client, "data:video/mp4;base64," + payload);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var body = await response.Content.ReadAsStringAsync();
        Assert.Contains("video", body, StringComparison.OrdinalIgnoreCase);
    }
}
