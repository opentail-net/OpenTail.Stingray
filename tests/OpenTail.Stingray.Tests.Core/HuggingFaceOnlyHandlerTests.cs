using System.Net;
using OpenTail.Stingray.Core.Catalog;
using OpenTail.Stingray.Core.Net;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>The download client (<see cref="ModelDownloader.CreateClient"/>) obeys the same network policy as <see cref="ExternalHttpClient"/>.</summary>
public sealed class HuggingFaceOnlyHandlerTests
{
    private sealed class Recorder(Func<HttpRequestMessage, HttpResponseMessage> reply) : HttpMessageHandler
    {
        public List<(Uri Url, string? Auth, string? Range)> Seen { get; } = [];
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
        {
            Seen.Add((request.RequestUri!, request.Headers.Authorization?.ToString(), request.Headers.Range?.ToString()));
            return Task.FromResult(reply(request));
        }
    }

    private static HttpResponseMessage Redirect(string to) =>
        new(HttpStatusCode.Found) { Headers = { Location = new Uri(to) } };

    private static (HttpClient Client, Recorder Inner) Client(Func<HttpRequestMessage, HttpResponseMessage> reply, string? token = "hf_secret")
    {
        var inner = new Recorder(reply);
        var env = new Dictionary<string, string?> { ["HF_TOKEN"] = token };
        return (new HttpClient(new HuggingFaceOnlyHandler(inner, n => env.GetValueOrDefault(n))), inner);
    }

    [Theory]
    [InlineData("https://example.com/file.gguf")]
    [InlineData("http://huggingface.co/a/b/resolve/main/f.gguf")]
    [InlineData("https://huggingface.co.evil.test/a")]
    public async Task ARequestOutsideHttpsHuggingFace_IsRefused_BeforeAnyBytesLeave(string url)
    {
        var (http, inner) = Client(_ => new HttpResponseMessage(HttpStatusCode.OK));

        await Assert.ThrowsAsync<ExternalAccessDeniedException>(() => http.GetAsync(url, TestContext.Current.CancellationToken));

        Assert.Empty(inner.Seen);
    }

    [Fact]
    public async Task ARedirectToAnotherHost_IsRefused_AfterTheFirstHopOnly()
    {
        var (http, inner) = Client(_ => Redirect("https://evil.example.org/steal"));

        await Assert.ThrowsAsync<ExternalAccessDeniedException>(
            () => http.GetAsync("https://huggingface.co/a/b/resolve/main/f.gguf", TestContext.Current.CancellationToken));

        Assert.Single(inner.Seen);
    }

    [Fact]
    public async Task TheToken_GoesToTheHub_AndIsDroppedOnTheCdnHop_WhileTheRangeSurvives()
    {
        var (http, inner) = Client(r => r.RequestUri!.Host == "huggingface.co"
            ? Redirect("https://cdn-lfs.hf.co/blob/xyz")
            : new HttpResponseMessage(HttpStatusCode.PartialContent));
        using var request = new HttpRequestMessage(HttpMethod.Get, "https://huggingface.co/a/b/resolve/main/f.gguf");
        request.Headers.Authorization = new("Bearer", "hf_secret");
        request.Headers.Range = new(100, null);

        using var response = await http.SendAsync(request, TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.PartialContent, response.StatusCode);
        Assert.Equal(2, inner.Seen.Count);
        Assert.Equal("Bearer hf_secret", inner.Seen[0].Auth);
        Assert.Null(inner.Seen[1].Auth);
        Assert.Equal("bytes=100-", inner.Seen[1].Range);
    }

    [Fact]
    public async Task EndlessRedirects_StopAtTheCap()
    {
        var (http, inner) = Client(_ => Redirect("https://huggingface.co/loop"));

        await Assert.ThrowsAsync<HttpRequestException>(() => http.GetAsync("https://huggingface.co/loop", TestContext.Current.CancellationToken));

        Assert.Equal(6, inner.Seen.Count);
    }

    [Fact]
    public async Task OfflineMode_IsRefused_WithoutARequest()
    {
        var inner = new Recorder(_ => new HttpResponseMessage(HttpStatusCode.OK));
        var env = new Dictionary<string, string?> { ["STINGRAY_OFFLINE"] = "1" };
        using var http = new HttpClient(new HuggingFaceOnlyHandler(inner, n => env.GetValueOrDefault(n)));

        await Assert.ThrowsAsync<ExternalAccessDeniedException>(
            () => http.GetAsync("https://huggingface.co/a", TestContext.Current.CancellationToken));

        Assert.Empty(inner.Seen);
    }

    [Fact]
    public async Task TheClientSetupAndPullUse_IsPolicyEnforced_WithoutTouchingTheNetwork()
    {
        using var http = ModelDownloader.CreateClient("test");

        await Assert.ThrowsAsync<ExternalAccessDeniedException>(
            () => http.GetAsync("https://example.com/model.gguf", TestContext.Current.CancellationToken));
    }
}
