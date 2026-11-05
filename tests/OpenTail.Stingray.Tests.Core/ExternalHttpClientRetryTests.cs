using System.Net;
using System.Net.Http.Headers;
using OpenTail.Stingray.Core.Net;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// scout backlog: "rate behaviour for the Hub (retries, 429)". A 429 or 503 is retried up to three times, waiting for Retry-After (capped at 30 s) or
/// 1, 2, 4 s when there is none; anything else, and the answer after the last retry, goes to the caller unchanged. The delay is injected, so the
/// tests run instantly and record what the client would have waited.
/// </summary>
public sealed class ExternalHttpClientRetryTests
{
    private static readonly Uri Url = new("https://huggingface.co/api/models/owner/repo");

    private sealed class ScriptedHandler(params Func<HttpResponseMessage>[] script) : HttpMessageHandler
    {
        public int Calls;
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            int i = Interlocked.Increment(ref Calls) - 1;
            return Task.FromResult(script[Math.Min(i, script.Length - 1)]());
        }
    }

    private static HttpResponseMessage Status(HttpStatusCode code, TimeSpan? retryAfter = null)
    {
        var r = new HttpResponseMessage(code) { Content = new StringContent("{}") };
        if (retryAfter is { } d) r.Headers.RetryAfter = new RetryConditionHeaderValue(d);
        return r;
    }

    private static (ExternalHttpClient Http, List<TimeSpan> Waits) Client(ScriptedHandler handler)
    {
        var waits = new List<TimeSpan>();
        var http = new ExternalHttpClient(userAgent: "test", inner: handler,
            env: key => key == "STINGRAY_ALLOW_EXTERNAL" ? "1" : null,
            delay: (d, _) => { waits.Add(d); return Task.CompletedTask; });
        return (http, waits);
    }

    [Fact]
    public async Task A_429_with_Retry_After_waits_that_long_and_then_succeeds()
    {
        var handler = new ScriptedHandler(() => Status((HttpStatusCode)429, TimeSpan.FromSeconds(2)), () => Status(HttpStatusCode.OK));
        var (http, waits) = Client(handler);
        using (http)
        {
            using var response = await http.GetAsync(Url, CancellationToken.None);
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        }
        Assert.Equal(2, handler.Calls);
        Assert.Equal([TimeSpan.FromSeconds(2)], waits);
    }

    [Fact]
    public async Task A_503_without_Retry_After_backs_off_one_two_four_seconds()
    {
        var handler = new ScriptedHandler(
            () => Status(HttpStatusCode.ServiceUnavailable), () => Status(HttpStatusCode.ServiceUnavailable),
            () => Status(HttpStatusCode.ServiceUnavailable), () => Status(HttpStatusCode.OK));
        var (http, waits) = Client(handler);
        using (http)
        {
            using var response = await http.GetAsync(Url, CancellationToken.None);
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        }
        Assert.Equal(4, handler.Calls);
        Assert.Equal([TimeSpan.FromSeconds(1), TimeSpan.FromSeconds(2), TimeSpan.FromSeconds(4)], waits);
    }

    [Fact]
    public async Task After_three_retries_the_last_429_goes_to_the_caller_and_GetString_throws()
    {
        var handler = new ScriptedHandler(() => Status((HttpStatusCode)429, TimeSpan.FromSeconds(1)));
        var (http, waits) = Client(handler);
        using (http)
        {
            using (var response = await http.GetAsync(Url, CancellationToken.None))
                Assert.Equal((HttpStatusCode)429, response.StatusCode);
            Assert.Equal(4, handler.Calls);            // the first try and three retries
            Assert.Equal(3, waits.Count);

            await Assert.ThrowsAsync<HttpRequestException>(() => http.GetStringAsync(Url, CancellationToken.None));
        }
    }

    [Fact]
    public async Task A_huge_Retry_After_is_capped_at_thirty_seconds()
    {
        var handler = new ScriptedHandler(() => Status((HttpStatusCode)429, TimeSpan.FromHours(1)), () => Status(HttpStatusCode.OK));
        var (http, waits) = Client(handler);
        using (http)
        {
            using var response = await http.GetAsync(Url, CancellationToken.None);
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        }
        Assert.Equal([TimeSpan.FromSeconds(30)], waits);
    }

    [Theory]
    [InlineData(HttpStatusCode.NotFound)]
    [InlineData(HttpStatusCode.Unauthorized)]
    [InlineData(HttpStatusCode.InternalServerError)]
    public async Task Other_failures_are_not_retried(HttpStatusCode code)
    {
        var handler = new ScriptedHandler(() => Status(code));
        var (http, waits) = Client(handler);
        using (http)
        {
            using var response = await http.GetAsync(Url, CancellationToken.None);
            Assert.Equal(code, response.StatusCode);
        }
        Assert.Equal(1, handler.Calls);
        Assert.Empty(waits);
    }
}
