using System.Net;
using OpenTail.Stingray.Core.Catalog;
using OpenTail.Stingray.Core.Net;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>The single external-access policy, and the gateway every network call goes through. No test touches the network.</summary>
public sealed class ExternalAccessTests
{
    private static Func<string, string?> Env(params (string Name, string Value)[] vars)
    {
        var d = vars.ToDictionary(v => v.Name, v => v.Value, StringComparer.Ordinal);
        return n => d.TryGetValue(n, out var v) ? v : null;
    }

    // ── policy ──
    [Fact]
    public void Default_is_allowed()
    {
        var d = ExternalAccess.Evaluate(Env());
        Assert.True(d.Allowed);
        Assert.Contains("default", d.Reason);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("false")]
    [InlineData("OFF")]
    [InlineData("No")]
    [InlineData("deny")]
    [InlineData("  off  ")]
    public void Switching_it_off_denies_and_says_how_to_undo_it(string value)
    {
        var d = ExternalAccess.Evaluate(Env(("STINGRAY_ALLOW_EXTERNAL", value)));
        Assert.False(d.Allowed);
        Assert.Contains("STINGRAY_ALLOW_EXTERNAL", d.Reason);
        Assert.Contains("STINGRAY_ALLOW_EXTERNAL", d.HowToEnable);
    }

    [Theory]
    [InlineData("1")]
    [InlineData("true")]
    [InlineData("ON")]
    [InlineData("yes")]
    [InlineData("allow")]
    public void Stating_the_default_out_loud_allows(string value) =>
        Assert.True(ExternalAccess.Evaluate(Env(("STINGRAY_ALLOW_EXTERNAL", value))).Allowed);

    [Theory]
    [InlineData("STINGRAY_OFFLINE")]
    [InlineData("HF_HUB_OFFLINE")]
    public void Deny_wins_over_an_explicit_allow(string offlineVariable)
    {
        var d = ExternalAccess.Evaluate(Env((offlineVariable, "1"), ("STINGRAY_ALLOW_EXTERNAL", "1")));
        Assert.False(d.Allowed);
        Assert.Contains(offlineVariable, d.Reason);
    }

    [Theory]
    [InlineData("maybe")]
    [InlineData("2")]
    [InlineData("enabled")]
    public void An_unrecognised_value_is_a_deny_not_a_guess(string value)
    {
        var d = ExternalAccess.Evaluate(Env(("STINGRAY_ALLOW_EXTERNAL", value)));
        Assert.False(d.Allowed);
        Assert.Contains("not recognised", d.Reason);
    }

    [Fact]
    public void Offline_variables_set_to_zero_do_not_deny()
    {
        Assert.True(ExternalAccess.Evaluate(Env(("STINGRAY_OFFLINE", "0"), ("HF_HUB_OFFLINE", "false"))).Allowed);
        Assert.False(ExternalAccess.OfflineFromEnvironment(Env(("STINGRAY_OFFLINE", "0"))));
        Assert.True(ExternalAccess.OfflineFromEnvironment(Env(("HF_HUB_OFFLINE", "TRUE"))));
    }

    // ── hosts and token ──
    [Theory]
    [InlineData("huggingface.co", true)]
    [InlineData("HuggingFace.co", true)]
    [InlineData("hf.co", true)]
    [InlineData("cdn-lfs.huggingface.co", true)]
    [InlineData("cas-bridge.xethub.hf.co", true)]
    [InlineData("huggingface.co.", true)]
    [InlineData("evilhuggingface.co", false)]
    [InlineData("huggingface.co.evil.example", false)]
    [InlineData("nothf.co", false)]
    [InlineData("example.com", false)]
    [InlineData("", false)]
    [InlineData(null, false)]
    public void Only_hugging_face_hosts_are_contactable(string? host, bool expected) =>
        Assert.Equal(expected, ExternalAccess.IsAllowedHost(host));

    [Theory]
    [InlineData("huggingface.co", true)]
    [InlineData("www.huggingface.co", true)]
    [InlineData("cdn-lfs.huggingface.co", false)]
    [InlineData("cas-bridge.xethub.hf.co", false)]
    [InlineData("hf.co", false)]
    public void The_token_goes_only_to_the_hub(string host, bool expected) =>
        Assert.Equal(expected, ExternalAccess.MayReceiveToken(host));

    // ── gateway ──
    private sealed class FakeHandler(Func<HttpRequestMessage, HttpResponseMessage> respond) : HttpMessageHandler
    {
        public List<(string Host, string Path, string? Auth, string? Range)> Seen { get; } = [];
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
        {
            Seen.Add((request.RequestUri!.Host, request.RequestUri.AbsolutePath, request.Headers.Authorization?.ToString(), request.Headers.Range?.ToString()));
            return Task.FromResult(respond(request));
        }
    }

    private static HttpResponseMessage Ok(byte[] body) => new(HttpStatusCode.OK) { Content = new ByteArrayContent(body) };
    private static HttpResponseMessage Redirect(string to) => new(HttpStatusCode.Found) { Headers = { Location = new Uri(to) } };

    private static ExternalHttpClient Client(FakeHandler h, params (string, string)[] env) =>
        new(inner: h, env: Env(env));

    [Fact]
    public async Task A_denied_policy_stops_the_request_before_it_is_made()
    {
        var h = new FakeHandler(_ => Ok([1]));
        using var c = Client(h, ("STINGRAY_ALLOW_EXTERNAL", "off"));
        var ex = await Assert.ThrowsAsync<ExternalAccessDeniedException>(() => c.GetAsync(new Uri("https://huggingface.co/api/models/x/y"), default));
        Assert.Contains("STINGRAY_ALLOW_EXTERNAL", ex.Message);
        Assert.Empty(h.Seen);
        Assert.Equal(0, c.Log.Requests);
    }

    [Fact]
    public async Task The_policy_is_checked_on_every_request_not_once()
    {
        var vars = new Dictionary<string, string>();
        var h = new FakeHandler(_ => Ok([1, 2, 3]));
        using var c = new ExternalHttpClient(inner: h, env: n => vars.TryGetValue(n, out var v) ? v : null);
        using (var r = await c.GetAsync(new Uri("https://huggingface.co/api/models/x/y"), default)) Assert.True(r.IsSuccessStatusCode);
        vars["STINGRAY_ALLOW_EXTERNAL"] = "0";
        await Assert.ThrowsAsync<ExternalAccessDeniedException>(() => c.GetAsync(new Uri("https://huggingface.co/api/models/x/y"), default));
        Assert.Single(h.Seen);
    }

    [Fact]
    public async Task The_token_is_sent_to_the_hub_but_stripped_before_the_cdn_hop_and_the_range_follows()
    {
        var h = new FakeHandler(r => r.RequestUri!.Host == "huggingface.co"
            ? Redirect("https://cdn-lfs.huggingface.co/blob/abc")
            : Ok(new byte[16]));
        using var c = Client(h, ("HF_TOKEN", "hf_secret"));
        using var resp = await c.GetAsync(new Uri("https://huggingface.co/o/r/resolve/main/m.gguf"), default, rangeFrom: 0, rangeTo: 15);
        Assert.Equal(2, h.Seen.Count);
        Assert.Equal("Bearer hf_secret", h.Seen[0].Auth);
        Assert.Null(h.Seen[1].Auth);
        Assert.Equal("cdn-lfs.huggingface.co", h.Seen[1].Host);
        Assert.Equal("bytes=0-15", h.Seen[1].Range);
        Assert.Equal(["cdn-lfs.huggingface.co", "huggingface.co"], c.Log.Hosts);
    }

    [Fact]
    public async Task A_redirect_to_a_foreign_host_is_refused_without_contacting_it()
    {
        var h = new FakeHandler(_ => Redirect("https://evil.example/steal"));
        using var c = Client(h, ("HF_TOKEN", "hf_secret"));
        var ex = await Assert.ThrowsAsync<ExternalAccessDeniedException>(() => c.GetAsync(new Uri("https://huggingface.co/o/r/resolve/main/m.gguf"), default));
        Assert.Contains("evil.example", ex.Message);
        Assert.Single(h.Seen); // the foreign host was never requested
    }

    [Theory]
    [InlineData("http://huggingface.co/api/models/x/y")]
    [InlineData("https://example.com/")]
    [InlineData("https://huggingface.co.evil.example/")]
    public async Task Plain_http_and_foreign_hosts_are_refused_up_front(string url)
    {
        var h = new FakeHandler(_ => Ok([1]));
        using var c = Client(h);
        await Assert.ThrowsAsync<ExternalAccessDeniedException>(() => c.GetAsync(new Uri(url), default));
        Assert.Empty(h.Seen);
    }

    [Fact]
    public async Task A_redirect_loop_stops()
    {
        var h = new FakeHandler(_ => Redirect("https://huggingface.co/loop"));
        using var c = Client(h);
        await Assert.ThrowsAsync<HttpRequestException>(() => c.GetAsync(new Uri("https://huggingface.co/loop"), default));
        Assert.Equal(6, h.Seen.Count); // first request plus five followed redirects
    }

    [Fact]
    public async Task The_log_counts_requests_hosts_and_bytes_actually_read_and_holds_no_urls_or_tokens()
    {
        var h = new FakeHandler(_ => Ok(new byte[1000]));
        using var c = Client(h, ("HF_TOKEN", "hf_secret"));
        string body = await c.GetStringAsync(new Uri("https://huggingface.co/api/models/o/r?blobs=true"), default);
        Assert.Equal(1000, body.Length);
        Assert.Equal(1, c.Log.Requests);
        Assert.Equal(1000, c.Log.BytesReceived);
        Assert.Equal(["huggingface.co"], c.Log.Hosts);
        Assert.DoesNotContain("hf_secret", string.Join(" ", c.Log.Hosts));
    }

    [Fact]
    public async Task An_oversized_body_is_refused()
    {
        var h = new FakeHandler(_ => Ok(new byte[5000]));
        using var c = Client(h);
        await Assert.ThrowsAsync<InvalidDataException>(() => c.GetStringAsync(new Uri("https://huggingface.co/api/models/o/r"), default, maxBytes: 1000));
    }

    [Fact]
    public void The_catalog_offline_check_now_follows_the_shared_policy()
    {
        // ModelInstaller.OfflineFromEnvironment reads the real environment; here we only assert it agrees with Evaluate() for it.
        Assert.Equal(!ExternalAccess.Evaluate().Allowed, ModelInstaller.OfflineFromEnvironment());
    }
}
