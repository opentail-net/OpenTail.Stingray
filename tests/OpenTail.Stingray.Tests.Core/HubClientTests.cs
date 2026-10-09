using System.Net;
using OpenTail.Stingray.Core.Net;

namespace OpenTail.Stingray.Tests.Core;

public sealed class HubClientTests
{
    // Trimmed from a real response (bartowski/SmolLM2-135M-Instruct-GGUF, ?blobs=true), 2026-10-09.
    private const string Sample = """
    {
      "_id": "abc", "id": "bartowski/SmolLM2-135M-Instruct-GGUF", "private": false, "gated": false, "downloads": 42063,
      "sha": "09816acd5d99df7be770d85ea30822623dab342c",
      "tags": ["gguf","text-generation","en","license:apache-2.0"],
      "cardData": { "license": "apache-2.0" },
      "gguf": { "total": 134515008, "architecture": "llama", "context_length": 8192, "totalFileSize": 270885952, "chat_template": "..." },
      "siblings": [
        { "rfilename": ".gitattributes", "size": 3124 },
        { "rfilename": "README.md", "size": 10039 },
        { "rfilename": "SmolLM2-135M-Instruct-IQ3_M.gguf", "size": 90213472, "lfs": { "sha256": "E8A0E942FE93529B7601A7620C8AA1BDC1BEDE40FF668ED9D6339686384392D7", "size": 90213472, "pointerSize": 134 } },
        { "rfilename": "SmolLM2-135M-Instruct-Q4_K_M.gguf", "size": 105454432, "lfs": { "sha256": "2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d", "size": 105454432 } },
        { "rfilename": "tiny.gguf", "size": 1000 }
      ]
    }
    """;

    private static HubRepo Parse(string json, string id = "o/r") { using var d = System.Text.Json.JsonDocument.Parse(json); return HubClient.ParseRepo(d.RootElement, id); }

    [Fact]
    public void A_real_shaped_response_is_parsed_with_provenance()
    {
        var r = Parse(Sample);
        Assert.Equal("bartowski/SmolLM2-135M-Instruct-GGUF", r.Id);
        Assert.Equal("09816acd5d99df7be770d85ea30822623dab342c", r.Revision);
        Assert.False(r.Private); Assert.Null(r.Gated);
        Assert.Equal(42063, r.Downloads30Days);
        Assert.Equal("apache-2.0", r.License);
        Assert.Equal("llama", r.Gguf!.Architecture);
        Assert.Equal(8192, r.Gguf.ContextLength);
        var f = r.Files.Single(x => x.Path.Contains("IQ3_M"));
        Assert.Equal(90213472, f.Size);
        Assert.Equal("e8a0e942fe93529b7601a7620c8aa1bdc1bede40ff668ed9d6339686384392d7", f.Sha256); // lower-cased
        Assert.Null(r.Files.Single(x => x.Path == "tiny.gguf").Sha256);                              // not published => null, not invented
    }

    [Theory]
    [InlineData("\"gated\": \"auto\"", "auto")]
    [InlineData("\"gated\": \"manual\"", "manual")]
    [InlineData("\"gated\": true", "true")]
    [InlineData("\"gated\": false", null)]
    public void Gated_is_reported_as_the_hub_says_it(string fragment, string? expected) =>
        Assert.Equal(expected, Parse("{" + fragment + ", \"sha\": \"" + new string('a', 40) + "\", \"siblings\": []}").Gated);

    [Fact]
    public void A_response_without_a_commit_sha_is_refused_because_nothing_could_be_pinned()
    {
        Assert.Throws<InvalidDataException>(() => Parse("{ \"id\": \"o/r\", \"siblings\": [] }"));
    }

    [Fact]
    public void Missing_or_odd_fields_become_null_not_guesses()
    {
        var r = Parse("{ \"sha\": \"" + new string('b', 40) + "\", \"downloads\": \"many\", \"siblings\": [ {\"rfilename\": 5}, {\"rfilename\": \"a.gguf\", \"lfs\": {\"sha256\": \"short\"}} ] }", "x/y");
        Assert.Equal("x/y", r.Id);
        Assert.Null(r.Downloads30Days); Assert.Null(r.License); Assert.Null(r.Gguf);
        var f = Assert.Single(r.Files);
        Assert.Null(f.Sha256); Assert.Null(f.Size);
    }

    // ── grouping ──
    private static HubFile F(string p, long? size = 1) => new(p, size, null);

    [Fact]
    public void Single_split_and_projector_files_are_grouped()
    {
        var models = HubClient.GroupModels([F("a-Q4_K_M.gguf"), F("big-00001-of-00003.gguf", 10), F("big-00002-of-00003.gguf", 20), F("big-00003-of-00003.gguf", 30), F("mmproj-a.gguf"), F("README.md")]);
        Assert.Equal(3, models.Count);
        var split = models.Single(m => m.IsSplit);
        Assert.Equal(["big-00001-of-00003.gguf", "big-00002-of-00003.gguf", "big-00003-of-00003.gguf"], split.Shards.Select(s => s.Path));
        Assert.Equal(60, split.TotalSize);
        Assert.True(HubClient.IsComplete(split));
        Assert.Null(split.Sha256); // one hash for several files would be a lie
        Assert.True(HubClient.IsProjector(models.Single(m => m.Name.StartsWith("mmproj"))));
        Assert.False(HubClient.IsProjector(models.Single(m => m.Name == "a-Q4_K_M.gguf")));
    }

    [Fact]
    public void A_split_model_with_a_missing_shard_is_flagged_incomplete()
    {
        var m = HubClient.GroupModels([F("big-00001-of-00003.gguf"), F("big-00003-of-00003.gguf")]).Single();
        Assert.False(HubClient.IsComplete(m));
    }

    [Fact]
    public void Sub_folders_keep_their_splits_apart()
    {
        var models = HubClient.GroupModels([F("q4/m-00001-of-00002.gguf"), F("q4/m-00002-of-00002.gguf"), F("q8/m-00001-of-00002.gguf"), F("q8/m-00002-of-00002.gguf")]);
        Assert.Equal(2, models.Count);
        Assert.All(models, m => Assert.Equal(2, m.Shards.Count));
    }

    [Fact]
    public void Total_size_is_unknown_when_any_shard_size_is_unknown()
    {
        Assert.Null(HubClient.GroupModels([F("m-00001-of-00002.gguf", 5), F("m-00002-of-00002.gguf", null)]).Single().TotalSize);
    }

    // ── repo ids and URLs ──
    [Theory]
    [InlineData("bartowski/Qwen2.5-7B-Instruct-GGUF", "bartowski/Qwen2.5-7B-Instruct-GGUF")]
    [InlineData("https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF", "bartowski/Qwen2.5-7B-Instruct-GGUF")]
    [InlineData("https://huggingface.co/o/r/tree/main/sub", "o/r")]
    [InlineData("  o/r  ", "o/r")]
    public void Valid_repo_ids_are_normalised(string input, string expected) => Assert.Equal(expected, HubClient.NormalizeRepoId(input));

    [Theory]
    [InlineData("")]
    [InlineData("noslash")]
    [InlineData("a/b/c")]
    [InlineData("../etc/passwd")]
    [InlineData("o/..")]
    [InlineData("o/r?x=1")]
    [InlineData("o/r name")]
    [InlineData("o/r%2f..")]
    [InlineData("https://evil.example/o/r")]
    [InlineData("https://huggingface.co.evil.example/o/r")]
    [InlineData("https://huggingface.co/onlyone")]
    public void Anything_else_is_not_a_repo_id(string input) => Assert.Null(HubClient.NormalizeRepoId(input));

    [Fact]
    public void Resolve_urls_are_pinned_and_escape_each_path_segment()
    {
        var u = HubClient.ResolveUrl("o/r", "abc123", "sub dir/My Model-Q4_K_M.gguf");
        Assert.Equal("https://huggingface.co/o/r/resolve/abc123/sub%20dir/My%20Model-Q4_K_M.gguf", u.AbsoluteUri); // the wire form; ToString() would display the spaces
        Assert.Throws<ArgumentException>(() => HubClient.ResolveUrl("../x", "abc", "f.gguf"));
    }

    // ── the API call ──
    private sealed class Handler(Func<HttpRequestMessage, HttpResponseMessage> respond) : HttpMessageHandler
    {
        public List<HttpRequestMessage> Seen { get; } = [];
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage r, CancellationToken ct) { Seen.Add(r); return Task.FromResult(respond(r)); }
    }

    [Fact]
    public async Task GetRepoAsync_asks_for_blobs_sends_the_token_to_the_hub_and_returns_the_pinned_revision()
    {
        var h = new Handler(_ => new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(Sample) });
        using var c = new ExternalHttpClient(inner: h, env: n => n == "HF_TOKEN" ? "hf_x" : null);
        var r = await HubClient.GetRepoAsync(c, "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF", null, default);
        Assert.Equal("09816acd5d99df7be770d85ea30822623dab342c", r.Revision);
        var req = Assert.Single(h.Seen);
        Assert.Equal("https://huggingface.co/api/models/bartowski/SmolLM2-135M-Instruct-GGUF?blobs=true", req.RequestUri!.ToString());
        Assert.Equal("Bearer hf_x", req.Headers.Authorization!.ToString());
    }

    [Fact]
    public async Task A_requested_revision_goes_into_the_path()
    {
        var h = new Handler(_ => new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(Sample) });
        using var c = new ExternalHttpClient(inner: h, env: _ => null);
        await HubClient.GetRepoAsync(c, "o/r", "refs/pr/3", default);
        Assert.Equal("https://huggingface.co/api/models/o/r/revision/refs%2Fpr%2F3?blobs=true", h.Seen.Single().RequestUri!.ToString());
    }

    [Fact]
    public async Task A_bad_repo_id_never_reaches_the_network()
    {
        var h = new Handler(_ => new HttpResponseMessage(HttpStatusCode.OK));
        using var c = new ExternalHttpClient(inner: h, env: _ => null);
        await Assert.ThrowsAsync<ArgumentException>(() => HubClient.GetRepoAsync(c, "../../x", null, default));
        Assert.Empty(h.Seen);
    }

    [Fact]
    public async Task A_not_found_repo_surfaces_as_an_http_error()
    {
        var h = new Handler(_ => new HttpResponseMessage(HttpStatusCode.NotFound));
        using var c = new ExternalHttpClient(inner: h, env: _ => null);
        await Assert.ThrowsAsync<HttpRequestException>(() => HubClient.GetRepoAsync(c, "o/missing", null, default));
    }
}
