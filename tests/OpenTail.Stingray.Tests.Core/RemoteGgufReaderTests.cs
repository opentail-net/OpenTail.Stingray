using System.Net;
using System.Net.Http.Headers;
using OpenTail.Stingray.Core.Net;
using static OpenTail.Stingray.Tests.Core.SyntheticGguf;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// The bounded remote index reader against a fake Hub that really implements Range, plus misbehaving variants for each guardrail.
/// No network is touched.
/// </summary>
public sealed class RemoteGgufReaderTests
{
    private const string Sha = "0123456789abcdef0123456789abcdef01234567";

    public enum Quirk { None, IgnoreRange, WrongStart, ChangeTotal, Forbidden, NotFound, OverServe, GarbageContentRange }

    /// <summary>Hub on huggingface.co 302-redirects to a CDN host that serves ranges, like the real one.</summary>
    private sealed class FakeHub(Dictionary<string, byte[]> files, Quirk quirk = Quirk.None) : HttpMessageHandler
    {
        public List<(string Host, string Path, long? From, long? To)> Requests { get; } = [];
        public long BodyBytesServed { get; private set; }
        private int _rangeCalls;

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage req, CancellationToken ct)
        {
            var uri = req.RequestUri!;
            var range = req.Headers.Range?.Ranges.FirstOrDefault();
            Requests.Add((uri.Host, uri.AbsolutePath, range?.From, range?.To));

            if (uri.Host == "huggingface.co")
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.Found) { Headers = { Location = new Uri("https://cdn-lfs.huggingface.co" + uri.AbsolutePath) } });

            if (quirk == Quirk.Forbidden) return Task.FromResult(new HttpResponseMessage(HttpStatusCode.Forbidden));
            if (quirk == Quirk.NotFound) return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));

            string key = uri.AbsolutePath;
            if (!files.TryGetValue(key, out var bytes)) return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));

            if (quirk == Quirk.IgnoreRange || range?.From is null)
            {
                BodyBytesServed += bytes.Length;
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(bytes) });
            }

            long from = range.From.Value;
            long to = Math.Min(range.To ?? bytes.Length - 1, bytes.Length - 1);
            if (quirk == Quirk.OverServe) to = Math.Min(bytes.Length - 1, to + 1000);
            long total = bytes.Length;
            _rangeCalls++;
            if (quirk == Quirk.ChangeTotal && _rangeCalls > 1) total += 7;
            long reportedFrom = quirk == Quirk.WrongStart ? from + 1 : from;

            int len = (int)(to - from + 1);
            var body = new byte[len]; Array.Copy(bytes, from, body, 0, len);
            BodyBytesServed += len;
            var resp = new HttpResponseMessage(HttpStatusCode.PartialContent) { Content = new ByteArrayContent(body) };
            resp.Content.Headers.ContentRange = quirk == Quirk.GarbageContentRange ? null : new ContentRangeHeaderValue(reportedFrom, to, total);
            return Task.FromResult(resp);
        }
    }

    private static string Key(string path) => $"/o/r/resolve/{Sha}/{path}";

    private static ExternalHttpClient Client(FakeHub hub, Func<string, string?>? env = null) => new(inner: hub, env: env ?? (_ => null));

    /// <summary>A valid GGUF whose index is roughly <paramref name="indexKb"/> KB, followed by <paramref name="dataMb"/> MB of "weights".</summary>
    private static byte[] Model(int indexKb, int dataMb, string arch = "llama") => Build(3,
        [KvStr("general.architecture", arch), indexKb > 0 ? KvBigStrArray("tokenizer.ggml.tokens", indexKb, 1000 - 8 - 0) : KvU32("llama.block_count", 2)],
        [new("token_embd.weight", [8, 3], 0, 0), new("blk.0.attn_q.weight", [8, 8], 0, 96)], dataBytes: Math.Max(dataMb << 20, 4096));

    // ── the happy paths ──
    [Fact]
    public async Task A_small_model_is_read_in_one_chunk_through_the_redirect()
    {
        byte[] file = Model(0, 0);
        var hub = new FakeHub(new() { [Key("m.gguf")] = file });
        using var c = Client(hub);
        var r = await RemoteGgufReader.ReadAsync(c, "o/r", Sha, ["m.gguf"], RemoteGgufReader.DefaultMaxIndexBytes, default);
        Assert.True(r.Outcome == RemoteIndexOutcome.Complete, r.Outcome + ": " + r.Detail);
        Assert.Equal(2, r.Index!.Tensors.Count);
        Assert.Equal(file.Length, r.ShardSizes[0]);
        Assert.Equal(["cdn-lfs.huggingface.co", "huggingface.co"], c.Log.Hosts);
    }

    [Fact]
    public async Task A_big_model_with_a_small_index_costs_one_chunk_and_never_a_byte_beyond_it()
    {
        byte[] file = Model(0, 6); // ~6 MB of weights after a tiny index
        var hub = new FakeHub(new() { [Key("m.gguf")] = file });
        using var c = Client(hub);
        var r = await RemoteGgufReader.ReadAsync(c, "o/r", Sha, ["m.gguf"], RemoteGgufReader.DefaultMaxIndexBytes, default);
        Assert.True(r.Outcome == RemoteIndexOutcome.Complete, r.Outcome + ": " + r.Detail);
        Assert.Equal(RemoteGgufReader.InitialChunkBytes, r.BytesRead);
        Assert.True(hub.BodyBytesServed <= RemoteGgufReader.InitialChunkBytes, $"the fake server was asked for {hub.BodyBytesServed} bytes of a {file.Length}-byte file");
        Assert.All(hub.Requests.Where(q => q.To is not null), q => Assert.True(q.To < RemoteGgufReader.InitialChunkBytes));
        Assert.True(c.Log.BytesReceived < file.Length / 2);
    }

    [Fact]
    public async Task An_index_larger_than_the_first_chunk_is_extended_by_ranges_until_it_parses()
    {
        byte[] file = Model(indexKb: 3500, dataMb: 4); // ~3.5 MB of token strings => index > 2 MB first chunk
        var hub = new FakeHub(new() { [Key("m.gguf")] = file });
        using var c = Client(hub);
        var r = await RemoteGgufReader.ReadAsync(c, "o/r", Sha, ["m.gguf"], RemoteGgufReader.DefaultMaxIndexBytes, default);
        Assert.True(r.Outcome == RemoteIndexOutcome.Complete, r.Outcome + ": " + r.Detail);
        Assert.True(r.BytesRead > RemoteGgufReader.InitialChunkBytes);
        Assert.True(r.BytesRead < file.Length, "must not have read the weights");
        var ranged = hub.Requests.Where(q => q.From is not null && q.Host != "huggingface.co").ToList();
        Assert.True(ranged.Count >= 2);
        Assert.Equal(0, ranged[0].From);
        Assert.True(ranged[1].From > 0, "the second request continues where the first ended, it does not start over");
    }

    [Fact]
    public async Task Requests_are_pinned_to_the_commit_never_to_a_branch()
    {
        var hub = new FakeHub(new() { [Key("sub%20dir/m.gguf")] = Model(0, 0) });
        using var c = Client(hub);
        var r = await RemoteGgufReader.ReadAsync(c, "o/r", Sha, ["sub dir/m.gguf"], RemoteGgufReader.DefaultMaxIndexBytes, default);
        Assert.True(r.Outcome == RemoteIndexOutcome.Complete, r.Outcome + ": " + r.Detail);
        Assert.All(hub.Requests, q => { Assert.Contains(Sha, q.Path); Assert.DoesNotContain("/main/", q.Path); });
    }

    [Fact]
    public async Task A_split_model_reads_every_shard_and_merges_them()
    {
        byte[] s0 = Build(3, [KvStr("general.architecture", "llama")], [new("token_embd.weight", [8, 3], 0, 0)], 3 << 20);
        byte[] s1 = Build(3, [KvStr("general.architecture", "llama")], [new("blk.0.attn_q.weight", [8, 8], 0, 0)], 3 << 20);
        var hub = new FakeHub(new() { [Key("m-00001-of-00002.gguf")] = s0, [Key("m-00002-of-00002.gguf")] = s1 });
        using var c = Client(hub);
        var r = await RemoteGgufReader.ReadAsync(c, "o/r", Sha, ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"], RemoteGgufReader.DefaultMaxIndexBytes, default);
        Assert.True(r.Outcome == RemoteIndexOutcome.Complete, r.Outcome + ": " + r.Detail);
        Assert.Equal([0, 1], r.Index!.Tensors.Select(t => t.ShardIndex));
        Assert.Equal([s0.Length, s1.Length], r.ShardSizes.Select(x => (int)x));
    }

    // ── the guardrails ──
    [Fact]
    public async Task Past_the_cap_the_answer_is_incomplete_and_nothing_further_is_fetched()
    {
        byte[] file = Model(indexKb: 3500, dataMb: 4);
        var hub = new FakeHub(new() { [Key("m.gguf")] = file });
        using var c = Client(hub);
        const long cap = 2L << 20;
        var r = await RemoteGgufReader.ReadAsync(c, "o/r", Sha, ["m.gguf"], cap, default);
        Assert.Equal(RemoteIndexOutcome.IncompleteOverCap, r.Outcome);
        Assert.Null(r.Index);
        Assert.True(r.BytesRead <= cap);
        Assert.True(hub.BodyBytesServed <= cap);
        Assert.Contains("not downloaded", r.Detail);
    }

    [Fact]
    public async Task A_server_that_ignores_Range_and_offers_a_big_file_is_refused_without_reading_the_body()
    {
        var hub = new FakeHub(new() { [Key("m.gguf")] = Model(0, 6) }, Quirk.IgnoreRange);
        using var c = Client(hub);
        var r = await RemoteGgufReader.ReadAsync(c, "o/r", Sha, ["m.gguf"], RemoteGgufReader.DefaultMaxIndexBytes, default);
        Assert.Equal(RemoteIndexOutcome.Failed, r.Outcome);
        Assert.Contains("ignored the Range", r.Detail);
        Assert.Equal(0, c.Log.BytesReceived); // not one body byte was consumed
    }

    [Fact]
    public async Task A_server_that_ignores_Range_for_a_file_smaller_than_the_chunk_is_accepted()
    {
        var hub = new FakeHub(new() { [Key("m.gguf")] = Model(0, 0) }, Quirk.IgnoreRange);
        using var c = Client(hub);
        var r = await RemoteGgufReader.ReadAsync(c, "o/r", Sha, ["m.gguf"], RemoteGgufReader.DefaultMaxIndexBytes, default);
        Assert.True(r.Outcome == RemoteIndexOutcome.Complete, r.Outcome + ": " + r.Detail);
    }

    [Theory]
    [InlineData(Quirk.WrongStart, "does not match")]
    [InlineData(Quirk.GarbageContentRange, "Content-Range")]
    [InlineData(Quirk.OverServe, "does not match")]
    public async Task A_ranged_answer_that_does_not_match_the_request_is_refused(Quirk quirk, string expected)
    {
        var hub = new FakeHub(new() { [Key("m.gguf")] = Model(0, 6) }, quirk);
        using var c = Client(hub);
        var r = await RemoteGgufReader.ReadAsync(c, "o/r", Sha, ["m.gguf"], RemoteGgufReader.DefaultMaxIndexBytes, default);
        Assert.Equal(RemoteIndexOutcome.Failed, r.Outcome);
        Assert.Contains(expected, r.Detail);
    }

    [Fact]
    public async Task A_file_whose_size_changes_between_requests_is_refused()
    {
        var hub = new FakeHub(new() { [Key("m.gguf")] = Model(indexKb: 3500, dataMb: 4) }, Quirk.ChangeTotal);
        using var c = Client(hub);
        var r = await RemoteGgufReader.ReadAsync(c, "o/r", Sha, ["m.gguf"], RemoteGgufReader.DefaultMaxIndexBytes, default);
        Assert.Equal(RemoteIndexOutcome.Failed, r.Outcome);
        Assert.Contains("changed between requests", r.Detail);
    }

    // ── errors ──
    [Fact]
    public async Task Access_restricted_and_missing_files_are_distinct_outcomes()
    {
        using var c1 = Client(new FakeHub(new() { [Key("m.gguf")] = Model(0, 0) }, Quirk.Forbidden));
        var r1 = await RemoteGgufReader.ReadAsync(c1, "o/r", Sha, ["m.gguf"], RemoteGgufReader.DefaultMaxIndexBytes, default);
        Assert.Equal(RemoteIndexOutcome.AccessRestricted, r1.Outcome);
        Assert.Contains("HF_TOKEN", r1.Detail);

        using var c2 = Client(new FakeHub(new(), Quirk.None));
        var r2 = await RemoteGgufReader.ReadAsync(c2, "o/r", Sha, ["absent.gguf"], RemoteGgufReader.DefaultMaxIndexBytes, default);
        Assert.Equal(RemoteIndexOutcome.NotFound, r2.Outcome);
    }

    [Fact]
    public async Task Something_that_is_not_a_gguf_is_a_failure_not_an_endless_read()
    {
        var hub = new FakeHub(new() { [Key("m.gguf")] = new byte[3 << 20] });
        using var c = Client(hub);
        var r = await RemoteGgufReader.ReadAsync(c, "o/r", Sha, ["m.gguf"], RemoteGgufReader.DefaultMaxIndexBytes, default);
        Assert.Equal(RemoteIndexOutcome.Failed, r.Outcome);
        Assert.Contains("Not a readable GGUF", r.Detail);
        Assert.Equal(RemoteGgufReader.InitialChunkBytes, r.BytesRead);
    }

    [Fact]
    public async Task A_file_that_ends_inside_its_own_index_is_reported_as_malformed()
    {
        byte[] file = Model(0, 0);
        var hub = new FakeHub(new() { [Key("m.gguf")] = file[..60] });
        using var c = Client(hub);
        var r = await RemoteGgufReader.ReadAsync(c, "o/r", Sha, ["m.gguf"], RemoteGgufReader.DefaultMaxIndexBytes, default);
        Assert.Equal(RemoteIndexOutcome.Failed, r.Outcome);
        Assert.Contains("ended inside its own index", r.Detail);
    }

    [Fact]
    public async Task A_denied_policy_throws_instead_of_being_reported_as_a_network_failure()
    {
        var hub = new FakeHub(new() { [Key("m.gguf")] = Model(0, 0) });
        using var c = Client(hub, env: n => n == "STINGRAY_ALLOW_EXTERNAL" ? "off" : null);
        await Assert.ThrowsAsync<ExternalAccessDeniedException>(() =>
            RemoteGgufReader.ReadAsync(c, "o/r", Sha, ["m.gguf"], RemoteGgufReader.DefaultMaxIndexBytes, default));
        Assert.Empty(hub.Requests);
    }

    [Fact]
    public async Task A_tiny_chunk_size_still_reaches_the_same_answer()
    {
        byte[] file = Model(0, 1);
        var hub = new FakeHub(new() { [Key("m.gguf")] = file });
        using var c = Client(hub);
        var r = await RemoteGgufReader.ReadAsync(c, "o/r", Sha, ["m.gguf"], RemoteGgufReader.DefaultMaxIndexBytes, default, initialChunkBytes: 40);
        Assert.True(r.Outcome == RemoteIndexOutcome.Complete, r.Outcome + ": " + r.Detail);
        Assert.Equal(2, r.Index!.Tensors.Count);
        Assert.True(hub.Requests.Count(q => q.Host != "huggingface.co") > 1);
    }

    [Fact]
    public async Task A_retired_storage_type_is_reported_as_unsupported_storage_not_as_a_read_failure()
    {
        byte[] file = Build(3, [KvStr("general.architecture", "llama")], [new("blk.0.ffn_down.weight", [8, 8], 31, 0)], 4096);
        var hub = new FakeHub(new() { [Key("m.gguf")] = file });
        using var c = Client(hub);
        var r = await RemoteGgufReader.ReadAsync(c, "o/r", Sha, ["m.gguf"], RemoteGgufReader.DefaultMaxIndexBytes, default);
        Assert.Equal(RemoteIndexOutcome.UnsupportedStorage, r.Outcome);
        Assert.Contains("unsupported GGML type ID 31", r.Detail);
        Assert.Null(r.Index);
    }
}