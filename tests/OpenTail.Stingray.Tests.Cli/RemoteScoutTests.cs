using System.Net;
using System.Net.Http.Headers;
using System.Text;
using OpenTail.Stingray.Cli.Scout;
using OpenTail.Stingray.Core.Net;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// Remote scout against a fake Hub (API + redirect + Range). No network. The headline test builds one GGUF, scouts it both from disk and
/// through the fake Hub, and requires identical analysis.
/// </summary>
public sealed class RemoteScoutTests
{
    private const string Sha = "0123456789abcdef0123456789abcdef01234567";
    private const string Hash = "aaaabbbbccccddddaaaabbbbccccddddaaaabbbbccccddddaaaabbbbccccdddd";

    // ── a tiny GGUF builder (the Core tests have one, but it is internal to that project) ──
    private static void Str(BinaryWriter w, string s) { var b = Encoding.UTF8.GetBytes(s); w.Write((ulong)b.Length); w.Write(b); }

    private static byte[] Gguf(string arch, int blocks, int extraTokenBytes = 0, int dataBytes = 1 << 16, bool qkNorm = false)
    {
        var tensors = new List<(string, long[], uint, ulong)>();
        ulong off = 0;
        void Add(string n, long[] d, uint type = 0) { tensors.Add((n, d, type, off)); off += (ulong)(d.Aggregate(1L, (a, b) => a * b) * 4); off = (off + 31) / 32 * 32; }
        Add("token_embd.weight", [16, 8]);
        for (int i = 0; i < blocks; i++)
        {
            Add($"blk.{i}.attn_norm.weight", [16]); Add($"blk.{i}.attn_q.weight", [16, 16]); Add($"blk.{i}.attn_k.weight", [16, 8]);
            Add($"blk.{i}.attn_v.weight", [16, 8]); Add($"blk.{i}.attn_output.weight", [16, 16]);
            if (qkNorm) { Add($"blk.{i}.attn_q_norm.weight", [8]); Add($"blk.{i}.attn_k_norm.weight", [8]); }
            Add($"blk.{i}.ffn_norm.weight", [16]); Add($"blk.{i}.ffn_gate.weight", [16, 32]); Add($"blk.{i}.ffn_up.weight", [16, 32]); Add($"blk.{i}.ffn_down.weight", [32, 16]);
        }
        using var ms = new MemoryStream(); using var w = new BinaryWriter(ms);
        var kvs = new List<(string, uint, Action<BinaryWriter>)>
        {
            ("general.architecture", 8, x => Str(x, arch)),
            ($"{arch}.block_count", 4, x => x.Write((uint)blocks)),
            ($"{arch}.embedding_length", 4, x => x.Write(16u)),
            ($"{arch}.attention.head_count", 4, x => x.Write(2u)),
            ($"{arch}.attention.head_count_kv", 4, x => x.Write(1u)),
            ($"{arch}.feed_forward_length", 4, x => x.Write(32u)),
            ($"{arch}.context_length", 4, x => x.Write(256u)),
            ("tokenizer.ggml.model", 8, x => Str(x, "gpt2")),
            ("tokenizer.ggml.merges", 9, x => { x.Write(8u); x.Write(1UL); Str(x, "a b"); }),
            ("tokenizer.ggml.tokens", 9, x => { x.Write(8u); x.Write(2UL); Str(x, "a"); Str(x, new string('x', Math.Max(1, extraTokenBytes))); }),
        };
        w.Write(0x46554747u); w.Write(3u); w.Write((ulong)tensors.Count); w.Write((ulong)kvs.Count);
        foreach (var (k, t, v) in kvs) { Str(w, k); w.Write(t); v(w); }
        foreach (var (n, d, t, o) in tensors) { Str(w, n); w.Write((uint)d.Length); foreach (var x in d) w.Write((ulong)x); w.Write(t); w.Write(o); }
        while (ms.Length % 32 != 0) w.Write((byte)0);
        w.Write(new byte[Math.Max(dataBytes, (int)off + 64)]);
        return ms.ToArray();
    }

    // ── fake Hub: model API, 302 to a CDN, ranged files ──
    private sealed record RepoFile(string Path, byte[] Bytes, string? Sha256 = Hash, long? ListedSize = null);

    private sealed class FakeHub(string repo, IEnumerable<RepoFile> files, string? hubArch = "llama", string? gated = null, HttpStatusCode apiStatus = HttpStatusCode.OK) : HttpMessageHandler
    {
        private readonly List<RepoFile> _files = files.ToList();
        public List<string> Requests { get; } = [];
        public long BodyBytes { get; private set; }

        private string ApiJson()
        {
            var sb = new StringBuilder();
            sb.Append("{\"id\":\"").Append(repo).Append("\",\"private\":false,\"gated\":").Append(gated is null ? "false" : $"\"{gated}\"")
              .Append(",\"downloads\":1234,\"sha\":\"").Append(Sha).Append("\",\"tags\":[\"gguf\",\"license:mit\"]");
            if (hubArch is not null) sb.Append(",\"gguf\":{\"architecture\":\"").Append(hubArch).Append("\",\"total\":1}");
            sb.Append(",\"siblings\":[");
            sb.Append(string.Join(",", _files.Select(f =>
                $"{{\"rfilename\":\"{f.Path}\",\"size\":{f.ListedSize ?? f.Bytes.Length}" + (f.Sha256 is null ? "" : $",\"lfs\":{{\"sha256\":\"{f.Sha256}\",\"size\":{f.Bytes.Length}}}") + "}")));
            sb.Append("]}");
            return sb.ToString();
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage req, CancellationToken ct)
        {
            var uri = req.RequestUri!;
            Requests.Add(uri.Host + uri.AbsolutePath);
            if (uri.AbsolutePath.StartsWith("/api/models/", StringComparison.Ordinal))
                return Task.FromResult(apiStatus == HttpStatusCode.OK
                    ? new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(ApiJson()) }
                    : new HttpResponseMessage(apiStatus));
            if (uri.Host == "huggingface.co")
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.Found) { Headers = { Location = new Uri("https://cdn-lfs.huggingface.co" + uri.AbsolutePath) } });

            string prefix = $"/{repo}/resolve/{Sha}/";
            if (!uri.AbsolutePath.StartsWith(prefix, StringComparison.Ordinal)) return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
            string path = Uri.UnescapeDataString(uri.AbsolutePath[prefix.Length..]);
            var file = _files.FirstOrDefault(f => f.Path == path);
            if (file is null) return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));

            var range = req.Headers.Range!.Ranges.First();
            long from = range.From!.Value, to = Math.Min(range.To ?? file.Bytes.Length - 1, file.Bytes.Length - 1);
            var body = file.Bytes[(int)from..(int)(to + 1)];
            BodyBytes += body.Length;
            var resp = new HttpResponseMessage(HttpStatusCode.PartialContent) { Content = new ByteArrayContent(body) };
            resp.Content.Headers.ContentRange = new ContentRangeHeaderValue(from, to, file.Bytes.Length);
            return Task.FromResult(resp);
        }
    }

    private static ExternalHttpClient Client(FakeHub hub, Func<string, string?>? env = null) => new(inner: hub, env: env ?? (_ => null));
    private static readonly ScoutOptions Opts = new("test-build");
    private static Task<RemoteScoutResult> Run(ExternalHttpClient c, string? file = null, long max = RemoteGgufReader.DefaultMaxIndexBytes) =>
        RemoteScout.RunAsync(c, new RemoteScoutRequest("o/r", file, null, max), Opts, default);

    // ── the headline: remote analysis == local analysis of the same bytes ──
    [Fact]
    public async Task Scouting_through_the_hub_gives_the_same_analysis_as_scouting_the_file_on_disk()
    {
        byte[] bytes = Gguf("llama", 3, extraTokenBytes: 10, qkNorm: true);
        string path = Path.Combine(Path.GetTempPath(), "remote-vs-local-" + Guid.NewGuid().ToString("N") + ".gguf");
        File.WriteAllBytes(path, bytes);
        try
        {
            using var model = GgufModel.Open(path);
            var local = ScoutAnalyzer.Analyze(new ScoutInput("m.gguf", bytes.Length, model.Header.Version, model.Header.TensorCount, model.Header.MetadataKvCount, model.Metadata, model.Tensors), Opts);

            var hub = new FakeHub("o/r", [new RepoFile("m.gguf", bytes)]);
            using var c = Client(hub);
            var r = await Run(c);
            var remote = r.Report!;

            string Json(ScoutReport x) => System.Text.Json.JsonSerializer.Serialize(x with { Network = null, Artifact = x.Artifact with { Source = null, Sha256State = "-" } }, ScoutJsonContext.Default.ScoutReport);
            Assert.Equal(Json(local), Json(remote));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task The_report_names_the_exact_commit_the_hub_hash_and_what_was_spent()
    {
        byte[] bytes = Gguf("llama", 2);
        var hub = new FakeHub("o/r", [new RepoFile("sub/m-Q4_K_M.gguf", bytes)], gated: null);
        using var c = Client(hub);
        var r = await Run(c);
        var a = r.Report!.Artifact;
        Assert.Equal("huggingface", a.Source!.Kind);
        Assert.Equal(Sha, a.Source.Revision);
        Assert.Equal(["sub/m-Q4_K_M.gguf"], a.Source.Paths);
        Assert.Equal("m-Q4_K_M.gguf", a.FileName);                 // file name only: no directories
        Assert.Equal(Hash, a.Source.Sha256);
        Assert.Equal("published_by_huggingface", a.Source.Sha256Source);
        Assert.Equal("published_by_huggingface", a.Sha256State);   // never presented as a locally verified hash
        Assert.Equal("mit", a.Source.License);
        Assert.Equal(1234, a.Source.Downloads30Days);
        Assert.Equal(bytes.Length, a.FileBytes);
        Assert.Equal(["cdn-lfs.huggingface.co", "huggingface.co"], r.Report.Network!.Hosts);
        Assert.True(r.Report.Network.BytesReceived < bytes.Length + 4096);
        Assert.True(a.Source.IndexBytesRead <= bytes.Length);
    }

    [Fact]
    public async Task A_huge_model_costs_one_chunk_not_the_model()
    {
        byte[] bytes = Gguf("llama", 2, dataBytes: 6 << 20);
        var hub = new FakeHub("o/r", [new RepoFile("m.gguf", bytes)]);
        using var c = Client(hub);
        var r = await Run(c);
        Assert.Equal(RemoteGgufReader.InitialChunkBytes, r.Report!.Artifact.Source!.IndexBytesRead);
        Assert.True(hub.BodyBytes <= RemoteGgufReader.InitialChunkBytes);
    }

    // ── choosing the file ──
    [Fact]
    public async Task Projectors_are_not_candidates_so_a_model_plus_its_mmproj_needs_no_flag()
    {
        var hub = new FakeHub("o/r", [new RepoFile("m.gguf", Gguf("llama", 1)), new RepoFile("mmproj-m.gguf", Gguf("clip", 1))]);
        using var c = Client(hub);
        var r = await Run(c);
        Assert.Equal("m.gguf", r.Report!.Artifact.FileName);
    }

    [Fact]
    public async Task Several_models_without_a_choice_returns_the_choices_and_no_report()
    {
        var hub = new FakeHub("o/r", [new RepoFile("a-Q4.gguf", Gguf("llama", 1)), new RepoFile("a-Q8.gguf", Gguf("llama", 1))]);
        using var c = Client(hub);
        var r = await Run(c);
        Assert.Null(r.Report);
        Assert.Equal(["a-Q4.gguf", "a-Q8.gguf"], r.Choices.Select(m => m.Name));
        Assert.Contains("choose one with -f", r.Message);
        Assert.DoesNotContain(hub.Requests, q => q.Contains("/resolve/")); // nothing was fetched before the user chose
    }

    [Theory]
    [InlineData("a-Q8.gguf")]
    [InlineData("A-q8.GGUF")]
    [InlineData("a-Q8")]
    public async Task A_file_is_chosen_by_name_ignoring_case_and_extension(string wanted)
    {
        var hub = new FakeHub("o/r", [new RepoFile("a-Q4.gguf", Gguf("llama", 1)), new RepoFile("a-Q8.gguf", Gguf("llama", 2))]);
        using var c = Client(hub);
        var r = await Run(c, wanted);
        Assert.Equal("a-Q8.gguf", r.Report!.Artifact.FileName);
    }

    [Fact]
    public async Task An_unknown_file_lists_what_exists()
    {
        var hub = new FakeHub("o/r", [new RepoFile("a.gguf", Gguf("llama", 1))]);
        using var c = Client(hub);
        var r = await Run(c, "zzz.gguf");
        Assert.Null(r.Report);
        Assert.Contains("zzz.gguf", r.Message);
        Assert.Single(r.Choices);
    }

    [Fact]
    public async Task A_repo_with_only_a_projector_has_nothing_to_inspect()
    {
        var hub = new FakeHub("o/r", [new RepoFile("mmproj-x.gguf", Gguf("clip", 1))]);
        using var c = Client(hub);
        var r = await Run(c);
        Assert.Null(r.Report);
        Assert.Contains("no GGUF model files", r.Message);
    }

    [Fact]
    public async Task A_split_model_is_read_shard_by_shard_and_gets_no_single_hash()
    {
        byte[] s0 = Gguf("llama", 1), s1 = Gguf("llama", 1);
        var hub = new FakeHub("o/r", [new RepoFile("big-00001-of-00002.gguf", s0, "1".PadRight(64, '1')), new RepoFile("big-00002-of-00002.gguf", s1, "2".PadRight(64, '2'))]);
        using var c = Client(hub);
        var r = await Run(c, "big");
        var a = r.Report!.Artifact;
        Assert.Equal(2, a.ShardCount);
        Assert.Equal(2, a.Source!.Paths.Count);
        Assert.Null(a.Source.Sha256);                       // two hashes cannot be summarised as one
        Assert.Equal("not_computed", a.Sha256State);
        Assert.Equal(s0.Length + s1.Length, a.FileBytes);
    }

    [Fact]
    public async Task A_split_model_with_a_missing_shard_is_refused_before_any_read()
    {
        var hub = new FakeHub("o/r", [new RepoFile("big-00001-of-00003.gguf", Gguf("llama", 1)), new RepoFile("big-00003-of-00003.gguf", Gguf("llama", 1))]);
        using var c = Client(hub);
        var r = await Run(c, "big");
        Assert.Null(r.Report);
        Assert.Contains("missing shards", r.Message);
        Assert.DoesNotContain(hub.Requests, q => q.Contains("/resolve/"));
    }

    // ── honesty about what the Hub says versus what the file says ──
    [Fact]
    public async Task A_hub_architecture_that_disagrees_with_the_file_is_reported_and_the_file_wins()
    {
        var hub = new FakeHub("o/r", [new RepoFile("m.gguf", Gguf("llama", 1))], hubArch: "qwen2");
        using var c = Client(hub);
        var r = await Run(c);
        Assert.Equal("llama", r.Report!.Architecture!.Declared);
        var b = Assert.Single(r.Report.Blockers, x => x.Id == "scout.source.hub_architecture_disagrees");
        Assert.Equal(BlockerKind.Suspected, b.Kind);
    }

    [Fact]
    public async Task A_listing_size_that_disagrees_with_the_server_is_flagged()
    {
        var hub = new FakeHub("o/r", [new RepoFile("m.gguf", Gguf("llama", 1), ListedSize: 5)]);
        using var c = Client(hub);
        var r = await Run(c);
        Assert.Contains(r.Report!.Blockers, b => b.Id == "scout.source.size_mismatch");
    }

    [Fact]
    public async Task A_gated_repo_is_reported_and_nothing_is_accepted_on_the_users_behalf()
    {
        var hub = new FakeHub("o/r", [new RepoFile("m.gguf", Gguf("llama", 1))], gated: "manual");
        using var c = Client(hub);
        var r = await Run(c);
        Assert.Equal("manual", r.Report!.Artifact.Source!.Gated);
        Assert.Contains(r.Report.Findings, f => f.Id == "source.gated");
    }

    // ── failure modes ──
    [Fact]
    public async Task An_index_past_the_cap_is_a_failed_inspection_with_a_clear_blocker_not_a_download()
    {
        byte[] big = Gguf("llama", 1, extraTokenBytes: 3 << 20, dataBytes: 4 << 20);
        var hub = new FakeHub("o/r", [new RepoFile("m.gguf", big)]);
        using var c = Client(hub);
        var r = await Run(c, max: 2L << 20);
        var rep = r.Report!;
        Assert.Equal(StageState.Failed, rep.Stages[0].State);
        Assert.Contains(rep.Blockers, b => b.Id == "scout.remote.index_over_cap");
        Assert.Null(rep.Tensors);
        Assert.True(hub.BodyBytes <= 2L << 20);
        // Network bytes are the index reads plus the small model-API response; the reads themselves stayed inside the cap (asserted above).
        Assert.InRange(rep.Network!.BytesReceived - hub.BodyBytes, 0, 4096);
    }

    [Theory]
    [InlineData(HttpStatusCode.NotFound, "was not found")]
    [InlineData(HttpStatusCode.Unauthorized, "refused access")]
    [InlineData(HttpStatusCode.Forbidden, "refused access")]
    public async Task Missing_and_restricted_repos_give_a_message_not_a_stack_trace(HttpStatusCode status, string expected)
    {
        var hub = new FakeHub("o/r", [], apiStatus: status);
        using var c = Client(hub);
        var r = await Run(c);
        Assert.Null(r.Report);
        Assert.Contains(expected, r.Message);
    }

    [Fact]
    public async Task A_switched_off_policy_throws_and_touches_nothing()
    {
        var hub = new FakeHub("o/r", [new RepoFile("m.gguf", Gguf("llama", 1))]);
        using var c = Client(hub, env: n => n == "STINGRAY_ALLOW_EXTERNAL" ? "off" : null);
        await Assert.ThrowsAsync<ExternalAccessDeniedException>(() => Run(c));
        Assert.Empty(hub.Requests);
    }

    // ── contributing a signature from a hosted file ──
    [Fact]
    public async Task A_hosted_admitted_file_yields_a_signature_whose_origin_is_the_hub_and_is_labelled_as_published()
    {
        var hub = new FakeHub("o/r", [new RepoFile("m.gguf", Gguf("llama", 2))]);
        using var c = Client(hub);
        var r = await Run(c);
        var sig = SignatureBuilder.TryBuild(r.Input!, r.Report!, r.Origin!, out string why);
        Assert.True(sig is not null, why);
        var o = Assert.Single(sig!.Origins);
        Assert.Equal(("o/r", Sha, Hash, "published_by_huggingface"), (o.SourceRepo, o.SourceRevision, o.Sha256, o.Sha256Source));
        Assert.Equal(SignatureBuilder.StructureId(sig.Structure), sig.StructureId);
    }

    [Fact]
    public async Task Bad_repo_text_never_reaches_the_network()
    {
        var hub = new FakeHub("o/r", []);
        using var c = Client(hub);
        await Assert.ThrowsAsync<ArgumentException>(() =>
            RemoteScout.RunAsync(c, new RemoteScoutRequest("../../etc", null, null, RemoteGgufReader.DefaultMaxIndexBytes), Opts, default));
        Assert.Empty(hub.Requests);
    }
}
