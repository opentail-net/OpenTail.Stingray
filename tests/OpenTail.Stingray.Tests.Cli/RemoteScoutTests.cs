using System.Net;
using System.Net.Http.Headers;
using System.Text;
using OpenTail.Stingray.Cli.Scout;
using OpenTail.Stingray.Core.Net;
using static OpenTail.Stingray.Tests.Cli.HubFixtures;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// Remote scout against a fake Hub (API + redirect + Range). No network. The headline test builds one GGUF, scouts it both from disk and
/// through the fake Hub, and requires identical analysis.
/// </summary>
public sealed class RemoteScoutTests
{

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
