using OpenTail.Stingray.Engine.Scout;
using OpenTail.Stingray.Cli.Scout;
using OpenTail.Stingray.Core.Net;
using static OpenTail.Stingray.Tests.Cli.HubFixtures;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>The quant picker against a fake Hub. Sizes differ through layer count, so the memory estimate differs per file, and budgets are set from the estimates themselves.</summary>
public sealed class QuantPickerTests
{
    private static ExternalHttpClient Client(FakeHub hub, Func<string, string?>? env = null) => new(inner: hub, env: env ?? (_ => null));

    private static ScoutOptions Options(long budget, long reserve = 0, int ctx = 256) => new("test-build", budget, reserve, null, ctx);

    private static Task<(QuantReport?, string?)> Run(ExternalHttpClient c, long budget, long reserve = 0, int maxFiles = QuantPicker.DefaultMaxFiles, long maxIndex = RemoteGgufReader.DefaultMaxIndexBytes, List<string>? progress = null) =>
        QuantPicker.RunAsync(c, "o/r", null, maxIndex, maxFiles, Options(budget, reserve), "given", progress is null ? null : progress.Add, default);

    /// <summary>Three quants of one model: more layers means more weights, so a bigger estimate.</summary>
    private static FakeHub Three(Action<List<RepoFile>>? tweak = null)
    {
        var files = new List<RepoFile>
        {
            new("m-Q8_0.gguf", Gguf("llama", 24, dataBytes: 1 << 20), "8".PadRight(64, '8')),
            new("m-Q5_K_M.gguf", Gguf("llama", 12, dataBytes: 1 << 20), "5".PadRight(64, '5')),
            new("m-Q4_K_M.gguf", Gguf("llama", 4, dataBytes: 1 << 20), "4".PadRight(64, '4')),
        };
        tweak?.Invoke(files);
        return new FakeHub("o/r", files);
    }

    private static async Task<QuantReport> Estimates()
    {
        using var c = Client(Three());
        var (r, msg) = await Run(c, budget: 1L << 40);
        Assert.True(r is not null, msg);
        return r!;
    }

    [Fact]
    public async Task Rows_are_largest_first_with_the_quant_label_the_estimate_and_a_pinned_pull_command()
    {
        var r = await Estimates();
        Assert.Equal(["m-Q8_0.gguf", "m-Q5_K_M.gguf", "m-Q4_K_M.gguf"], r.Rows.Select(x => x.File));
        Assert.Equal(["Q8_0", "Q5_K_M", "Q4_K_M"], r.Rows.Select(x => x.Quant));
        Assert.True(r.Rows[0].EstimatedPeakBytes > r.Rows[1].EstimatedPeakBytes && r.Rows[1].EstimatedPeakBytes > r.Rows[2].EstimatedPeakBytes);
        Assert.All(r.Rows, row => Assert.Equal(QuantVerdict.Fits, row.Verdict));
        Assert.Equal("m-Q8_0.gguf", r.Recommended);
        Assert.Equal($"stingray pull -r o/r --revision {Sha} -q \"m-Q8_0.gguf\"", r.Rows[0].PullCommand);
        Assert.Equal(Sha, r.Revision);
        Assert.Equal("5".PadRight(64, '5'), r.Rows[1].Sha256);
    }

    [Fact]
    public async Task The_largest_quant_that_fits_is_recommended_and_a_one_byte_difference_flips_the_verdict()
    {
        var est = await Estimates();
        long q8 = est.Rows[0].EstimatedPeakBytes!.Value, q5 = est.Rows[1].EstimatedPeakBytes!.Value;

        using (var c = Client(Three()))
        {
            var (r, _) = await Run(c, budget: q5);   // exactly fits Q5, not Q8
            Assert.Equal(["m-Q8_0.gguf", "m-Q5_K_M.gguf", "m-Q4_K_M.gguf"], r!.Rows.Select(x => x.File));
            Assert.Equal(QuantVerdict.TooBig, r.Rows[0].Verdict);
            Assert.Equal(QuantVerdict.Fits, r.Rows[1].Verdict);
            Assert.Equal("m-Q5_K_M.gguf", r.Recommended);
        }
        using (var c = Client(Three()))
        {
            var (r, _) = await Run(c, budget: q5 - 1);  // one byte short of Q5
            Assert.Equal("m-Q4_K_M.gguf", r!.Recommended);
            Assert.Equal(QuantVerdict.TooBig, r.Rows[1].Verdict);
        }
        using (var c = Client(Three()))
        {
            var (r, _) = await Run(c, budget: q8);       // everything fits
            Assert.Equal("m-Q8_0.gguf", r!.Recommended);
        }
    }

    [Fact]
    public async Task The_reserve_is_part_of_the_decision_and_a_too_big_row_says_why()
    {
        var est = await Estimates();
        long q4 = est.Rows[2].EstimatedPeakBytes!.Value;
        using var c = Client(Three());
        var (r, _) = await Run(c, budget: q4 + 99, reserve: 100);   // the reserve pushes even the smallest over
        Assert.Null(r!.Recommended);
        Assert.All(r.Rows, row => Assert.Equal(QuantVerdict.TooBig, row.Verdict));
        Assert.Contains("exceeds", r.Rows[2].Why);
    }

    [Fact]
    public async Task A_storage_type_the_loader_rejects_is_unsupported_not_unreadable()
    {
        var hub = Three(files => files.Add(new RepoFile("m-Q4_0_4_4.gguf", Gguf("llama", 4, dataBytes: 1 << 20, extraTensorType: 31), "9".PadRight(64, '9'))));
        using var c = Client(hub);
        var (r, _) = await Run(c, budget: 1L << 40);
        var bad = r!.Rows.Single(x => x.File == "m-Q4_0_4_4.gguf");
        Assert.Equal(QuantVerdict.Unsupported, bad.Verdict);
        Assert.Contains("unsupported GGML type ID 31", bad.Why);
        Assert.Equal("Q4_0_4_4", bad.Quant);
        Assert.Equal("m-Q8_0.gguf", r.Recommended);              // the others are unaffected
    }

    [Fact]
    public async Task A_known_but_unrunnable_weight_type_is_unsupported_even_though_the_file_parses()
    {
        // GGML type 24 (int8) parses, but the CPU path has no kernel for it as a weight: the dtype gate refuses it.
        var hub = new FakeHub("o/r", [new RepoFile("m-Q8_0.gguf", Gguf("llama", 4, dataBytes: 1 << 20, extraTensorType: 24))]);
        using var c = Client(hub);
        var (r, _) = await Run(c, budget: 1L << 40);
        var row = r!.Rows.Single();
        Assert.Equal(QuantVerdict.Unsupported, row.Verdict);
        Assert.Null(r.Recommended);
        Assert.Contains("cannot execute", row.Why);
    }

    [Fact]
    public async Task An_architecture_this_engine_does_not_run_makes_every_row_unsupported_with_one_note()
    {
        var hub = new FakeHub("o/r", [new RepoFile("a.gguf", Gguf("mycoolllm", 2)), new RepoFile("b.gguf", Gguf("mycoolllm", 3))], hubArch: "mycoolllm");
        using var c = Client(hub);
        var (r, _) = await Run(c, budget: 1L << 40);
        Assert.All(r!.Rows, row => Assert.Equal(QuantVerdict.Unsupported, row.Verdict));
        Assert.Contains(r.Notes, n => n.StartsWith("No quantisation of this model can run here", StringComparison.Ordinal));
        Assert.Null(r.Recommended);
    }

    [Fact]
    public async Task An_unmodelled_family_is_Unknown_never_too_big_and_never_fits()
    {
        // deepseek2 is admitted but its MLA state is not modelled by the estimator: nothing may be promised either way.
        var hub = new FakeHub("o/r", [new RepoFile("a.gguf", Gguf("deepseek2", 2))], hubArch: "deepseek2");
        using var c = Client(hub);
        var (r, _) = await Run(c, budget: 1L << 40);
        var row = r!.Rows.Single();
        Assert.Equal(QuantVerdict.Unknown, row.Verdict);
        Assert.Null(row.EstimatedPeakBytes);
        Assert.Null(r.Recommended);
    }

    [Fact]
    public async Task One_unreadable_index_does_not_spoil_the_other_rows()
    {
        var hub = Three(files => files.Add(new RepoFile("m-Q2_K.gguf", Gguf("llama", 2, extraTokenBytes: 3 << 20, dataBytes: 1 << 20))));
        using var c = Client(hub);
        var (r, _) = await Run(c, budget: 1L << 40, maxIndex: 2L << 20);
        var cut = r!.Rows.Single(x => x.File == "m-Q2_K.gguf");
        Assert.Equal(QuantVerdict.NotInspected, cut.Verdict);
        Assert.Contains("not downloaded", cut.Why);
        Assert.Equal("m-Q8_0.gguf", r.Recommended);
    }

    [Fact]
    public async Task Projectors_are_ignored_split_models_are_one_row_and_a_split_with_a_missing_shard_is_skipped_with_a_note()
    {
        var hub = Three(files =>
        {
            files.Add(new RepoFile("mmproj-m.gguf", Gguf("clip", 1)));
            files.Add(new RepoFile("big-00001-of-00002.gguf", Gguf("llama", 2), "1".PadRight(64, '1')));
            files.Add(new RepoFile("big-00002-of-00002.gguf", Gguf("llama", 2), "2".PadRight(64, '2')));
            files.Add(new RepoFile("gone-00001-of-00003.gguf", Gguf("llama", 2)));
            files.Add(new RepoFile("gone-00003-of-00003.gguf", Gguf("llama", 2)));
        });
        using var c = Client(hub);
        var (r, _) = await Run(c, budget: 1L << 40);
        Assert.DoesNotContain(r!.Rows, x => x.File.Contains("mmproj"));
        var split = r.Rows.Single(x => x.Shards == 2);
        Assert.Contains("2 shards", split.File);
        Assert.Null(split.Sha256);                                   // two hashes are not one
        Assert.Contains("-q \"big\"", split.PullCommand);            // the stem selects every shard
        Assert.Contains(r.Notes, n => n.Contains("gone") && n.Contains("missing"));
        Assert.Equal(4, r.Rows.Count);
    }

    [Fact]
    public async Task The_cap_limits_how_many_files_are_inspected_and_says_so()
    {
        var hub = Three();
        using var c = Client(hub);
        var progress = new List<string>();
        var (r, _) = await Run(c, budget: 1L << 40, maxFiles: 2, progress: progress);
        Assert.Equal(2, r!.Rows.Count);
        Assert.Equal(2, progress.Count);
        Assert.Contains(r.Notes, n => n.Contains("only the first 2"));
        Assert.Equal(2, hub.Requests.Count(q => q.Contains("/resolve/") && q.StartsWith("huggingface.co", StringComparison.Ordinal)));
    }

    [Fact]
    public async Task The_hub_is_asked_about_the_repo_once_not_once_per_file()
    {
        var hub = Three();
        using var c = Client(hub);
        await Run(c, budget: 1L << 40);
        Assert.Single(hub.Requests, q => q.Contains("/api/models/"));
        Assert.Equal(3, hub.Requests.Count(q => q.Contains("/resolve/") && q.StartsWith("huggingface.co", StringComparison.Ordinal)));
    }

    [Fact]
    public async Task No_weights_are_downloaded_whatever_the_file_sizes()
    {
        var hub = new FakeHub("o/r", [new RepoFile("a-Q8_0.gguf", Gguf("llama", 2, dataBytes: 8 << 20)), new RepoFile("a-Q4_K_M.gguf", Gguf("llama", 2, dataBytes: 6 << 20))]);
        using var c = Client(hub);
        var (r, _) = await Run(c, budget: 1L << 40);
        Assert.True(hub.BodyBytes <= 2 * RemoteGgufReader.InitialChunkBytes, $"served {hub.BodyBytes}");
        Assert.True(r!.Network.BytesReceived < 14L << 20);
    }

    [Fact]
    public async Task Repo_problems_come_back_as_a_message()
    {
        using var c = Client(new FakeHub("o/r", [], apiStatus: System.Net.HttpStatusCode.NotFound));
        var (r, msg) = await Run(c, budget: 1L << 40);
        Assert.Null(r);
        Assert.Contains("was not found", msg);

        using var c2 = Client(new FakeHub("o/r", [new RepoFile("mmproj-x.gguf", Gguf("clip", 1))]));
        var (r2, msg2) = await Run(c2, budget: 1L << 40);
        Assert.Null(r2);
        Assert.Contains("no complete GGUF model files", msg2);
    }

    [Fact]
    public async Task A_switched_off_policy_throws_and_touches_nothing()
    {
        var hub = Three();
        using var c = Client(hub, env: n => n == "STINGRAY_ALLOW_EXTERNAL" ? "off" : null);
        await Assert.ThrowsAsync<ExternalAccessDeniedException>(() => Run(c, budget: 1L << 40));
        Assert.Empty(hub.Requests);
    }

    [Fact]
    public async Task The_report_serialises_with_string_verdicts_and_nulls_for_unknowns()
    {
        var hub = new FakeHub("o/r", [new RepoFile("a.gguf", Gguf("deepseek2", 2))], hubArch: "deepseek2");
        using var c = Client(hub);
        var (r, _) = await Run(c, budget: 1L << 40);
        string json = System.Text.Json.JsonSerializer.Serialize(r, QuantJsonContext.Default.QuantReport);
        using var doc = System.Text.Json.JsonDocument.Parse(json);
        var row = doc.RootElement.GetProperty("rows")[0];
        Assert.Equal("Unknown", row.GetProperty("verdict").GetString());
        Assert.Equal(System.Text.Json.JsonValueKind.Null, row.GetProperty("estimated_peak_bytes").ValueKind);
        Assert.Equal(System.Text.Json.JsonValueKind.Null, doc.RootElement.GetProperty("recommended").ValueKind);
        Assert.DoesNotContain(":\\", json);
        Assert.Equal(Sha, doc.RootElement.GetProperty("revision").GetString());
    }

    [Theory]
    [InlineData("SmolLM2-135M-Instruct-Q4_K_M.gguf", "Q4_K_M")]
    [InlineData("model-Q4_0_4_4.gguf", "Q4_0_4_4")]
    [InlineData("model.IQ3_XS.gguf", "IQ3_XS")]
    [InlineData("model-BF16.gguf", "BF16")]
    [InlineData("Model-f16.gguf", "F16")]
    [InlineData("qwen3-MXFP4.gguf", "MXFP4")]
    [InlineData("no-label-here.gguf", "?")]
    public async Task Quant_labels_are_read_from_the_file_name(string file, string expected)
    {
        var hub = new FakeHub("o/r", [new RepoFile(file, Gguf("llama", 2))]);
        using var c = Client(hub);
        var (r, _) = await Run(c, budget: 1L << 40);
        Assert.Equal(expected, r!.Rows.Single().Quant);
    }
}
