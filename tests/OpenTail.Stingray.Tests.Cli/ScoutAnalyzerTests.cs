using OpenTail.Stingray.Engine.Scout;
using System.Text.Json;
using OpenTail.Stingray.Cli.Scout;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>Fast, synthetic-index tests for <c>stingray scout</c>; no checkpoint is needed or opened.</summary>
public sealed class ScoutAnalyzerTests
{
    private static GgufTensorInfo T(string name, DType dt, params long[] dims) => new(name, dims.Length, dims, dt, 0);

    private static IEnumerable<GgufTensorInfo> Layer(int i, DType w = DType.Q4_K) =>
    [
        T($"blk.{i}.attn_norm.weight", DType.Float32, 64),
        T($"blk.{i}.attn_q.weight", w, 64, 64),
        T($"blk.{i}.attn_k.weight", w, 64, 64),
        T($"blk.{i}.ffn_down.weight", w, 256, 64),
    ];

    private static ScoutInput Input(string arch, IEnumerable<GgufTensorInfo> tensors, Action<Dictionary<string, object>>? tweak = null)
    {
        var md = new Dictionary<string, object>
        {
            ["general.architecture"] = arch,
            ["tokenizer.ggml.model"] = "gpt2",
            ["tokenizer.ggml.merges"] = new string[] { "a b" },
            ["tokenizer.ggml.tokens"] = new string[] { "a", "b", "ab" },
            ["tokenizer.chat_template"] = "{{ x }}",
        };
        tweak?.Invoke(md);
        var list = tensors.ToArray();
        return new ScoutInput("tiny.gguf", 12345, 3, (ulong)list.Length, (ulong)md.Count, md, list);
    }

    private static readonly ScoutOptions Opts = new("test-build");

    private static ScoutReport Run(ScoutInput i, ScoutOptions? o = null) => ScoutAnalyzer.Analyze(i, o ?? Opts);

    [Fact]
    public void Admitted_dense_arch_has_no_confirmed_blockers_and_correct_tensor_facts()
    {
        var r = Run(Input("llama", Enumerable.Range(0, 3).SelectMany(i => Layer(i))));
        Assert.DoesNotContain(r.Blockers, b => b.Kind == BlockerKind.Confirmed);
        Assert.True(r.Architecture!.Resolved);
        Assert.Equal("Admitted", r.Architecture.Status);
        Assert.Equal(3, r.Tensors!.LayerCount);
        Assert.Equal(12ul, r.Tensors.Count);
        Assert.Contains(r.Tensors.Patterns, p => p.Pattern == "blk.*.attn_q.weight" && p.Count == 3 && p.Layers == "0-2");
        Assert.Empty(r.Tensors.Irregularities);
        Assert.Equal(StageState.Passed, r.Stages.Single(s => s.Stage == "1_static_contract").State);
    }

    [Fact]
    public void Bytes_by_dtype_sum_to_total()
    {
        var r = Run(Input("llama", Enumerable.Range(0, 2).SelectMany(i => Layer(i))));
        Assert.Equal(r.Tensors!.TotalBytes, r.Tensors.ByDType.Sum(d => d.Bytes));
        Assert.Contains(r.Tensors.ByDType, d => d.DType == "Float32" && d.Count == 2);
    }

    [Fact]
    public void Unregistered_architecture_is_a_confirmed_blocker_and_suggests_admit_arch()
    {
        var r = Run(Input("totally_new_arch", Layer(0)));
        Assert.False(r.Architecture!.Resolved);
        Assert.Contains(r.Blockers, b => b.Id == "scout.arch.not_registered" && b.Kind == BlockerKind.Confirmed);
        Assert.Contains(r.NextActions, a => a.Command.StartsWith("admit-arch", StringComparison.Ordinal));
        Assert.Equal(StageState.Failed, r.Stages.Single(s => s.Stage == "1_static_contract").State);
    }

    [Fact]
    public void Layer_missing_a_tensor_is_reported_as_partial_layers()
    {
        var tensors = Enumerable.Range(0, 4).SelectMany(i => Layer(i)).Where(t => t.Name != "blk.2.attn_k.weight");
        var r = Run(Input("llama", tensors));
        var irr = Assert.Single(r.Tensors!.Irregularities, x => x.Kind == "partial_layers");
        Assert.Equal("blk.*.attn_k.weight", irr.Pattern);
        Assert.Contains("0-1,3", irr.Detail);
    }

    [Fact]
    public void Same_name_with_different_shape_or_dtype_across_layers_is_flagged_not_matched()
    {
        var tensors = Layer(0).Concat(Layer(1, DType.Q6_K)).Concat([T("blk.2.attn_q.weight", DType.Q4_K, 32, 64)]);
        var r = Run(Input("llama", tensors));
        Assert.Contains(r.Tensors!.Irregularities, x => x.Pattern == "blk.*.attn_q.weight" && x.Kind == "dtype_varies");
        Assert.Contains(r.Tensors.Irregularities, x => x.Pattern == "blk.*.attn_q.weight" && x.Kind == "shape_varies");
    }

    [Fact]
    public void Dtype_the_engine_cannot_execute_is_a_confirmed_blocker_with_tensor_evidence()
    {
        var r = Run(Input("llama", Layer(0).Concat([T("blk.0.weird.weight", DType.Int32, 8, 8)])));
        var b = Assert.Single(r.Blockers, x => x.Id == "scout.dtype.unsupported");
        Assert.Equal(BlockerKind.Confirmed, b.Kind);
        Assert.Contains(b.Evidence, e => e.Name == "blk.0.weird.weight");
    }

    [Theory]
    [InlineData("llama", false, false, "scout.tokenizer.llama_no_merges_no_scores")]
    [InlineData("gpt2", false, false, "scout.tokenizer.bpe_no_merges")]
    public void Tokenizer_warning_shapes_are_suspected_not_confirmed(string model, bool merges, bool scores, string id)
    {
        var r = Run(Input("llama", Layer(0), md =>
        {
            md["tokenizer.ggml.model"] = model;
            if (!merges) md.Remove("tokenizer.ggml.merges");
            if (scores) md["tokenizer.ggml.scores"] = new float[] { 1f };
        }));
        Assert.Contains(r.Blockers, b => b.Id == id && b.Kind == BlockerKind.Suspected);
    }

    [Fact]
    public void Scores_only_spm_is_an_observed_finding_not_a_blocker()
    {
        var r = Run(Input("llama", Layer(0), md =>
        {
            md["tokenizer.ggml.model"] = "llama";
            md.Remove("tokenizer.ggml.merges");
            md["tokenizer.ggml.scores"] = new float[] { 1f };
        }));
        Assert.Contains(r.Findings, f => f.Id == "tokenizer.spm_scores_only" && f.Certainty == Certainty.Known && f.Evidence.Count > 0);
        Assert.DoesNotContain(r.Blockers, b => b.Id.StartsWith("scout.tokenizer", StringComparison.Ordinal));
    }

    [Fact]
    public void Unknown_working_set_is_never_allowed_even_under_a_huge_budget()
    {
        var none = Run(Input("llama", Layer(0)));
        Assert.Equal("not_assessed", none.Resources.ExecutionDecision);

        var big = Run(Input("llama", Layer(0)), new ScoutOptions("b", BudgetBytes: 1L << 40));
        Assert.Equal("blocked", big.Resources.ExecutionDecision);
        Assert.Equal(Certainty.Unknown, big.Resources.HostWorkingSet.Certainty);
        Assert.Null(big.Resources.HostWorkingSet.Bytes);
        Assert.Equal(StageState.Blocked, big.Stages.Single(s => s.Stage == "2_feasibility").State);
    }

    [Fact]
    public void Execution_stages_are_NotRun_never_Passed_in_static_mode()
    {
        var r = Run(Input("llama", Layer(0)));
        foreach (var s in r.Stages.Where(s => s.Stage[0] >= '3'))
            Assert.Equal(StageState.NotRun, s.State);
    }

    [Fact]
    public void Report_is_deterministic_and_carries_no_machine_paths()
    {
        var i = Input("llama", Enumerable.Range(0, 3).SelectMany(x => Layer(x)));
        string a = JsonSerializer.Serialize(Run(i), ScoutJsonContext.Default.ScoutReport);
        string b = JsonSerializer.Serialize(Run(Input("llama", Enumerable.Range(0, 3).SelectMany(x => Layer(x)).Reverse())), ScoutJsonContext.Default.ScoutReport);
        Assert.Equal(a, b); // input tensor order must not change the report
        Assert.DoesNotContain(":\\", a);
        Assert.DoesNotContain("Users", a);
        using var doc = JsonDocument.Parse(a);
        Assert.Equal(ScoutReport.CurrentSchemaVersion, doc.RootElement.GetProperty("schema_version").GetInt32());
        Assert.Equal("not_computed", doc.RootElement.GetProperty("artifact").GetProperty("sha256_state").GetString());
    }

    [Fact]
    public void Unknown_values_serialize_as_null_not_zero()
    {
        string json = JsonSerializer.Serialize(Run(Input("llama", Layer(0))), ScoutJsonContext.Default.ScoutReport);
        using var doc = JsonDocument.Parse(json);
        var ws = doc.RootElement.GetProperty("resources").GetProperty("host_working_set");
        Assert.Equal(JsonValueKind.Null, ws.GetProperty("bytes").ValueKind);
        Assert.Equal("Unknown", ws.GetProperty("certainty").GetString());
    }

    [Fact]
    public void Unreadable_file_yields_a_failed_artifact_stage_and_nothing_invented()
    {
        var r = ScoutAnalyzer.Unreadable("bad.gguf", 10, "InvalidDataException: bad magic", Opts);
        Assert.Equal(StageState.Failed, r.Stages[0].State);
        Assert.Null(r.Tensors);
        Assert.Null(r.Architecture);
        Assert.Contains(r.Blockers, b => b.Id == "scout.artifact.unreadable");
    }

    [Fact]
    public void Index_only_tensor_source_refuses_to_read_values()
    {
        var src = new IndexOnlyTensorSource(new Dictionary<string, object>(), [T("a", DType.Float32, 1)]);
        Assert.NotNull(src.FindTensor("a"));
        Assert.Throws<InvalidOperationException>(() => src.GetTensorData(src.Tensors[0]));
    }

    [Theory]
    [InlineData("blk.12.attn_q.weight", "blk.*.attn_q.weight", "blk", 12)]
    [InlineData("v.blk.3.ln1.weight", "v.blk.*.ln1.weight", "v.blk", 3)]
    [InlineData("token_embd.weight", "token_embd.weight", null, -1)]
    [InlineData("mm.0.weight", "mm.0.weight", null, -1)]
    public void Normalize_collapses_only_layer_indices(string name, string pattern, string? stack, int layer)
    {
        var n = ScoutAnalyzer.Normalize(name);
        Assert.Equal((pattern, stack, layer), (n.Pattern, n.Stack, n.Layer));
    }

    [Theory]
    [InlineData(new[] { 0, 1, 2, 3 }, "0-3")]
    [InlineData(new[] { 0, 1, 3 }, "0-1,3")]
    [InlineData(new[] { 5 }, "5")]
    public void Ranges_are_compact(int[] layers, string expected) => Assert.Equal(expected, ScoutAnalyzer.Ranges(layers));

    [Theory]
    [InlineData("64G", 64L << 30)]
    [InlineData("512M", 512L << 20)]
    [InlineData("1.5GiB", 3L << 29)]
    [InlineData("2048", 2048)]
    public void Size_parses(string text, long expected)
    {
        Assert.True(ScoutSize.TryParse(text, out long v));
        Assert.Equal(expected, v);
    }

    [Theory]
    [InlineData("")]
    [InlineData("abc")]
    [InlineData("-1G")]
    [InlineData("12X")]
    public void Size_rejects_garbage(string text) => Assert.False(ScoutSize.TryParse(text, out _));
}

