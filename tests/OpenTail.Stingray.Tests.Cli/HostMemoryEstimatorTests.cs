using OpenTail.Stingray.Cli.Scout;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>The host working-set estimate and the budget gate built on it. Pure arithmetic over synthetic indexes; no model files.</summary>
public sealed class HostMemoryEstimatorTests
{
    private static ModelHyperparams Hp(Func<ModelHyperparams, ModelHyperparams>? tweak = null)
    {
        var hp = new ModelHyperparams
        {
            NumLayers = 2, EmbeddingDim = 64, NumHeads = 8, NumKvHeads = 2, HeadDim = 8,
            IntermediateDim = 128, ContextLength = 1000, VocabSize = 100,
        };
        return tweak is null ? hp : tweak(hp);
    }

    // 4096 elements is a whole number of Q4_K (256) blocks: 144 bytes per block => 2304 bytes.
    private static GgufTensorInfo Q4K(string name) => new(name, 1, [4096], DType.Q4_K, 0);
    private static GgufTensorInfo F32(string name, long n) => new(name, 1, [n], DType.Float32, 0);
    private const long Q4KBytes = 4096 / 256 * 144;

    private static long Part(HostWorkingSet e, string name) => e.Components.Single(c => c.Name == name).Bytes!.Value;

    // ── KV ──
    [Fact]
    public void Kv_bytes_are_fp32_K_plus_V_over_layers()
    {
        // 2 layers x (K+V = 2) x 2 kv heads x 8 dims x 100 tokens x 4 bytes
        Assert.Equal(2L * 2 * 2 * 8 * 100 * 4, HostMemoryEstimator.KvBytes(Hp(), 100));
    }

    [Fact]
    public void Sliding_window_layers_are_capped_and_aliased_layers_cost_nothing()
    {
        var swa = HostMemoryEstimator.KvBytes(Hp(h => h with { IsSwaLayer = new[] { true, false }, SlidingWindowSize = 10 }), 100);
        Assert.Equal(2L * 2 * 8 * 4 * (10 + 100), swa);

        var aliased = HostMemoryEstimator.KvBytes(Hp(h => h with { KvSourceLayer = new[] { -1, 0 } }), 100);
        Assert.Equal(2L * 2 * 8 * 4 * 100, aliased);
    }

    [Fact]
    public void Per_layer_head_dims_and_kv_heads_are_used_when_present()
    {
        var hp = Hp(h => h with { LayerKvHeads = new[] { 1, 4 }, LayerHeadDim = new[] { 8, 16 } });
        Assert.Equal(2L * 4 * (1 * 8 + 4 * 16) * 10, HostMemoryEstimator.KvBytes(hp, 10));
    }

    // ── components ──
    [Fact]
    public void Weights_are_the_exact_sum_and_repack_counts_only_dense_Q4_K()
    {
        var tensors = new[]
        {
            Q4K("blk.0.attn_q.weight"), Q4K("blk.0.ffn_down.weight"),
            Q4K("blk.0.ffn_down_exps.weight"),            // routed experts: not repacked
            F32("blk.0.attn_norm.weight", 64),
        };
        var e = HostMemoryEstimator.Estimate(tensors, Hp(), "Dense", 100, null);
        Assert.Equal(3 * Q4KBytes + 64 * 4, Part(e, "weights"));
        Assert.Equal((long)(2 * Q4KBytes * (1216.0 / 1152.0)), Part(e, "q4k_repack"));
        Assert.Equal(Certainty.Known, e.Components.Single(c => c.Name == "weights").Certainty);
    }

    [Fact]
    public void Repack_is_capped_at_a_quarter_of_the_budget()
    {
        var tensors = Enumerable.Range(0, 100).Select(i => Q4K($"blk.{i}.ffn_up.weight")).ToArray();
        long uncapped = Part(HostMemoryEstimator.Estimate(tensors, Hp(), "Dense", 100, null), "q4k_repack");
        long cap = uncapped / 2 * 4; // a budget whose quarter is half the uncapped copy
        var capped = HostMemoryEstimator.Estimate(tensors, Hp(), "Dense", 100, cap);
        Assert.Equal(cap / 4, Part(capped, "q4k_repack"));
        Assert.Contains("capped", capped.Components.Single(c => c.Name == "q4k_repack").Basis);
    }

    [Fact]
    public void Total_is_the_sum_of_its_components_and_Estimated()
    {
        var e = HostMemoryEstimator.Estimate([Q4K("blk.0.attn_q.weight")], Hp(), "Dense", 100, null);
        Assert.Equal(Certainty.Estimated, e.Certainty);
        Assert.Equal(e.Components.Sum(c => c.Bytes!.Value), e.Bytes);
        Assert.Equal(HostMemoryEstimator.BaseOverheadBytes, Part(e, "base"));
    }

    [Fact]
    public void Estimate_grows_with_context_and_the_context_is_capped_at_the_models_limit()
    {
        var small = HostMemoryEstimator.Estimate([Q4K("a")], Hp(), "Dense", 100, null).Bytes!.Value;
        var large = HostMemoryEstimator.Estimate([Q4K("a")], Hp(), "Dense", 900, null).Bytes!.Value;
        var beyond = HostMemoryEstimator.Estimate([Q4K("a")], Hp(), "Dense", 50_000, null).Bytes!.Value; // model limit is 1000
        var atLimit = HostMemoryEstimator.Estimate([Q4K("a")], Hp(), "Dense", 1000, null).Bytes!.Value;
        Assert.True(large > small);
        Assert.Equal(atLimit, beyond);
    }

    // ── Unknown stays Unknown ──
    [Theory]
    [InlineData("DeepSeek2Mla")]
    [InlineData("HybridGdn")]
    [InlineData("Rwkv")]
    [InlineData(null)]
    public void Families_without_a_modelled_state_layout_are_Unknown_with_null_terms(string? family)
    {
        var e = HostMemoryEstimator.Estimate([Q4K("a")], Hp(), family, 100, null);
        Assert.Equal(Certainty.Unknown, e.Certainty);
        Assert.Null(e.Bytes);
        Assert.All(e.Components.Where(c => c.Name is "kv_cache" or "prefill_scratch"), c => { Assert.Null(c.Bytes); Assert.Equal(Certainty.Unknown, c.Certainty); });
        Assert.Equal(Q4KBytes, Part(e, "weights")); // what is known is still reported
    }

    [Fact]
    public void Unresolvable_hyperparameters_are_Unknown()
    {
        var e = HostMemoryEstimator.Estimate([Q4K("a")], null, "Dense", 100, null);
        Assert.Equal(Certainty.Unknown, e.Certainty);
        Assert.Null(e.Bytes);
    }

    [Theory]
    [InlineData("block_count")]
    [InlineData("embedding_length")]
    [InlineData("attention.head_count_kv")]
    [InlineData("head dimension")]
    [InlineData("feed_forward_length")]
    public void Zero_valued_essentials_make_the_estimate_Unknown_instead_of_understating_it(string missing)
    {
        var hp = missing switch
        {
            "block_count" => Hp(h => h with { NumLayers = 0 }),
            "embedding_length" => Hp(h => h with { EmbeddingDim = 0 }),
            "attention.head_count_kv" => Hp(h => h with { NumKvHeads = 0 }),
            "head dimension" => Hp(h => h with { HeadDim = 0 }),
            _ => Hp(h => h with { IntermediateDim = 0 }),
        };
        var e = HostMemoryEstimator.Estimate([Q4K("a")], hp, "Dense", 100, null);
        Assert.Null(e.Bytes);
        Assert.Contains(missing, e.Source);
    }

    // ── the gate, end to end through the analyzer ──
    private static ScoutInput LlamaInput(int layers = 2, DType dt = DType.Q4_K)
    {
        var md = new Dictionary<string, object>
        {
            ["general.architecture"] = "llama",
            ["llama.block_count"] = (uint)layers,
            ["llama.embedding_length"] = 64u,
            ["llama.attention.head_count"] = 8u,
            ["llama.attention.head_count_kv"] = 2u,
            ["llama.feed_forward_length"] = 128u,
            ["llama.context_length"] = 1000u,
            ["tokenizer.ggml.model"] = "gpt2",
            ["tokenizer.ggml.merges"] = new string[] { "a b" },
        };
        var t = new List<GgufTensorInfo> { new("token_embd.weight", 2, [64, 100], dt, 0) };
        for (int i = 0; i < layers; i++)
            foreach (string n in new[] { "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up" })
                t.Add(new($"blk.{i}.{n}.weight", 2, [64, 64], dt, 0));
        return new ScoutInput("t.gguf", 1, 3, (ulong)t.Count, (ulong)md.Count, md, t);
    }

    private static ScoutReport Run(long? budget, long reserve = 0, int ctx = 100) =>
        ScoutAnalyzer.Analyze(LlamaInput(), new ScoutOptions("t", budget, reserve, null, ctx));

    [Fact]
    public void A_complete_dense_file_gets_an_Estimated_peak_and_the_gate_compares_it_to_the_budget()
    {
        var open = Run(1L << 40);
        Assert.Equal(Certainty.Estimated, open.Resources.HostWorkingSet.Certainty);
        long peak = open.Resources.HostWorkingSet.Bytes!.Value;
        Assert.Equal("allowed", open.Resources.ExecutionDecision);
        Assert.Equal(StageState.Passed, open.Stages.Single(s => s.Stage == "2_feasibility").State);

        Assert.Equal("allowed", Run(peak).Resources.ExecutionDecision);       // exactly fits
        Assert.Equal("blocked", Run(peak - 1).Resources.ExecutionDecision);   // one byte short
    }

    [Fact]
    public void The_reserve_counts_against_the_budget()
    {
        long peak = Run(1L << 40).Resources.HostWorkingSet.Bytes!.Value;
        Assert.Equal("allowed", Run(peak + 1000, reserve: 1000).Resources.ExecutionDecision);
        var r = Run(peak + 999, reserve: 1000);
        Assert.Equal("blocked", r.Resources.ExecutionDecision);
        Assert.Contains("exceeds", r.Resources.Reason);
        Assert.Equal(StageState.Blocked, r.Stages.Single(s => s.Stage == "2_feasibility").State);
    }

    [Fact]
    public void Without_a_budget_the_estimate_is_shown_but_nothing_is_decided()
    {
        var r = Run(null);
        Assert.Equal("not_assessed", r.Resources.ExecutionDecision);
        Assert.Contains("estimated peak", r.Resources.Reason);
        Assert.Equal(StageState.NotRun, r.Stages.Single(s => s.Stage == "2_feasibility").State);
        Assert.Equal(100, r.Resources.ContextTokens);
    }

    [Fact]
    public void Unknown_is_blocked_under_any_budget_for_an_unregistered_architecture()
    {
        var input = LlamaInput();
        var md = new Dictionary<string, object>(input.Metadata) { ["general.architecture"] = "mycoolllm" };
        var r = ScoutAnalyzer.Analyze(input with { Metadata = md }, new ScoutOptions("t", 1L << 50, 0, null, 100));
        Assert.Equal(Certainty.Unknown, r.Resources.HostWorkingSet.Certainty);
        Assert.Null(r.Resources.HostWorkingSet.Bytes);
        Assert.Equal("blocked", r.Resources.ExecutionDecision);
    }

    [Fact]
    public void Kv_cache_is_reported_as_its_own_estimate_for_modelled_families()
    {
        var r = Run(1L << 40, ctx: 100);
        Assert.Equal(2L * 2 * 2 * 8 * 100 * 4, r.Resources.KvCache.Bytes); // 2 layers x K+V x 2 kv heads x head_dim 8 x 100 x 4
        Assert.Equal(Certainty.Estimated, r.Resources.KvCache.Certainty);
        Assert.Equal(5, r.Resources.WorkingSetComponents.Count);
    }

    [Fact]
    public void Components_serialize_with_null_bytes_for_unknown_terms()
    {
        var input = LlamaInput();
        var md = new Dictionary<string, object>(input.Metadata) { ["general.architecture"] = "mycoolllm" };
        string json = System.Text.Json.JsonSerializer.Serialize(
            ScoutAnalyzer.Analyze(input with { Metadata = md }, new ScoutOptions("t", 1L << 40)), ScoutJsonContext.Default.ScoutReport);
        using var doc = System.Text.Json.JsonDocument.Parse(json);
        var kv = doc.RootElement.GetProperty("resources").GetProperty("working_set_components").EnumerateArray().Single(c => c.GetProperty("name").GetString() == "kv_cache");
        Assert.Equal(System.Text.Json.JsonValueKind.Null, kv.GetProperty("bytes").ValueKind);
        Assert.Equal("Unknown", kv.GetProperty("certainty").GetString());
    }
}
