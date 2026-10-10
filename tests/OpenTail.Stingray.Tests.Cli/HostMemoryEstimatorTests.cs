using OpenTail.Stingray.Engine.Scout;
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

    // ── MLA (DeepSeek2): calibrated on DeepSeek-V2-Lite (fit) and Kimi-VL (slope blind), 2026-10-10; see 061-coverage-tooling.md ──
    [Fact]
    public void Mla_kv_is_the_expanded_fp32_K_plus_V_per_head_and_the_estimate_is_Estimated()
    {
        var hp = Hp(h => h with { KvLoraRank = 512, MlaVHeadDim = 128, HeadDim = 192, NumHeads = 16 });
        var e = HostMemoryEstimator.Estimate([Q4K("a")], hp, "DeepSeek2Mla", 100, null);
        Assert.Equal(Certainty.Estimated, e.Certainty);
        Assert.Equal(2L * 16 * (192 + 128) * 100 * 4, Part(e, "kv_cache"));
        Assert.Contains(e.Components, c => c.Name == "mla_base");
        Assert.DoesNotContain(e.Components, c => c.Name == "mla_absorbed_expansion");
    }

    [Fact]
    public void Mla_with_split_kv_b_tensors_adds_the_fixed_expansion_term()
    {
        var hp = Hp(h => h with { KvLoraRank = 512, MlaVHeadDim = 128, HeadDim = 192, NumHeads = 16 });
        var e = HostMemoryEstimator.Estimate([Q4K("blk.0.attn_k_b.weight")], hp, "DeepSeek2Mla", 100, null);
        Assert.Equal(2L * 512 * 16 * (192 + 128) * 4, Part(e, "mla_absorbed_expansion"));
    }

    [Fact]
    public void Mla_without_latent_rank_stays_Unknown()
    {
        var e = HostMemoryEstimator.Estimate([Q4K("a")], Hp(h => h with { KvLoraRank = 0, MlaVHeadDim = 128 }), "DeepSeek2Mla", 100, null);
        Assert.Equal(Certainty.Unknown, e.Certainty);
    }

    // ── RWKV7: calibrated on G1h 1.5B and 2.9B Q4_K_M peak runs, 2026-10-10 ──
    [Fact]
    public void Rwkv_has_no_kv_no_repack_and_a_fixed_size_state_even_when_head_count_is_zero()
    {
        // RWKV GGUFs declare attention.head_count 0, which the attention-shaped completeness check would reject.
        var hp = Hp(h => h with { NumHeads = 0, NumKvHeads = 0, NumLayers = 24, EmbeddingDim = 2048, HeadDim = 64, IntermediateDim = 8192 });
        var e = HostMemoryEstimator.Estimate([Q4K("a")], hp, "Rwkv", 100, null);
        var e2 = HostMemoryEstimator.Estimate([Q4K("a")], hp, "Rwkv", 100000, null);
        Assert.Equal(Certainty.Estimated, e.Certainty);
        Assert.Equal(0, Part(e, "q4k_repack"));
        Assert.Equal(0, Part(e, "kv_cache"));
        Assert.Equal(24L * (2048 * 64 + 2 * 2048) * 4, Part(e, "rwkv_state"));
        Assert.Equal(e.Bytes, e2.Bytes); // flat in context
    }

    [Fact]
    public void Rwkv_without_layers_or_width_is_Unknown()
    {
        var e = HostMemoryEstimator.Estimate([Q4K("a")], Hp(h => h with { NumLayers = 0 }), "Rwkv", 100, null);
        Assert.Equal(Certainty.Unknown, e.Certainty);
    }

    // ── Unknown stays Unknown ──
    [Theory]
    [InlineData("DeepSeek2Mla")]
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

    // ── the hybrid recurrent family (Gated-DeltaNet + full attention) ──
    private static ModelHyperparams HybridHp(Func<ModelHyperparams, ModelHyperparams>? tweak = null, int layers = 8, int interval = 4)
    {
        var types = Enumerable.Range(0, layers).Select(i => (i + 1) % interval == 0 ? LayerType.Attention : LayerType.GatedDeltaNet).ToArray();
        var hp = new ModelHyperparams
        {
            NumLayers = layers, EmbeddingDim = 64, NumHeads = 4, NumKvHeads = 2, HeadDim = 16, IntermediateDim = 128,
            ContextLength = 1000, VocabSize = 100, IsHybridSsm = true, LayerTypes = types,
            Gdn = new GdnConfig(NumKHeads: 2, NumVHeads: 4, HeadDim: 8, InnerSize: 32, ConvKernel: 4, FullAttentionInterval: interval),
        };
        return tweak is null ? hp : tweak(hp);
    }

    [Fact]
    public void Hybrid_terms_are_the_documented_formulas_with_no_repack_and_attention_layers_only_in_the_kv()
    {
        var e = HostMemoryEstimator.Estimate([Q4K("blk.0.attn_q.weight"), Q4K("blk.0.ffn_down.weight")], HybridHp(), "HybridGdn", 100, null);
        Assert.Equal(Certainty.Estimated, e.Certainty);
        Assert.Equal(["weights", "q4k_repack", "base", "gdn_state", "kv_cache", "prefill_scratch"], e.Components.Select(c => c.Name));
        Assert.Equal(0, Part(e, "q4k_repack"));                                    // Q4_K weights are present, but this path builds no repack cache
        Assert.Equal(HostMemoryEstimator.HybridBaseOverheadBytes, Part(e, "base"));
        // 6 GDN layers x ((4-1) x convChannels 64 + 4 heads x 8 x 8) floats x 4 bytes; convChannels = 2 x (2x8) + 4x8
        Assert.Equal(6L * (3 * 64 + 4 * 8 * 8) * 4, Part(e, "gdn_state"));
        // 2 attention layers x K+V x 2 kv heads x head dim 16 x 100 tokens x 4 bytes
        Assert.Equal(2L * 2 * 2 * 16 * 100 * 4, Part(e, "kv_cache"));
        // 1.10 x 100 x (3 x 64 + 5 x valueDim 32 + 3 x ffn 128) x 4
        Assert.Equal((long)(1.10 * 100 * (3 * 64 + 5 * 32 + 3 * 128) * 4), Part(e, "prefill_scratch"));
        Assert.Equal(e.Components.Sum(c => c.Bytes!.Value), e.Bytes);
    }

    [Fact]
    public void Hybrid_state_does_not_grow_with_context_but_kv_and_scratch_do()
    {
        var small = HostMemoryEstimator.Estimate([Q4K("a")], HybridHp(), "HybridGdn", 100, null);
        var large = HostMemoryEstimator.Estimate([Q4K("a")], HybridHp(), "HybridGdn", 900, null);
        Assert.Equal(Part(small, "gdn_state"), Part(large, "gdn_state"));
        Assert.True(Part(large, "kv_cache") > Part(small, "kv_cache"));
        Assert.True(Part(large, "prefill_scratch") > Part(small, "prefill_scratch"));
    }

    [Fact]
    public void Hybrid_mtp_head_layers_each_add_a_kv_layer()
    {
        long plain = Part(HostMemoryEstimator.Estimate([Q4K("a")], HybridHp(), "HybridGdn", 100, null), "kv_cache");
        long withMtp = Part(HostMemoryEstimator.Estimate([Q4K("a")], HybridHp(h => h with { NumMtpLayers = 1 }), "HybridGdn", 100, null), "kv_cache");
        Assert.Equal(plain / 2 * 3, withMtp); // 2 attention layers -> 3
    }

    [Fact]
    public void Hybrid_moe_adds_the_expert_buffers_to_the_scratch()
    {
        var dense = HostMemoryEstimator.Estimate([Q4K("a")], HybridHp(), "HybridGdn", 100, null);
        var moe = HostMemoryEstimator.Estimate([Q4K("a")], HybridHp(h => h with { IsMoE = true, NumExperts = 16, NumActiveExperts = 4, ExpertIntermediateDim = 32, SharedExpertIntermediateDim = 32 }), "HybridGdn", 100, null);
        long ffn = Math.Max(128, 4L * 32 + 32);                 // max(dense ffn, topk x expertFfn + shared)
        long moeFloats = 4L * (2 * 32 + 64);                    // topk x (2 x expertFfn + hidden)
        Assert.Equal((long)(1.10 * 100 * (3 * 64 + 5 * 32 + 3 * ffn + moeFloats) * 4), Part(moe, "prefill_scratch"));
        Assert.True(Part(moe, "prefill_scratch") > Part(dense, "prefill_scratch"));
    }

    [Theory]
    [InlineData("noGdn")]
    [InlineData("noLayerTypes")]
    [InlineData("emptyLayerTypes")]
    [InlineData("zeroHeads")]
    public void Hybrid_with_incomplete_configuration_is_Unknown_with_null_terms_not_a_guess(string what)
    {
        var hp = what switch
        {
            "noGdn" => HybridHp(h => h with { Gdn = null }),
            "noLayerTypes" => HybridHp(h => h with { LayerTypes = null }),
            "emptyLayerTypes" => HybridHp(h => h with { LayerTypes = [] }),
            _ => HybridHp(h => h with { Gdn = h.Gdn! with { NumVHeads = 0 } }),
        };
        var e = HostMemoryEstimator.Estimate([Q4K("a")], hp, "HybridGdn", 100, null);
        Assert.Equal(Certainty.Unknown, e.Certainty);
        Assert.Null(e.Bytes);
        Assert.All(e.Components.Where(c => c.Name is "kv_cache" or "prefill_scratch"), c => Assert.Null(c.Bytes));
        Assert.Equal(Q4KBytes, Part(e, "weights"));      // what is known is still reported
    }

    [Fact]
    public void The_same_Q4_K_weights_carry_a_repack_copy_on_the_dense_path_and_none_on_the_hybrid_path()
    {
        // The measured fact behind the separate model: a dense Q4_K_M file costs ~1.8x its size, a hybrid one about its size plus a constant.
        var tensors = Enumerable.Range(0, 200).Select(i => Q4K($"blk.{i}.ffn_up.weight")).ToArray();
        var dense = HostMemoryEstimator.Estimate(tensors, Hp(), "Dense", 100, null);
        var hybrid = HostMemoryEstimator.Estimate(tensors, HybridHp(), "HybridGdn", 100, null);
        Assert.Equal((long)(200 * Q4KBytes * (1216.0 / 1152.0)), Part(dense, "q4k_repack"));
        Assert.Equal(0, Part(hybrid, "q4k_repack"));
        Assert.Equal(Part(dense, "weights"), Part(hybrid, "weights"));
    }
    // ── dense MoE now counts the expert prefill buffers ──
    [Fact]
    public void Dense_moe_scratch_includes_the_expert_buffers()
    {
        var hp = Hp(h => h with { IsMoE = true, NumExperts = 8, NumActiveExperts = 2, ExpertIntermediateDim = 64, SharedExpertIntermediateDim = 0 });
        var e = HostMemoryEstimator.Estimate([Q4K("a")], hp, "Dense", 100, null);
        long ffn = Math.Max(128, 2L * 64);
        long expected = (long)(1.5 * 100 * (3 * ffn + 4L * 64) * 4) + 100L * (2L * (2 * 64 + 64)) * 4;
        Assert.Equal(expected, Part(e, "prefill_scratch"));
        var notMoe = HostMemoryEstimator.Estimate([Q4K("a")], Hp(), "Dense", 100, null);
        Assert.True(Part(e, "prefill_scratch") > Part(notMoe, "prefill_scratch"));
    }
}