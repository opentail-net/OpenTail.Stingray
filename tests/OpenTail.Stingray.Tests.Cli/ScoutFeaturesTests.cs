using OpenTail.Stingray.Cli.Scout;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>Positive, negative and ambiguous cases for each scout feature rule, over synthetic metadata/tensor indexes.</summary>
public sealed class ScoutFeaturesTests
{
    private static GgufTensorInfo T(string name, params long[] dims) => new(name, dims.Length, dims, DType.Q4_K, 0);

    private static IReadOnlyList<ScoutFinding> Run(string arch, Dictionary<string, object>? md = null, params GgufTensorInfo[] tensors)
    {
        md ??= [];
        md["general.architecture"] = arch;
        return ScoutFeatures.Evaluate(md, tensors);
    }

    private static ScoutFinding? Get(IReadOnlyList<ScoutFinding> fs, string id) => fs.FirstOrDefault(f => f.Id == id);

    // ── MoE ──
    [Fact]
    public void Moe_positive_needs_metadata_and_stacked_tensors_for_Known()
    {
        var fs = Run("x", new() { ["x.expert_count"] = 8u, ["x.expert_used_count"] = 2u },
            T("blk.0.ffn_gate_inp.weight", 64, 8), T("blk.0.ffn_gate_exps.weight", 64, 128, 8));
        var m = Get(fs, "ffn.expert_routed")!;
        Assert.Equal(Certainty.Known, m.Certainty);
        Assert.Contains("top-2", m.Summary);
        Assert.Contains(m.Evidence, e => e.Name == "x.expert_count" && e.Value == "8");
    }

    [Fact]
    public void Moe_negative_for_a_dense_model() =>
        Assert.Null(Get(Run("x", null, T("blk.0.ffn_down.weight", 256, 64)), "ffn.expert_routed"));

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Moe_one_sided_evidence_is_only_a_Hypothesis(bool metadataOnly)
    {
        var md = metadataOnly ? new Dictionary<string, object> { ["x.expert_count"] = 8u } : null;
        var tensors = metadataOnly ? [] : new[] { T("blk.0.ffn_up_exps.weight", 64, 128, 8) };
        Assert.Equal(Certainty.Hypothesis, Get(Run("x", md, tensors), "ffn.expert_routed")!.Certainty);
    }

    [Fact]
    public void A_single_expert_count_of_one_is_not_a_moe()
    {
        Assert.Null(Get(Run("x", new() { ["x.expert_count"] = 1u }), "ffn.expert_routed"));
    }

    [Fact]
    public void Shared_expert_Known_only_with_count_and_tensors()
    {
        var both = Get(Run("x", new() { ["x.expert_shared_count"] = 1u }, T("blk.0.ffn_gate_shexp.weight", 64, 128)), "ffn.shared_expert")!;
        var tensorOnly = Get(Run("x", null, T("blk.0.ffn_gate_shexp.weight", 64, 128)), "ffn.shared_expert")!;
        Assert.Equal(Certainty.Known, both.Certainty);
        Assert.Equal(Certainty.Hypothesis, tensorOnly.Certainty);
        Assert.Null(Get(Run("x", null, T("blk.0.ffn_down.weight", 1, 1)), "ffn.shared_expert"));
    }

    // ── attention layout ──
    [Fact]
    public void Fused_separate_and_mixed_qkv_layouts()
    {
        Assert.NotNull(Get(Run("x", null, T("blk.0.attn_qkv.weight", 64, 192)), "attn.fused_qkv"));
        Assert.NotNull(Get(Run("x", null, T("blk.0.attn_q.weight", 64, 64), T("blk.0.attn_k.weight", 64, 64), T("blk.0.attn_v.weight", 64, 64)), "attn.separate_qkv"));
        var mixed = Run("x", null, T("blk.0.attn_qkv.weight", 64, 192), T("blk.1.attn_q.weight", 64, 64));
        Assert.NotNull(Get(mixed, "attn.qkv_layout_mixed"));
        Assert.Null(Get(mixed, "attn.fused_qkv"));
    }

    [Fact]
    public void Separate_qkv_needs_all_three()
    {
        Assert.Null(Get(Run("x", null, T("blk.0.attn_q.weight", 64, 64), T("blk.0.attn_k.weight", 64, 64)), "attn.separate_qkv"));
    }

    [Fact]
    public void Qk_norm_both_is_Known_one_sided_is_Hypothesis_none_is_absent()
    {
        Assert.Equal(Certainty.Known, Get(Run("x", null, T("blk.0.attn_q_norm.weight", 64), T("blk.0.attn_k_norm.weight", 64)), "attn.qk_norm")!.Certainty);
        Assert.Equal(Certainty.Hypothesis, Get(Run("x", null, T("blk.0.attn_q_norm.weight", 64)), "attn.qk_norm")!.Certainty);
        Assert.Null(Get(Run("x", null, T("blk.0.attn_norm.weight", 64)), "attn.qk_norm"));
    }

    [Theory]
    [InlineData(32u, 8u, "attn.gqa")]
    [InlineData(32u, 32u, "attn.mha")]
    [InlineData(32u, 1u, "attn.mqa")]
    public void Head_counts_classify_attention(uint heads, uint kv, string id)
    {
        var fs = Run("x", new() { ["x.attention.head_count"] = heads, ["x.attention.head_count_kv"] = kv });
        Assert.Equal(Certainty.Known, Get(fs, id)!.Certainty);
        Assert.Equal(2, Get(fs, id)!.Evidence.Count);
    }

    [Fact]
    public void Per_layer_kv_head_arrays_are_flagged_not_misclassified()
    {
        var fs = Run("x", new() { ["x.attention.head_count"] = 32u, ["x.attention.head_count_kv"] = new uint[] { 8, 8, 0, 8 } });
        Assert.NotNull(Get(fs, "attn.kv_heads_per_layer"));
        Assert.Null(Get(fs, "attn.gqa"));
    }

    [Fact]
    public void Missing_head_metadata_yields_no_head_finding() =>
        Assert.DoesNotContain(Run("x"), f => f.Id.StartsWith("attn.g", StringComparison.Ordinal) || f.Id is "attn.mha" or "attn.mqa");

    // ── MLA / indexer ──
    [Fact]
    public void Mla_Known_with_both_Hypothesis_with_one_absent_without()
    {
        var both = Run("x", new() { ["x.attention.kv_lora_rank"] = 512u }, T("blk.0.attn_kv_a_mqa.weight", 64, 576));
        Assert.Equal(Certainty.Known, Get(both, "attn.mla_low_rank")!.Certainty);
        Assert.Equal(Certainty.Hypothesis, Get(Run("x", new() { ["x.attention.kv_lora_rank"] = 512u }), "attn.mla_low_rank")!.Certainty);
        Assert.Equal(Certainty.Hypothesis, Get(Run("x", null, T("blk.0.attn_q_a.weight", 64, 64)), "attn.mla_low_rank")!.Certainty);
        Assert.Null(Get(Run("x", null, T("blk.0.attn_q.weight", 64, 64)), "attn.mla_low_rank"));
    }

    [Fact]
    public void Indexer_requires_name_or_metadata_evidence()
    {
        Assert.NotNull(Get(Run("x", null, T("blk.0.indexer.proj.weight", 64, 64)), "attn.sparse_indexer"));
        Assert.Null(Get(Run("x", null, T("blk.0.attn_q.weight", 64, 64)), "attn.sparse_indexer"));
    }

    // ── recurrent / hybrid ──
    [Fact]
    public void Ssm_tensors_in_some_layers_with_attention_in_others_is_a_hybrid()
    {
        var fs = Run("x", null,
            T("blk.0.ssm_a", 16), T("blk.0.ssm_conv1d.weight", 4, 64),
            T("blk.1.attn_q.weight", 64, 64), T("blk.2.ssm_a", 16));
        var h = Get(fs, "recurrent.hybrid_ssm")!;
        Assert.Equal(Certainty.Known, h.Certainty);
        Assert.Contains(h.Evidence, e => e.Kind == "layers" && e.Name == "ssm" && e.Value == "0,2");
        Assert.Contains("do not say which recurrence", h.Caveat);
    }

    [Fact]
    public void Ssm_without_attention_layers_is_only_a_Hypothesis()
    {
        var fs = Run("x", null, T("blk.0.ssm_a", 16), T("blk.1.ssm_a", 16));
        Assert.Equal(Certainty.Hypothesis, Get(fs, "recurrent.ssm")!.Certainty);
        Assert.Null(Get(fs, "recurrent.hybrid_ssm"));
    }

    [Fact]
    public void A_tensor_merely_containing_ssm_text_is_not_recurrent() =>
        Assert.Null(Get(Run("x", null, T("blk.0.classmates_x.weight", 4)), "recurrent.ssm"));

    [Fact]
    public void Rwkv_time_mix_is_a_Hypothesis() =>
        Assert.Equal(Certainty.Hypothesis, Get(Run("x", null, T("blk.0.time_mix_w1.weight", 4, 4)), "recurrent.rwkv_time_mix")!.Certainty);

    // ── MTP ──
    [Fact]
    public void One_isolated_mtp_like_tensor_is_never_Known()
    {
        var one = Get(Run("x", null, T("blk.40.nextn.eh_proj.weight", 8, 4)), "mtp.nextn_head")!;
        Assert.Equal(Certainty.Hypothesis, one.Certainty);
        Assert.Contains("Not proof", one.Caveat);
    }

    [Fact]
    public void Mtp_Known_needs_metadata_count_and_several_tensors()
    {
        var fs = Run("x", new() { ["x.nextn_predict_layers"] = 1u },
            T("blk.40.nextn.eh_proj.weight", 8, 4), T("blk.40.nextn.enorm.weight", 4));
        Assert.Equal(Certainty.Known, Get(fs, "mtp.nextn_head")!.Certainty);
        Assert.Null(Get(Run("x", null, T("blk.0.attn_q.weight", 4, 4)), "mtp.nextn_head"));
        Assert.Equal(Certainty.Hypothesis, Get(Run("x", new() { ["x.nextn_predict_layers"] = 0u }, T("a.nextn.x", 1), T("b.nextn.y", 1)), "mtp.nextn_head")!.Certainty);
    }

    // ── RoPE ──
    [Fact]
    public void Rope_sections_scaling_and_partial_are_detected_from_metadata()
    {
        var fs = Run("x", new()
        {
            ["x.rope.dimension_sections"] = new int[] { 16, 24, 24, 0 },
            ["x.rope.scaling.type"] = "yarn",
            ["x.rope.scaling.factor"] = 4.0f,
            ["x.rope.dimension_count"] = 64u,
            ["x.attention.key_length"] = 128u,
        });
        Assert.NotNull(Get(fs, "rope.multi_axis"));
        Assert.Contains(Get(fs, "rope.scaling")!.Evidence, e => e.Name == "x.rope.scaling.factor");
        Assert.Contains("64 of 128", Get(fs, "rope.partial")!.Summary);
    }

    [Fact]
    public void Full_rope_none_scaling_and_missing_metadata_report_nothing()
    {
        Assert.Empty(Run("x", new()
        {
            ["x.rope.scaling.type"] = "none",
            ["x.rope.dimension_count"] = 128u,
            ["x.attention.key_length"] = 128u,
        }));
        Assert.Empty(Run("x"));
    }

    [Fact]
    public void Partial_rope_falls_back_to_embedding_over_heads_when_key_length_absent()
    {
        var fs = Run("x", new() { ["x.rope.dimension_count"] = 32u, ["x.embedding_length"] = 4096u, ["x.attention.head_count"] = 32u });
        Assert.Contains("32 of 128", Get(fs, "rope.partial")!.Summary);
    }

    // ── multimodal ──
    [Fact]
    public void Projector_file_is_Known_but_text_file_with_v_prefix_is_only_a_Hypothesis()
    {
        var clip = Run("clip", null, T("v.blk.0.attn_q.weight", 4, 4), T("mm.0.weight", 4, 4));
        Assert.Equal(Certainty.Known, Get(clip, "multimodal.projector_file")!.Certainty);
        Assert.Contains("not a complete user-facing", Get(clip, "multimodal.projector_file")!.Caveat);
        Assert.Null(Get(clip, "multimodal.vision_tensors_in_text_file"));

        var text = Run("llama", null, T("v.blk.0.attn_q.weight", 4, 4));
        Assert.Equal(Certainty.Hypothesis, Get(text, "multimodal.vision_tensors_in_text_file")!.Certainty);
        Assert.Null(Get(text, "multimodal.projector_file"));
    }

    [Fact]
    public void Plain_text_model_has_no_multimodal_findings() =>
        Assert.DoesNotContain(Run("llama", null, T("token_embd.weight", 4, 4), T("blk.0.attn_q.weight", 4, 4)),
            f => f.Id.StartsWith("multimodal", StringComparison.Ordinal));

    // ── quant ──
    [Fact]
    public void Exotic_storage_is_reported_only_when_present()
    {
        var t = new GgufTensorInfo("blk.0.ffn_down.weight", 2, [256, 64], DType.IQ1_S, 0);
        Assert.NotNull(Get(ScoutFeatures.Evaluate(new Dictionary<string, object>(), [t]), "quant.low_bit_or_block_fp"));
        Assert.Null(Get(Run("x", null, T("blk.0.ffn_down.weight", 256, 64)), "quant.low_bit_or_block_fp"));
    }

    // ── integration with the analyzer ──
    [Fact]
    public void Analyzer_includes_feature_findings_in_deterministic_order()
    {
        var md = new Dictionary<string, object> { ["general.architecture"] = "llama", ["llama.attention.head_count"] = 8u, ["llama.attention.head_count_kv"] = 2u };
        var input = new ScoutInput("t.gguf", 1, 3, 1, 2, md, [T("blk.0.attn_q_norm.weight", 4), T("blk.0.attn_k_norm.weight", 4)]);
        var r = ScoutAnalyzer.Analyze(input, new ScoutOptions("b"));
        Assert.Contains(r.Findings, f => f.Id == "attn.gqa");
        Assert.Equal(r.Findings.Select(f => f.Id).Order(StringComparer.Ordinal), r.Findings.Select(f => f.Id));
        Assert.All(r.Findings, f => { Assert.NotEmpty(f.Evidence); Assert.False(string.IsNullOrWhiteSpace(f.Caveat)); });
    }
}
