namespace OpenTail.Stingray.Engine;

// ============================================================================================
// ALPHA / UNTESTED -- DeepSeek-V3.2 ("deepseek32" GGUF architecture) implementation. Phase 1 of
// docs/058-deepseek-full-lineage-implementation-plan.md, started immediately after Phase 0's
// deepseek4 (V4) work rather than waiting on a V4 checkpoint download, per user direction.
//
// Same status/scope caveats as DeepSeek4Alpha.cs's header apply here: ported from
// examples/llama.cpp/llama.cpp/src/models/deepseek32.cpp (read in full earlier this session) by
// reading the C++ source alone -- NO real DeepSeek-V3.2 GGUF has been loaded, NO output has been
// compared against any reference. "deepseek32" is NOT admitted in ModelCompatibility.cs.
//
// CORRECTION TO THE ORIGINAL PHASE 1 SCOPING (docs/058-...md): deepseek32 is NOT a subset of V4's
// work. It uses DeepSeek-V2/V3's classic MLA (compressed-latent Q/K/V via wq_a/wq_b/wkv_a_mqa/
// wk_b/wv_b, kv_lora_rank) -- confirmed by load_arch_tensors requiring hparams.is_mla() (throws
// otherwise) and creating wk_b/wv_b decompression tensors deepseek4 has no equivalent of. What IS
// shared with deepseek4: the lightning indexer / DSA mechanism (deepseek32's indexer is SIMPLER
// than deepseek4's, though -- see IndexerAttnK below) and single-block MTP. No hyper-connections,
// no CSA/HCA compressed-KV-state cache, no hash-layer routing, no grouped output LoRA -- none of
// deepseek4's Phase 0 CSA/HCA/hyper-connection code is reusable here.
// ============================================================================================

/// <summary>
/// ALPHA/UNTESTED. DeepSeek-V3.2-specific hyperparameters, mirroring the GGUF keys read by
/// <c>llama_model_deepseek32::load_arch_hparams</c> (deepseek32.cpp:6-50). Kept separate from
/// <c>OpenTail.Stingray.Core.ModelHyperparameters</c> for the same reason as
/// <c>DeepSeek4Hyperparams</c> -- not wired into GGUF loading yet.
/// </summary>
public sealed record DeepSeek32Hyperparams
{
    public int NumLayerAll { get; init; }
    public int NumLayerNextn { get; init; }
    public int NumLayer => NumLayerAll - NumLayerNextn;

    public int EmbedDim { get; init; }
    public int NumHeads { get; init; }
    public int NumExperts { get; init; }
    public int NumExpertsUsed { get; init; }

    /// <summary>Expert FFN intermediate width ({arch}.expert_feed_forward_length). deepseek32.cpp:7,25.</summary>
    public int ExpertFeedForwardLength { get; init; }

    /// <summary>RMSNorm epsilon ({arch}.attention.layer_norm_rms_epsilon). deepseek32.cpp:8.</summary>
    public float RmsNormEps { get; init; } = 1e-5f;

    /// <summary>Number of leading dense (non-MoE) layers ({arch}.leading_dense_block_count). deepseek32.cpp:16.</summary>
    public int LeadingDenseBlockCount { get; init; }

    /// <summary>Post-top-k routed-expert weight scale ({arch}.expert_weights_scale). deepseek32.cpp:17.</summary>
    public float ExpertWeightsScale { get; init; } = 1f;

    /// <summary>Whether routed-expert top-k weights are renormalized to sum to 1 ({arch}.expert_weights_norm). deepseek32.cpp:18.</summary>
    public bool ExpertWeightsNorm { get; init; }

    /// <summary>Number of always-active shared experts ({arch}.expert_shared_count). deepseek32.cpp:15,26.</summary>
    public int ExpertSharedCount { get; init; }

    /// <summary>Q down-projection LoRA rank ({arch}.attention.q_lora_rank). deepseek32.cpp:21.</summary>
    public int QLoraRank { get; init; }

    /// <summary>KV down-projection LoRA rank ({arch}.attention.kv_lora_rank) -- the classic MLA compressed latent width. deepseek32.cpp:22.</summary>
    public int KvLoraRank { get; init; }

    /// <summary>MLA's decompressed per-head Q/K width, if declared ({arch}.attention.key_length_mla). deepseek32.cpp:23.</summary>
    public int EmbedHeadKMlaOverride { get; init; }

    /// <summary>MLA's decompressed per-head V width, if declared ({arch}.attention.value_length_mla). deepseek32.cpp:24.</summary>
    public int EmbedHeadVMlaOverride { get; init; }

    /// <summary>Fallback per-head K/Q width when no MLA override is declared ({arch}.attention.key_length).</summary>
    public int HeadDim { get; init; }

    /// <summary>RoPE-rotated portion of each head's width (n_rot).</summary>
    public int RopeDim { get; init; }

    /// <summary>Lightning-indexer head count ({arch}.attention.indexer.head_count). deepseek32.cpp:29.</summary>
    public int IndexerNumHeads { get; init; }

    /// <summary>Lightning-indexer per-head key width ({arch}.attention.indexer.key_length). deepseek32.cpp:30.</summary>
    public int IndexerHeadSize { get; init; }

    /// <summary>Lightning-indexer top-k selection count ({arch}.attention.indexer.top_k). deepseek32.cpp:31.</summary>
    public int IndexerTopK { get; init; }

    /// <summary>MoE gating function ({arch}.expert_gating_func) -- unlike deepseek4, deepseek32 does NOT hard-require sqrt-softplus; read generically. deepseek32.cpp:34.</summary>
    public int ExpertGatingFunc { get; init; }

    /// <summary>
    /// DeepSeek2-style YaRN attention-score correction ({arch}.rope.scaling.yarn_log_multiplier),
    /// pre-divided by 0.1 at load time exactly like deepseek2's [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
    /// (deepseek32.cpp:36-40 -- the SAME correction, confirming deepseek32 reuses deepseek2's
    /// MLA/YaRN formula chain, not deepseek4's). Reuse this codebase's already-verified
    /// deepseek2 kq_scale/mscale derivation (docs/done/032-...md) rather than re-deriving here.
    /// </summary>
    public float RopeYarnLogMul { get; init; }

    public float RopeYarnFactor { get; init; } = 1f;
    public int RopeYarnOrigCtxLen { get; init; }
    public float RopeFreqBase { get; init; } = 10000f;

    /// <summary>Effective MLA per-head K/Q width: the override if declared, else the plain <see cref="HeadDim"/>.</summary>
    public int EffectiveHeadDimK => EmbedHeadKMlaOverride > 0 ? EmbedHeadKMlaOverride : HeadDim;

    /// <summary>Effective MLA per-head V width: the override if declared, else the plain <see cref="HeadDim"/>.</summary>
    public int EffectiveHeadDimV => EmbedHeadVMlaOverride > 0 ? EmbedHeadVMlaOverride : HeadDim;

    /// <summary>
    /// ALPHA/UNTESTED. Reads the <c>deepseek32</c>-specific GGUF metadata keys, using the exact
    /// key strings from <c>llama-arch.cpp</c>'s <c>LLM_KV_*</c> table (same cross-checking
    /// discipline as <c>DeepSeek4Hyperparams.FromGgufMetadata</c>). Pure metadata parsing, no
    /// tensor access.
    /// </summary>
    public static DeepSeek32Hyperparams FromGgufMetadata(
        IReadOnlyDictionary<string, object> metadata, string arch, int numLayerAll,
        int embedDim, int numHeads, int headDim, int ropeDim, int numExperts, int numExpertsUsed)
    {
        int numLayerNextn = GetInt(metadata, $"{arch}.nextn_predict_layers", 0);
        float yarnLogMul = GetFloat(metadata, $"{arch}.rope.scaling.yarn_log_multiplier", 0f);
        if (yarnLogMul != 0f) yarnLogMul /= 0.1f; // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]

        return new DeepSeek32Hyperparams
        {
            NumLayerAll = numLayerAll,
            NumLayerNextn = numLayerNextn,
            EmbedDim = embedDim,
            NumHeads = numHeads,
            HeadDim = headDim,
            RopeDim = ropeDim,
            NumExperts = numExperts,
            NumExpertsUsed = numExpertsUsed,
            ExpertFeedForwardLength = GetInt(metadata, $"{arch}.expert_feed_forward_length"),
            RmsNormEps = GetFloat(metadata, $"{arch}.attention.layer_norm_rms_epsilon", 1e-5f),
            LeadingDenseBlockCount = GetInt(metadata, $"{arch}.leading_dense_block_count"),
            ExpertWeightsScale = GetFloat(metadata, $"{arch}.expert_weights_scale", 1f),
            ExpertWeightsNorm = GetBool(metadata, $"{arch}.expert_weights_norm"),
            ExpertSharedCount = GetInt(metadata, $"{arch}.expert_shared_count"),
            QLoraRank = GetInt(metadata, $"{arch}.attention.q_lora_rank"),
            KvLoraRank = GetInt(metadata, $"{arch}.attention.kv_lora_rank"),
            EmbedHeadKMlaOverride = GetInt(metadata, $"{arch}.attention.key_length_mla"),
            EmbedHeadVMlaOverride = GetInt(metadata, $"{arch}.attention.value_length_mla"),
            IndexerNumHeads = GetInt(metadata, $"{arch}.attention.indexer.head_count"),
            IndexerHeadSize = GetInt(metadata, $"{arch}.attention.indexer.key_length"),
            IndexerTopK = GetInt(metadata, $"{arch}.attention.indexer.top_k"),
            ExpertGatingFunc = GetInt(metadata, $"{arch}.expert_gating_func"),
            RopeYarnLogMul = yarnLogMul,
            RopeYarnFactor = GetFloat(metadata, $"{arch}.rope.scaling.factor", 1f),
            RopeYarnOrigCtxLen = GetInt(metadata, $"{arch}.rope.scaling.original_context_length"),
            RopeFreqBase = GetFloat(metadata, $"{arch}.rope.freq_base", 10000f),
        };
    }

    private static int GetInt(IReadOnlyDictionary<string, object> m, string key, int fallback = 0)
    {
        if (!m.TryGetValue(key, out var v)) return fallback;
        if (v is System.Collections.IList list) return list.Count > 0 ? Convert.ToInt32(list[0]) : fallback;
        return Convert.ToInt32(v);
    }

    private static float GetFloat(IReadOnlyDictionary<string, object> m, string key, float fallback = 0f) =>
        m.TryGetValue(key, out var v) ? Convert.ToSingle(v) : fallback;

    private static bool GetBool(IReadOnlyDictionary<string, object> m, string key, bool fallback = false) =>
        m.TryGetValue(key, out var v) ? Convert.ToBoolean(v) : fallback;
}
