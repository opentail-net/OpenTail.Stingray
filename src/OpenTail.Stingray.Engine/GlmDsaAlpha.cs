using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine;

// ============================================================================================
// ALPHA / UNTESTED -- GLM-DSA ("glm-dsa" GGUF architecture, GLM-5.2). Ported 2026-10-03 from
// examples/llama.cpp/llama.cpp/src/models/glm-dsa.cpp (upstream b10306) and official GLM-5.2
// configuration (https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json).
//
// "Port now, prove later" (CLAUDE.md rule 14): NO real GLM-5.2 GGUF has been loaded yet.
// "glm-dsa" is deliberately NOT admitted in ModelCompatibility.cs until validated against real checkpoints.
//
// ARCHITECTURE SPEC:
//  - MLA attention with weight absorption:
//      n_head = 64, q_lora_rank = 2048, kv_lora_rank = 512, qk_head_dim = 256 (192 NoPE + 64 RoPE),
//      v_head_dim = 256. Single compressed KV cache: 512 latent + 64 RoPE = 576 floats/token/layer.
//  - DSA Lightning Indexer:
//      indexer_n_head = 32, indexer_head_size = 128, indexer_top_k = 2048.
//      Layout per head is [rope | nope] with n_rot = 64 rotated under LLAMA_ROPE_TYPE_NORM.
//      Orthonormal Sylvester-Walsh-Hadamard transform (PrismHadamard.ApplySylvesterHadamard)
//      applied in-place to indexer_q and indexer_k on 128-element blocks scaled by 1/sqrt(128).
//  - Indexer schedule:
//      Full layers (0, 1, then every 4th layer from 2: 2, 6, 10...) execute the full indexer
//      and update prev_top_k; intermediate layers reuse prev_top_k.
//  - FFN:
//      Leading dense blocks (default 3) using standard SwiGLU.
//      Remaining layers: MoE with 256 routed experts + 1 shared expert. Top-8 routed selection
//      using sigmoid gating, optional ffn_exp_probs_b bias, normalized top-8 weights, scaled by 2.5.
// ============================================================================================

/// <summary>
/// Hyperparameters for GLM-DSA (GLM-5.2), mirroring GGUF metadata keys read by
/// <c>llama_model_glm_dsa::load_arch_hparams</c> (glm-dsa.cpp:29-72).
/// </summary>
public sealed record GlmDsaHyperparams
{
    public int NumLayerAll { get; init; }
    public int NumLayerNextn { get; init; }
    public int NumLayer => NumLayerAll - NumLayerNextn;

    public int EmbedDim { get; init; }
    public int NumHeads { get; init; }
    public int NumExperts { get; init; }
    public int NumExpertsUsed { get; init; }

    /// <summary>Expert intermediate dimension ({arch}.expert_feed_forward_length).</summary>
    public int ExpertFeedForwardLength { get; init; }

    /// <summary>Dense FFN intermediate dimension ({arch}.feed_forward_length).</summary>
    public int IntermediateDim { get; init; }

    /// <summary>RMSNorm epsilon ({arch}.attention.layer_norm_rms_epsilon).</summary>
    public float RmsNormEps { get; init; } = 1e-5f;

    /// <summary>Leading dense (non-MoE) layer count ({arch}.leading_dense_block_count). Default 3.</summary>
    public int LeadingDenseBlockCount { get; init; } = 3;

    /// <summary>Post-top-k routed-expert weight scale ({arch}.expert_weights_scale). Default 2.5.</summary>
    public float ExpertWeightsScale { get; init; } = 2.5f;

    /// <summary>Whether routed expert weights are renormalized to sum to 1 ({arch}.expert_weights_norm). Default true.</summary>
    public bool ExpertWeightsNorm { get; init; } = true;

    /// <summary>Shared expert count ({arch}.expert_shared_count). Default 1.</summary>
    public int ExpertSharedCount { get; init; } = 1;

    /// <summary>Q down-projection LoRA rank ({arch}.attention.q_lora_rank). Default 2048.</summary>
    public int QLoraRank { get; init; } = 2048;

    /// <summary>KV down-projection LoRA rank ({arch}.attention.kv_lora_rank). Default 512.</summary>
    public int KvLoraRank { get; init; } = 512;

    /// <summary>Decompressed MLA key head dim ({arch}.attention.key_length_mla). Default 256.</summary>
    public int EmbedHeadKMlaOverride { get; init; } = 256;

    /// <summary>Decompressed MLA value head dim ({arch}.attention.value_length_mla). Default 256.</summary>
    public int EmbedHeadVMlaOverride { get; init; } = 256;

    /// <summary>Fallback head dim when MLA override is not set ({arch}.attention.key_length).</summary>
    public int HeadDim { get; init; } = 256;

    public int EffectiveHeadDimK => EmbedHeadKMlaOverride > 0 ? EmbedHeadKMlaOverride : HeadDim;
    public int EffectiveHeadDimV => EmbedHeadVMlaOverride > 0 ? EmbedHeadVMlaOverride : HeadDim;

    /// <summary>RoPE-rotated portion of each head's width (n_rot, default 64).</summary>
    public int RopeDim { get; init; } = 64;

    /// <summary>Indexer head count ({arch}.attention.indexer.head_count). Default 32.</summary>
    public int IndexerNumHeads { get; init; } = 32;

    /// <summary>Indexer key length per head ({arch}.attention.indexer.key_length). Default 128.</summary>
    public int IndexerHeadSize { get; init; } = 128;

    /// <summary>Indexer top-k candidate count ({arch}.attention.indexer.top_k). Default 2048.</summary>
    public int IndexerTopK { get; init; } = 2048;

    /// <summary>MoE gating function ({arch}.expert_gating_func). 2 = Sigmoid.</summary>
    public int ExpertGatingFunc { get; init; } = 2;

    /// <summary>Indexer type schedule per layer (1 = full indexer, 0 = shared/reuse prev_top_k).</summary>
    public int[]? IndexerTypes { get; init; }

    /// <summary>Determines if a given layer runs a full indexer refresh or reuses prev_top_k.</summary>
    public bool IsIndexerFull(int layer)
    {
        if (IndexerTypes is not null && layer < IndexerTypes.Length)
            return IndexerTypes[layer] != 0;
        // GLM-5.2 default pattern: layers 0, 1 full; then 1 full every 4 layers starting at 2 (2, 6, 10, ...)
        return layer == 0 || layer == 1 || ((layer - 2) % 4 == 0);
    }

    public float RopeYarnLogMul { get; init; }
    public float RopeYarnFactor { get; init; } = 1f;
    public int RopeYarnOrigCtxLen { get; init; }
    public float RopeFreqBase { get; init; } = 10000f;

    /// <summary>
    /// Reads glm-dsa GGUF metadata keys, mirroring llama_model_glm_dsa::load_arch_hparams.
    /// </summary>
    public static GlmDsaHyperparams FromGgufMetadata(
        IReadOnlyDictionary<string, object> metadata, string arch, int numLayerAll,
        int embedDim, int numHeads, int headDim, int ropeDim, int numExperts, int numExpertsUsed)
    {
        int numLayerNextn = GetInt(metadata, $"{arch}.nextn_predict_layers", 0);
        float yarnLogMul = GetFloat(metadata, $"{arch}.rope.scaling.yarn_log_multiplier", 0f);
        if (yarnLogMul != 0f) yarnLogMul /= 0.1f;

        int[]? indexerTypes = null;
        if (metadata.TryGetValue($"{arch}.attention.indexer.types", out var itVal) && itVal is System.Collections.IList itList)
        {
            indexerTypes = new int[itList.Count];
            for (int i = 0; i < itList.Count; i++) indexerTypes[i] = Convert.ToInt32(itList[i]);
        }

        return new GlmDsaHyperparams
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
            IntermediateDim = GetInt(metadata, $"{arch}.feed_forward_length"),
            RmsNormEps = GetFloat(metadata, $"{arch}.attention.layer_norm_rms_epsilon", 1e-5f),
            LeadingDenseBlockCount = GetInt(metadata, $"{arch}.leading_dense_block_count", 3),
            ExpertWeightsScale = GetFloat(metadata, $"{arch}.expert_weights_scale", 2.5f),
            ExpertWeightsNorm = GetBool(metadata, $"{arch}.expert_weights_norm", true),
            ExpertSharedCount = GetInt(metadata, $"{arch}.expert_shared_count", 1),
            QLoraRank = GetInt(metadata, $"{arch}.attention.q_lora_rank", 2048),
            KvLoraRank = GetInt(metadata, $"{arch}.attention.kv_lora_rank", 512),
            EmbedHeadKMlaOverride = GetInt(metadata, $"{arch}.attention.key_length_mla", 256),
            EmbedHeadVMlaOverride = GetInt(metadata, $"{arch}.attention.value_length_mla", 256),
            IndexerNumHeads = GetInt(metadata, $"{arch}.attention.indexer.head_count", 32),
            IndexerHeadSize = GetInt(metadata, $"{arch}.attention.indexer.key_length", 128),
            IndexerTopK = GetInt(metadata, $"{arch}.attention.indexer.top_k", 2048),
            IndexerTypes = indexerTypes,
            ExpertGatingFunc = GetInt(metadata, $"{arch}.expert_gating_func", 2),
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

/// <summary>
/// Resolved tensor references for one GLM-DSA decoder layer.
/// </summary>
public sealed unsafe class GlmDsaLayerTensors
{
    public DeepSeek4TensorRef? AttnNorm;
    public DeepSeek4TensorRef? AttnQANorm;
    public DeepSeek4TensorRef? AttnKvANorm;
    public DeepSeek4TensorRef? WqA;
    public DeepSeek4TensorRef? WqB;
    public DeepSeek4TensorRef? WkvAMqa;
    public DeepSeek4TensorRef? WkB;
    public DeepSeek4TensorRef? WvB;
    public DeepSeek4TensorRef? Wo;
    public DeepSeek4TensorRef? FfnNorm;

    // DSA indexer tensors
    public DeepSeek4TensorRef? IndexerKNorm;
    public DeepSeek4TensorRef? IndexerKNormBias;
    public DeepSeek4TensorRef? IndexerProj;
    public DeepSeek4TensorRef? IndexerAttnK;
    public DeepSeek4TensorRef? IndexerAttnQB;

    // Dense leading layers (i < LeadingDenseBlockCount)
    public DeepSeek4TensorRef? FfnGate;
    public DeepSeek4TensorRef? FfnUp;
    public DeepSeek4TensorRef? FfnDown;

    // MoE layers (i >= LeadingDenseBlockCount)
    public DeepSeek4TensorRef? FfnGateInp;
    public DeepSeek4TensorRef? FfnExpProbsB;
    public DeepSeek4TensorRef? FfnGateExps;
    public DeepSeek4TensorRef? FfnDownExps;
    public DeepSeek4TensorRef? FfnUpExps;
    public DeepSeek4TensorRef? FfnGateShexp;
    public DeepSeek4TensorRef? FfnDownShexp;
    public DeepSeek4TensorRef? FfnUpShexp;
}

/// <summary>
/// Resolves GLM-DSA tensors from a GGUF model container.
/// </summary>
public sealed unsafe class GlmDsaTensorSet
{
    public DeepSeek4TensorRef TokEmbd { get; }
    public DeepSeek4TensorRef OutputNorm { get; }
    public DeepSeek4TensorRef Output { get; }
    public IReadOnlyList<GlmDsaLayerTensors> Layers { get; }

    public GlmDsaTensorSet(DeepSeek4TensorRef tokEmbd, DeepSeek4TensorRef outputNorm, DeepSeek4TensorRef output, IReadOnlyList<GlmDsaLayerTensors> layers)
    {
        TokEmbd = tokEmbd;
        OutputNorm = outputNorm;
        Output = output;
        Layers = layers;
    }

    public static GlmDsaTensorSet Load(GgufModel model, GlmDsaHyperparams hp)
    {
        DeepSeek4TensorRef Required(string name)
        {
            var info = model.FindTensor(name)
                ?? throw new InvalidOperationException($"Missing required glm-dsa tensor: {name}");
            return new DeepSeek4TensorRef(name, info, model.GetTensorDataPtr(info));
        }

        DeepSeek4TensorRef? Optional(string name)
        {
            var info = model.FindTensor(name);
            return info is null ? null : new DeepSeek4TensorRef(name, info.Value, model.GetTensorDataPtr(info.Value));
        }

        var tokEmbd = Required("token_embd.weight");
        var outputNorm = Required("output_norm.weight");
        var output = model.FindTensor("output.weight") is { } outInfo
            ? new DeepSeek4TensorRef("output.weight", outInfo, model.GetTensorDataPtr(outInfo))
            : tokEmbd;

        var layers = new List<GlmDsaLayerTensors>(hp.NumLayer);
        for (int i = 0; i < hp.NumLayer; i++)
        {
            var lt = new GlmDsaLayerTensors
            {
                AttnNorm = Required($"blk.{i}.attn_norm.weight"),
                AttnQANorm = Required($"blk.{i}.attn_q_a_norm.weight"),
                AttnKvANorm = Required($"blk.{i}.attn_kv_a_norm.weight"),
                WqA = Required($"blk.{i}.attn_q_a.weight"),
                WqB = Required($"blk.{i}.attn_q_b.weight"),
                WkvAMqa = Required($"blk.{i}.attn_kv_a_mqa.weight"),
                WkB = Required($"blk.{i}.attn_k_b.weight"),
                WvB = Required($"blk.{i}.attn_v_b.weight"),
                Wo = Optional($"blk.{i}.attn_output.weight") ?? Required($"blk.{i}.attn_out.weight"),
                FfnNorm = Required($"blk.{i}.ffn_norm.weight"),

                // DSA indexer tensors
                IndexerKNorm = Optional($"blk.{i}.indexer_k_norm.weight"),
                IndexerKNormBias = Optional($"blk.{i}.indexer_k_norm.bias"),
                IndexerProj = Optional($"blk.{i}.indexer_proj.weight"),
                IndexerAttnK = Optional($"blk.{i}.indexer_attn_k.weight"),
                IndexerAttnQB = Optional($"blk.{i}.indexer_attn_q_b.weight"),
            };

            if (i < hp.LeadingDenseBlockCount)
            {
                lt.FfnGate = Required($"blk.{i}.ffn_gate.weight");
                lt.FfnUp = Required($"blk.{i}.ffn_up.weight");
                lt.FfnDown = Required($"blk.{i}.ffn_down.weight");
            }
            else
            {
                lt.FfnGateInp = Required($"blk.{i}.ffn_gate_inp.weight");
                lt.FfnExpProbsB = Optional($"blk.{i}.ffn_exp_probs_b.bias");
                lt.FfnGateExps = Required($"blk.{i}.ffn_gate_exps.weight");
                lt.FfnDownExps = Required($"blk.{i}.ffn_down_exps.weight");
                lt.FfnUpExps = Required($"blk.{i}.ffn_up_exps.weight");
                lt.FfnGateShexp = Required($"blk.{i}.ffn_gate_shexp.weight");
                lt.FfnDownShexp = Required($"blk.{i}.ffn_down_shexp.weight");
                lt.FfnUpShexp = Required($"blk.{i}.ffn_up_shexp.weight");
            }

            layers.Add(lt);
        }

        return new GlmDsaTensorSet(tokEmbd, outputNorm, output, layers);
    }
}
