using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine;

// ============================================================================================
// ALPHA / UNTESTED -- GLM5-Next (GLM-5.3-Flash, "glm5next" GGUF architecture). Ported 2026-10-03
// from examples/llama.cpp/llama.cpp/src/models/glm5-next.cpp (upstream b10306) and official
// Glm5NextForConditionalGeneration configuration.
//
// "Port now, prove later" (CLAUDE.md rule 14): NO real GLM-5.3-Flash GGUF has been loaded yet.
// "glm5next" is deliberately NOT admitted in ModelCompatibility.cs until validated against real checkpoints.
//
// ARCHITECTURE SPEC:
//  - Hybrid 45-layer trunk in a strict 3:1 pattern (34 KDA + 11 MLA layers):
//      i % 4 != 3 -> KDA (Gated DeltaNet linear attention with 1D causal conv)
//      i % 4 == 3 -> DSA / NoPE MLA (compressed latent MLA with 4-token K-pool indexer)
//  - 4-stream mHC (multi-head hyper-connections):
//      4 residual streams, Sinkhorn balancing (20 iterations, eps 1e-6) on the 4x4 combine matrix,
//      mixing down before each sublayer norm and expanding back across streams.
//  - K-pool DSA Indexer:
//      Pools consecutive blocks of 4 tokens using learned gate + attention position embeddings (APE),
//      selecting top-k pools for sparse MLA attention.
//  - MoE:
//      3 leading dense SwiGLU blocks; remaining layers have 288 routed experts + 1 shared expert.
//      Top-8 routed selection using sigmoid gating with bias, normalized top-8 weights, scale 2.5.
//      SwiGLU intermediate clamp at 10.0.
// ============================================================================================

/// <summary>
/// Hyperparameters for GLM5-Next (GLM-5.3-Flash), mirroring GGUF metadata keys read by
/// <c>llama_model_glm5_next::load_arch_hparams</c> (glm5-next.cpp:7-59).
/// </summary>
public sealed record Glm5NextHyperparams
{
    public int NumLayerAll { get; init; } = 45;
    public int NumLayerNextn { get; init; }
    public int NumLayer => NumLayerAll - NumLayerNextn;

    public int EmbedDim { get; init; } = 4096;
    public int NumHeads { get; init; } = 32;
    public int NumExperts { get; init; } = 288;
    public int NumExpertsUsed { get; init; } = 8;

    // KDA parameters
    public int HeadDimKda { get; init; } = 128;
    public int SsmDConv { get; init; } = 4;
    public float KdaGateLowerBound { get; init; } = float.NegativeInfinity;
    public int DInner => HeadDimKda * NumHeads;

    // MLA parameters
    public int QLoraRank { get; init; } = 1536;
    public int KvLoraRank { get; init; } = 512;
    public int EmbedHeadKMla { get; init; } = 128;
    public int EmbedHeadVMla { get; init; } = 128;

    // DSA K-pool parameters
    public int IndexerNumHeads { get; init; } = 32;
    public int IndexerHeadSize { get; init; } = 128;
    public int IndexerTopK { get; init; } = 2048;
    public int IndexerKPool { get; init; } = 4;
    public bool IndexerKPoolSelectTail { get; init; } = true;

    // mHC parameters
    public int HcMult { get; init; } = 4;
    public int HcSinkhornIters { get; init; } = 20;
    public float HcEps { get; init; } = 1e-6f;

    // MoE & FFN parameters
    public int ExpertFeedForwardLength { get; init; } = 2048;
    public int IntermediateDim { get; init; } = 11008;
    public int LeadingDenseBlockCount { get; init; } = 3;
    public float ExpertWeightsScale { get; init; } = 2.5f;
    public bool ExpertWeightsNorm { get; init; } = true;
    public int ExpertSharedCount { get; init; } = 1;
    public int ExpertGatingFunc { get; init; } = 2; // Sigmoid
    public float SwiGluClampExp { get; init; } = 10.0f;

    public float RmsNormEps { get; init; } = 1e-5f;

    /// <summary>
    /// Strict 3:1 pattern: layers i % 4 != 3 are recurrent KDA; i % 4 == 3 are MLA/DSA.
    /// </summary>
    public bool IsRecurrent(int layer) => (layer % 4) != 3;

    public static Glm5NextHyperparams FromGgufMetadata(
        IReadOnlyDictionary<string, object> metadata, string arch, int numLayerAll,
        int embedDim, int numHeads, int headDim, int numExperts, int numExpertsUsed)
    {
        int numLayerNextn = GetInt(metadata, $"{arch}.nextn_predict_layers", 0);

        return new Glm5NextHyperparams
        {
            NumLayerAll = numLayerAll,
            NumLayerNextn = numLayerNextn,
            EmbedDim = embedDim,
            NumHeads = numHeads,
            HeadDimKda = GetInt(metadata, $"{arch}.kda.head_dim", headDim > 0 ? headDim : 128),
            SsmDConv = GetInt(metadata, $"{arch}.ssm.conv_kernel", 4),
            KdaGateLowerBound = GetFloat(metadata, $"{arch}.kda.gate_lower_bound", float.NegativeInfinity),
            QLoraRank = GetInt(metadata, $"{arch}.attention.q_lora_rank", 1536),
            KvLoraRank = GetInt(metadata, $"{arch}.attention.kv_lora_rank", 512),
            EmbedHeadKMla = GetInt(metadata, $"{arch}.attention.key_length_mla", 128),
            EmbedHeadVMla = GetInt(metadata, $"{arch}.attention.value_length_mla", 128),
            IndexerNumHeads = GetInt(metadata, $"{arch}.attention.indexer.head_count", 32),
            IndexerHeadSize = GetInt(metadata, $"{arch}.attention.indexer.key_length", 128),
            IndexerTopK = GetInt(metadata, $"{arch}.attention.indexer.top_k", 2048),
            IndexerKPool = GetInt(metadata, $"{arch}.attention.indexer.kpool", 4),
            IndexerKPoolSelectTail = GetBool(metadata, $"{arch}.attention.indexer.kpool_select_tail", true),
            HcMult = GetInt(metadata, $"{arch}.hyper_connection.count", 4),
            HcSinkhornIters = GetInt(metadata, $"{arch}.hyper_connection.sinkhorn_iterations", 20),
            HcEps = GetFloat(metadata, $"{arch}.hyper_connection.epsilon", 1e-6f),
            NumExperts = numExperts,
            NumExpertsUsed = numExpertsUsed,
            ExpertFeedForwardLength = GetInt(metadata, $"{arch}.expert_feed_forward_length", 2048),
            IntermediateDim = GetInt(metadata, $"{arch}.feed_forward_length", 11008),
            LeadingDenseBlockCount = GetInt(metadata, $"{arch}.leading_dense_block_count", 3),
            ExpertWeightsScale = GetFloat(metadata, $"{arch}.expert_weights_scale", 2.5f),
            ExpertWeightsNorm = GetBool(metadata, $"{arch}.expert_weights_norm", true),
            ExpertSharedCount = GetInt(metadata, $"{arch}.expert_shared_count", 1),
            ExpertGatingFunc = GetInt(metadata, $"{arch}.expert_gating_func", 2),
            SwiGluClampExp = GetFloat(metadata, $"{arch}.swiglu_clamp_exp", 10.0f),
            RmsNormEps = GetFloat(metadata, $"{arch}.attention.layer_norm_rms_epsilon", 1e-5f),
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
/// Resolved tensor references for one GLM5-Next decoder layer.
/// </summary>
public sealed unsafe class Glm5NextLayerTensors
{
    public DeepSeek4TensorRef? AttnNorm;
    public DeepSeek4TensorRef? FfnNorm;

    // mHC tensors
    public DeepSeek4TensorRef? HcAttnFn;
    public DeepSeek4TensorRef? HcAttnBase;
    public DeepSeek4TensorRef? HcAttnScale;
    public DeepSeek4TensorRef? HcFfnFn;
    public DeepSeek4TensorRef? HcFfnBase;
    public DeepSeek4TensorRef? HcFfnScale;

    // KDA sublayer tensors (is_recr == true)
    public DeepSeek4TensorRef? SsmQConv;
    public DeepSeek4TensorRef? SsmKConv;
    public DeepSeek4TensorRef? SsmVConv;
    public DeepSeek4TensorRef? Wq;
    public DeepSeek4TensorRef? Wk;
    public DeepSeek4TensorRef? Wv;
    public DeepSeek4TensorRef? SsmFA;
    public DeepSeek4TensorRef? SsmFB;
    public DeepSeek4TensorRef? SsmBeta;
    public DeepSeek4TensorRef? SsmA;
    public DeepSeek4TensorRef? SsmDtB;
    public DeepSeek4TensorRef? SsmGA;
    public DeepSeek4TensorRef? SsmGB;
    public DeepSeek4TensorRef? SsmONorm;
    public DeepSeek4TensorRef? Wo;

    // MLA / DSA sublayer tensors (is_recr == false)
    public DeepSeek4TensorRef? AttnQANorm;
    public DeepSeek4TensorRef? AttnKvANorm;
    public DeepSeek4TensorRef? WqA;
    public DeepSeek4TensorRef? WqB;
    public DeepSeek4TensorRef? WkvAMqa;
    public DeepSeek4TensorRef? WkB;
    public DeepSeek4TensorRef? WvB;
    public DeepSeek4TensorRef? IndexerKNorm;
    public DeepSeek4TensorRef? IndexerKNormBias;
    public DeepSeek4TensorRef? IndexerProj;
    public DeepSeek4TensorRef? IndexerAttnK;
    public DeepSeek4TensorRef? IndexerAttnQB;
    public DeepSeek4TensorRef? IndexerKPoolGate;
    public DeepSeek4TensorRef? IndexerKPoolApe;

    // Dense FFN (leading layers)
    public DeepSeek4TensorRef? FfnGate;
    public DeepSeek4TensorRef? FfnUp;
    public DeepSeek4TensorRef? FfnDown;

    // MoE FFN (remaining layers)
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
/// Resolves GLM5-Next tensors from a GGUF model container.
/// </summary>
public sealed unsafe class Glm5NextTensorSet
{
    public DeepSeek4TensorRef TokEmbd { get; }
    public DeepSeek4TensorRef OutputNorm { get; }
    public DeepSeek4TensorRef Output { get; }
    public IReadOnlyList<Glm5NextLayerTensors> Layers { get; }

    public Glm5NextTensorSet(DeepSeek4TensorRef tokEmbd, DeepSeek4TensorRef outputNorm, DeepSeek4TensorRef output, IReadOnlyList<Glm5NextLayerTensors> layers)
    {
        TokEmbd = tokEmbd;
        OutputNorm = outputNorm;
        Output = output;
        Layers = layers;
    }

    public static Glm5NextTensorSet Load(GgufModel model, Glm5NextHyperparams hp)
    {
        DeepSeek4TensorRef Required(string name)
        {
            var info = model.FindTensor(name)
                ?? throw new InvalidOperationException($"Missing required glm5next tensor: {name}");
            return new DeepSeek4TensorRef(name, info, model.GetTensorDataPtr(info));
        }

        DeepSeek4TensorRef? Optional(string name)
        {
            var info = model.FindTensor(name);
            return info is null ? null : new DeepSeek4TensorRef(name, info.Value, model.GetTensorDataPtr(info.Value));
        }

        var tokEmbd = Required("token_embd.weight");
        var outputNorm = Required("output_norm.weight");
        var output = Required("output.weight");

        var layers = new List<Glm5NextLayerTensors>(hp.NumLayer);
        for (int i = 0; i < hp.NumLayer; i++)
        {
            var lt = new Glm5NextLayerTensors
            {
                AttnNorm = Required($"blk.{i}.attn_norm.weight"),
                FfnNorm = Required($"blk.{i}.ffn_norm.weight"),
                HcAttnFn = Required($"blk.{i}.hc_attn_fn.weight"),
                HcAttnBase = Required($"blk.{i}.hc_attn_base.weight"),
                HcAttnScale = Required($"blk.{i}.hc_attn_scale.weight"),
                HcFfnFn = Required($"blk.{i}.hc_ffn_fn.weight"),
                HcFfnBase = Required($"blk.{i}.hc_ffn_base.weight"),
                HcFfnScale = Required($"blk.{i}.hc_ffn_scale.weight"),
            };

            if (hp.IsRecurrent(i))
            {
                lt.SsmQConv = Required($"blk.{i}.ssm_conv1d_q.weight");
                lt.SsmKConv = Required($"blk.{i}.ssm_conv1d_k.weight");
                lt.SsmVConv = Required($"blk.{i}.ssm_conv1d_v.weight");
                lt.Wq = Required($"blk.{i}.attn_q.weight");
                lt.Wk = Required($"blk.{i}.attn_k.weight");
                lt.Wv = Required($"blk.{i}.attn_v.weight");
                lt.SsmFA = Required($"blk.{i}.ssm_f_a.weight");
                lt.SsmFB = Required($"blk.{i}.ssm_f_b.weight");
                lt.SsmBeta = Required($"blk.{i}.ssm_beta.weight");
                lt.SsmA = Required($"blk.{i}.ssm_a.weight");
                lt.SsmDtB = Required($"blk.{i}.ssm_dt.bias");
                lt.SsmGA = Required($"blk.{i}.ssm_g_a.weight");
                lt.SsmGB = Required($"blk.{i}.ssm_g_b.weight");
                lt.SsmONorm = Required($"blk.{i}.ssm_norm.weight");
                lt.Wo = Optional($"blk.{i}.attn_output.weight") ?? Required($"blk.{i}.attn_out.weight");
            }
            else
            {
                lt.AttnQANorm = Required($"blk.{i}.attn_q_a_norm.weight");
                lt.AttnKvANorm = Required($"blk.{i}.attn_kv_a_norm.weight");
                lt.WqA = Required($"blk.{i}.attn_q_a.weight");
                lt.WqB = Required($"blk.{i}.attn_q_b.weight");
                lt.WkvAMqa = Required($"blk.{i}.attn_kv_a_mqa.weight");
                lt.WkB = Required($"blk.{i}.attn_k_b.weight");
                lt.WvB = Required($"blk.{i}.attn_v_b.weight");
                lt.Wo = Optional($"blk.{i}.attn_output.weight") ?? Required($"blk.{i}.attn_out.weight");
                lt.IndexerKNorm = Required($"blk.{i}.indexer_k_norm.weight");
                lt.IndexerKNormBias = Optional($"blk.{i}.indexer_k_norm.bias");
                lt.IndexerProj = Required($"blk.{i}.indexer_proj.weight");
                lt.IndexerAttnK = Required($"blk.{i}.indexer_attn_k.weight");
                lt.IndexerAttnQB = Required($"blk.{i}.indexer_attn_q_b.weight");
                lt.IndexerKPoolGate = Required($"blk.{i}.indexer_kpool_gate.weight");
                lt.IndexerKPoolApe = Required($"blk.{i}.indexer_kpool_ape.weight");
            }

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

        return new Glm5NextTensorSet(tokEmbd, outputNorm, output, layers);
    }
}
