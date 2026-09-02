using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine;

// ============================================================================================
// ALPHA / UNTESTED -- see DeepSeek4Alpha.cs's file header for the overall status/scope note;
// everything there applies here too. This file wires GGUF tensor NAME resolution (not tensor
// data verification -- nothing here has ever run against a real GGUF) for the deepseek4
// architecture, following deepseek4.cpp:79-178's load_arch_tensors exactly, using the real GGUF
// tensor-name strings confirmed by grepping llama-arch.cpp's LLM_TENSOR_* table directly (not
// assumed from naming convention -- e.g. LLM_TENSOR_ATTN_KV_NORM maps to the GGUF string
// "attn_kv_a_norm", not "attn_kv_norm" as its own enum name would suggest; ATTN_OUT_A/B map to
// "attn_output_a"/"attn_output_b", not "attn_out_a/b"; HC_HEAD_* tensors are NOT per-layer
// ("output_hc_fn", no blk.%d prefix) unlike every other HC_* tensor).
//
// This does NOT touch OpenTail.Stingray.Core.ModelGraph.cs or ForwardPass*.cs -- deliberately,
// per the same reasoning in DeepSeek4Alpha.cs's header: this codebase's tensor loading is done
// per-architecture, inline, directly against GgufModel by each ForwardPass class (confirmed by
// reading ForwardPass.cs's existing `_model.FindTensor($"blk.{i}.…")`/`ResolveTensor(…)` calls),
// not through a shared per-architecture tensor registry -- so a new architecture's tensor
// resolution can live in its own file without risking any existing architecture's loading path.
// ============================================================================================

/// <summary>
/// ALPHA/UNTESTED. A resolved GGUF tensor: name, shape/dtype descriptor, and a raw pointer into
/// the memory-mapped file (zero-copy, matching this codebase's existing `TensorRef` pattern in
/// <c>ForwardPass.Helpers.cs</c>/<c>HybridGdnForwardPass.cs</c> -- duplicated here rather than
/// reused because those are `private` to their respective classes).
/// </summary>
public readonly unsafe struct DeepSeek4TensorRef
{
    public readonly string Name;
    public readonly GgufTensorInfo Info;
    public readonly DType DType;
    public readonly byte* DataPtr;

    public DeepSeek4TensorRef(string name, GgufTensorInfo info, byte* dataPtr)
    {
        Name = name;
        Info = info;
        DType = info.DType;
        DataPtr = dataPtr;
    }
}

/// <summary>
/// ALPHA/UNTESTED. One deepseek4 trunk layer's tensors, per <c>load_arch_tensors</c>
/// (deepseek4.cpp:106-177). Every field is nullable: some are only present when this layer has a
/// non-zero <see cref="DeepSeek4Hyperparams.CompressRatios"/> entry (attn_comp_*/indexer_*), only
/// for hash-routed layers (ffn_gate_tid2eid vs. ffn_exp_probs_b -- mutually exclusive per
/// deepseek4.cpp:154-158), or only for the MTP tail layers (nextn.*, deepseek4.cpp:169-176).
/// <see cref="DeepSeek4TensorSet.LoadLayer"/> resolves what a given checkpoint actually declares
/// rather than assuming a fixed shape across every layer.
/// </summary>
public sealed unsafe class DeepSeek4LayerTensors
{
    // Always present.
    public DeepSeek4TensorRef? AttnNorm;
    public DeepSeek4TensorRef? AttnSinks;
    public DeepSeek4TensorRef? WqA;
    public DeepSeek4TensorRef? AttnQANorm;
    public DeepSeek4TensorRef? WqB;
    public DeepSeek4TensorRef? Wkv;
    public DeepSeek4TensorRef? AttnKvNorm;
    public DeepSeek4TensorRef? WoA;
    public DeepSeek4TensorRef? WoB;
    public DeepSeek4TensorRef? HcAttnFn;
    public DeepSeek4TensorRef? HcAttnBase;
    public DeepSeek4TensorRef? HcAttnScale;
    public DeepSeek4TensorRef? HcFfnFn;
    public DeepSeek4TensorRef? HcFfnBase;
    public DeepSeek4TensorRef? HcFfnScale;
    public DeepSeek4TensorRef? FfnGateInp;
    public DeepSeek4TensorRef? FfnNorm;
    public DeepSeek4TensorRef? FfnGateExps;
    public DeepSeek4TensorRef? FfnDownExps;
    public DeepSeek4TensorRef? FfnUpExps;
    public DeepSeek4TensorRef? FfnGateShexp;
    public DeepSeek4TensorRef? FfnDownShexp;
    public DeepSeek4TensorRef? FfnUpShexp;

    // Present only when this layer's compress ratio != 0 (deepseek4.cpp:129-151).
    public DeepSeek4TensorRef? AttnCompWkv;
    public DeepSeek4TensorRef? AttnCompWgate;
    public DeepSeek4TensorRef? AttnCompApe;
    public DeepSeek4TensorRef? AttnCompNorm;

    // Present only when this layer's compress ratio == 4 (CSA -- also gets the indexer, deepseek4.cpp:138-147).
    public DeepSeek4TensorRef? IndexerProj;
    public DeepSeek4TensorRef? IndexerAttnQB;
    public DeepSeek4TensorRef? IndexerCompWkv;
    public DeepSeek4TensorRef? IndexerCompWgate;
    public DeepSeek4TensorRef? IndexerCompApe;
    public DeepSeek4TensorRef? IndexerCompNorm;

    // Exactly one of these two is present, depending on whether this layer is hash-routed
    // (deepseek4.cpp:154-158).
    public DeepSeek4TensorRef? FfnGateTid2Eid;
    public DeepSeek4TensorRef? FfnExpProbsB;

    // Present only on MTP tail layers (deepseek4.cpp:169-176).
    public DeepSeek4TensorRef? NextnEhProj;
    public DeepSeek4TensorRef? NextnENorm;
    public DeepSeek4TensorRef? NextnHNorm;
    public DeepSeek4TensorRef? NextnEmbedTokens;
    public DeepSeek4TensorRef? NextnSharedHeadHead;
    public DeepSeek4TensorRef? NextnSharedHeadNorm;
}

/// <summary>
/// ALPHA/UNTESTED. Resolves every deepseek4 GGUF tensor by name against a real
/// <see cref="GgufModel"/>, mirroring <c>llama_model_deepseek4::load_arch_tensors</c>
/// (deepseek4.cpp:79-178) tensor-for-tensor. NOT yet exercised against a real GGUF -- the tensor
/// name strings were cross-checked against <c>llama-arch.cpp</c>'s name table directly, but only
/// a real file can confirm shapes/dtypes actually match what this class expects.
/// </summary>
public sealed unsafe class DeepSeek4TensorSet
{
    public DeepSeek4TensorRef TokEmbd { get; }
    public DeepSeek4TensorRef OutputNorm { get; }
    public DeepSeek4TensorRef Output { get; }
    public DeepSeek4TensorRef HcHeadFn { get; }
    public DeepSeek4TensorRef HcHeadBase { get; }
    public DeepSeek4TensorRef HcHeadScale { get; }
    public IReadOnlyList<DeepSeek4LayerTensors> Layers { get; }

    private DeepSeek4TensorSet(
        DeepSeek4TensorRef tokEmbd, DeepSeek4TensorRef outputNorm, DeepSeek4TensorRef output,
        DeepSeek4TensorRef hcHeadFn, DeepSeek4TensorRef hcHeadBase, DeepSeek4TensorRef hcHeadScale,
        IReadOnlyList<DeepSeek4LayerTensors> layers)
    {
        TokEmbd = tokEmbd;
        OutputNorm = outputNorm;
        Output = output;
        HcHeadFn = hcHeadFn;
        HcHeadBase = hcHeadBase;
        HcHeadScale = hcHeadScale;
        Layers = layers;
    }

    /// <summary>
    /// Loads every deepseek4 tensor from <paramref name="model"/> for <paramref name="hp"/>'s
    /// layer count and per-layer compress ratios / hash-layer count. Throws
    /// <see cref="InvalidOperationException"/> naming the missing tensor for anything
    /// unconditionally required (matching the reference's non-`TENSOR_NOT_REQUIRED` tensors);
    /// conditional tensors resolve to <c>null</c> when the checkpoint doesn't declare them,
    /// rather than throwing, since this port doesn't yet replicate the reference's
    /// mtp_only/trunk_only detection (deepseek4.cpp:93-95) that decides which subset a given
    /// GGUF is expected to carry.
    /// </summary>
    public static DeepSeek4TensorSet Load(GgufModel model, DeepSeek4Hyperparams hp)
    {
        DeepSeek4TensorRef Required(string name)
        {
            var info = model.FindTensor(name)
                ?? throw new InvalidOperationException($"Missing required deepseek4 tensor: {name}");
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
        // HC_HEAD_* tensors are the one HC_* family that is NOT per-layer -- llama-arch.cpp maps
        // them to "output_hc_fn"/"output_hc_base"/"output_hc_scale" with no blk.%d prefix, unlike
        // HC_ATTN_*/HC_FFN_* below.
        var hcHeadFn = Required("output_hc_fn.weight");
        var hcHeadBase = Required("output_hc_base.weight");
        var hcHeadScale = Required("output_hc_scale.weight");

        int numLayerAll = hp.NumLayerAll;
        var layers = new DeepSeek4LayerTensors[numLayerAll];
        for (int i = 0; i < numLayerAll; i++)
        {
            var layer = new DeepSeek4LayerTensors
            {
                AttnNorm = Optional($"blk.{i}.attn_norm.weight"),
                AttnSinks = Optional($"blk.{i}.attn_sinks.weight"),
                WqA = Optional($"blk.{i}.attn_q_a.weight"),
                AttnQANorm = Optional($"blk.{i}.attn_q_a_norm.weight"),
                WqB = Optional($"blk.{i}.attn_q_b.weight"),
                Wkv = Optional($"blk.{i}.attn_kv.weight"),
                AttnKvNorm = Optional($"blk.{i}.attn_kv_a_norm.weight"),
                WoA = Optional($"blk.{i}.attn_output_a.weight"),
                WoB = Optional($"blk.{i}.attn_output_b.weight"),
                HcAttnFn = Optional($"blk.{i}.hc_attn_fn.weight"),
                HcAttnBase = Optional($"blk.{i}.hc_attn_base.weight"),
                HcAttnScale = Optional($"blk.{i}.hc_attn_scale.weight"),
                HcFfnFn = Optional($"blk.{i}.hc_ffn_fn.weight"),
                HcFfnBase = Optional($"blk.{i}.hc_ffn_base.weight"),
                HcFfnScale = Optional($"blk.{i}.hc_ffn_scale.weight"),
                FfnGateInp = Optional($"blk.{i}.ffn_gate_inp.weight"),
                FfnNorm = Optional($"blk.{i}.ffn_norm.weight"),
                FfnGateExps = Optional($"blk.{i}.ffn_gate_exps.weight"),
                FfnDownExps = Optional($"blk.{i}.ffn_down_exps.weight"),
                FfnUpExps = Optional($"blk.{i}.ffn_up_exps.weight"),
                FfnGateShexp = Optional($"blk.{i}.ffn_gate_shexp.weight"),
                FfnDownShexp = Optional($"blk.{i}.ffn_down_shexp.weight"),
                FfnUpShexp = Optional($"blk.{i}.ffn_up_shexp.weight"),

                AttnCompWkv = Optional($"blk.{i}.attn_compressor_kv.weight"),
                AttnCompWgate = Optional($"blk.{i}.attn_compressor_gate.weight"),
                AttnCompApe = Optional($"blk.{i}.attn_compressor_ape.weight"),
                AttnCompNorm = Optional($"blk.{i}.attn_compressor_norm.weight"),

                IndexerProj = Optional($"blk.{i}.indexer.proj.weight"),
                IndexerAttnQB = Optional($"blk.{i}.indexer.attn_q_b.weight"),
                IndexerCompWkv = Optional($"blk.{i}.indexer_compressor_kv.weight"),
                IndexerCompWgate = Optional($"blk.{i}.indexer_compressor_gate.weight"),
                IndexerCompApe = Optional($"blk.{i}.indexer_compressor_ape.weight"),
                IndexerCompNorm = Optional($"blk.{i}.indexer_compressor_norm.weight"),

                FfnGateTid2Eid = Optional($"blk.{i}.ffn_gate_tid2eid.weight"),
                FfnExpProbsB = Optional($"blk.{i}.exp_probs_b.bias"),

                NextnEhProj = Optional($"blk.{i}.nextn.eh_proj.weight"),
                NextnENorm = Optional($"blk.{i}.nextn.enorm.weight"),
                NextnHNorm = Optional($"blk.{i}.nextn.hnorm.weight"),
                NextnEmbedTokens = Optional($"blk.{i}.nextn.embed_tokens.weight"),
                NextnSharedHeadHead = Optional($"blk.{i}.nextn.shared_head_head.weight"),
                NextnSharedHeadNorm = Optional($"blk.{i}.nextn.shared_head_norm.weight"),
            };
            layers[i] = layer;
        }

        return new DeepSeek4TensorSet(tokEmbd, outputNorm, output, hcHeadFn, hcHeadBase, hcHeadScale, layers);
    }
}
