using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine;

// ============================================================================================
// ALPHA / UNTESTED -- see DeepSeek32Alpha.cs's file header. Tensor resolution for deepseek32,
// mirroring deepseek32.cpp:52-159's load_arch_tensors, using GGUF tensor-name strings confirmed
// by grepping llama-arch.cpp's LLM_TENSOR_* table directly. Same "own file, doesn't touch
// ModelGraph.cs/ForwardPass*.cs" pattern as DeepSeek4TensorSet.cs, for the same reason.
// ============================================================================================

/// <summary>
/// ALPHA/UNTESTED. One deepseek32 trunk layer's tensors, per <c>load_arch_tensors</c>
/// (deepseek32.cpp:93-158). <c>FfnGate</c>/<c>FfnUp</c>/<c>FfnDown</c> are populated for dense
/// (leading) layers; <c>FfnGateInp</c>/<c>*Exps</c>/<c>*Shexp</c> for MoE layers -- mutually
/// exclusive per <c>i &lt; hparams.n_layer_dense_lead</c> (deepseek32.cpp:121).
/// </summary>
public sealed unsafe class DeepSeek32LayerTensors
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

    public DeepSeek4TensorRef? IndexerKNorm;
    public DeepSeek4TensorRef? IndexerKNormBias;
    public DeepSeek4TensorRef? IndexerProj;
    public DeepSeek4TensorRef? IndexerAttnK;
    public DeepSeek4TensorRef? IndexerAttnQB;

    // Dense (leading) layers only.
    public DeepSeek4TensorRef? FfnGate;
    public DeepSeek4TensorRef? FfnUp;
    public DeepSeek4TensorRef? FfnDown;

    // MoE layers only.
    public DeepSeek4TensorRef? FfnGateInp;
    public DeepSeek4TensorRef? FfnExpProbsB;
    public DeepSeek4TensorRef? FfnGateExps;
    public DeepSeek4TensorRef? FfnDownExps;
    public DeepSeek4TensorRef? FfnUpExps;
    public DeepSeek4TensorRef? FfnGateShexp;
    public DeepSeek4TensorRef? FfnDownShexp;
    public DeepSeek4TensorRef? FfnUpShexp;

    // MTP tail layers only.
    public DeepSeek4TensorRef? NextnEhProj;
    public DeepSeek4TensorRef? NextnENorm;
    public DeepSeek4TensorRef? NextnHNorm;
    public DeepSeek4TensorRef? NextnEmbedTokens;
    public DeepSeek4TensorRef? NextnSharedHeadHead;
    public DeepSeek4TensorRef? NextnSharedHeadNorm;
}

/// <summary>
/// ALPHA/UNTESTED. Resolves every deepseek32 GGUF tensor by name against a real
/// <see cref="GgufModel"/>. Reuses <see cref="DeepSeek4TensorRef"/> as the resolved-tensor
/// wrapper type (structurally identical need, no reason for a second type).
/// </summary>
public sealed unsafe class DeepSeek32TensorSet
{
    public DeepSeek4TensorRef TokEmbd { get; }
    public DeepSeek4TensorRef OutputNorm { get; }
    public DeepSeek4TensorRef Output { get; }
    public IReadOnlyList<DeepSeek32LayerTensors> Layers { get; }

    private DeepSeek32TensorSet(DeepSeek4TensorRef tokEmbd, DeepSeek4TensorRef outputNorm, DeepSeek4TensorRef output, IReadOnlyList<DeepSeek32LayerTensors> layers)
    {
        TokEmbd = tokEmbd;
        OutputNorm = outputNorm;
        Output = output;
        Layers = layers;
    }

    public static DeepSeek32TensorSet Load(GgufModel model, DeepSeek32Hyperparams hp)
    {
        DeepSeek4TensorRef Required(string name)
        {
            var info = model.FindTensor(name)
                ?? throw new InvalidOperationException($"Missing required deepseek32 tensor: {name}");
            return new DeepSeek4TensorRef(name, info, model.GetTensorDataPtr(info));
        }

        DeepSeek4TensorRef? Optional(string name)
        {
            var info = model.FindTensor(name);
            return info is null ? null : new DeepSeek4TensorRef(name, info.Value, model.GetTensorDataPtr(info.Value));
        }

        var tokEmbd = Required("token_embd.weight");
        var outputNorm = Required("output_norm.weight");
        // deepseek32.cpp:87-91: output.weight falls back to tied token_embd if absent.
        var output = model.FindTensor("output.weight") is { } outInfo
            ? new DeepSeek4TensorRef("output.weight", outInfo, model.GetTensorDataPtr(outInfo))
            : tokEmbd;

        int numLayerAll = hp.NumLayerAll;
        var layers = new DeepSeek32LayerTensors[numLayerAll];
        for (int i = 0; i < numLayerAll; i++)
        {
            bool isDense = i < hp.LeadingDenseBlockCount;
            var layer = new DeepSeek32LayerTensors
            {
                AttnNorm = Optional($"blk.{i}.attn_norm.weight"),
                AttnQANorm = Optional($"blk.{i}.attn_q_a_norm.weight"),
                AttnKvANorm = Optional($"blk.{i}.attn_kv_a_norm.weight"),
                WqA = Optional($"blk.{i}.attn_q_a.weight"),
                WqB = Optional($"blk.{i}.attn_q_b.weight"),
                WkvAMqa = Optional($"blk.{i}.attn_kv_a_mqa.weight"),
                WkB = Optional($"blk.{i}.attn_k_b.weight"),
                WvB = Optional($"blk.{i}.attn_v_b.weight"),
                Wo = Optional($"blk.{i}.attn_output.weight"),
                FfnNorm = Optional($"blk.{i}.ffn_norm.weight"),

                IndexerKNorm = Optional($"blk.{i}.indexer.k_norm.weight"),
                IndexerKNormBias = Optional($"blk.{i}.indexer.k_norm.bias"),
                IndexerProj = Optional($"blk.{i}.indexer.proj.weight"),
                IndexerAttnK = Optional($"blk.{i}.indexer.attn_k.weight"),
                IndexerAttnQB = Optional($"blk.{i}.indexer.attn_q_b.weight"),

                FfnGate = isDense ? Optional($"blk.{i}.ffn_gate.weight") : null,
                FfnUp = isDense ? Optional($"blk.{i}.ffn_up.weight") : null,
                FfnDown = isDense ? Optional($"blk.{i}.ffn_down.weight") : null,

                FfnGateInp = !isDense ? Optional($"blk.{i}.ffn_gate_inp.weight") : null,
                FfnExpProbsB = !isDense ? Optional($"blk.{i}.exp_probs_b.bias") : null,
                FfnGateExps = !isDense ? Optional($"blk.{i}.ffn_gate_exps.weight") : null,
                FfnDownExps = !isDense ? Optional($"blk.{i}.ffn_down_exps.weight") : null,
                FfnUpExps = !isDense ? Optional($"blk.{i}.ffn_up_exps.weight") : null,
                FfnGateShexp = !isDense ? Optional($"blk.{i}.ffn_gate_shexp.weight") : null,
                FfnDownShexp = !isDense ? Optional($"blk.{i}.ffn_down_shexp.weight") : null,
                FfnUpShexp = !isDense ? Optional($"blk.{i}.ffn_up_shexp.weight") : null,

                NextnEhProj = Optional($"blk.{i}.nextn.eh_proj.weight"),
                NextnENorm = Optional($"blk.{i}.nextn.enorm.weight"),
                NextnHNorm = Optional($"blk.{i}.nextn.hnorm.weight"),
                NextnEmbedTokens = Optional($"blk.{i}.nextn.embed_tokens.weight"),
                NextnSharedHeadHead = Optional($"blk.{i}.nextn.shared_head_head.weight"),
                NextnSharedHeadNorm = Optional($"blk.{i}.nextn.shared_head_norm.weight"),
            };
            layers[i] = layer;
        }

        return new DeepSeek32TensorSet(tokEmbd, outputNorm, output, layers);
    }
}
