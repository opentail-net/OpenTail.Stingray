using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine;

// ============================================================================================
// ALPHA / UNTESTED -- see GptOssAlpha.cs's file header. Tensor resolution for gpt-oss, mirroring
// openai-moe.cpp:24-58's load_arch_tensors, using GGUF tensor-name strings confirmed by grepping
// llama-arch.cpp's LLM_TENSOR_* table directly. Reuses DeepSeek4TensorRef as the resolved-tensor
// wrapper (structurally identical need across every architecture this project has ported this
// way so far -- no reason for yet another duplicate type).
// ============================================================================================

/// <summary>
/// ALPHA/UNTESTED. One gpt-oss layer's tensors, per <c>load_arch_tensors</c> (openai-moe.cpp:
/// 35-57). Every attention and MoE tensor has a bias counterpart -- gpt-oss is unusual among
/// this codebase's MoE architectures in biasing every one of them, including per-expert.
/// </summary>
public sealed unsafe class GptOssLayerTensors
{
    public DeepSeek4TensorRef? AttnNorm;
    public DeepSeek4TensorRef? AttnPostNorm;
    public DeepSeek4TensorRef? Wq, WqB;
    public DeepSeek4TensorRef? Wk, WkB;
    public DeepSeek4TensorRef? Wv, WvB;
    public DeepSeek4TensorRef? Wo, WoB;
    public DeepSeek4TensorRef? AttnSinks;

    public DeepSeek4TensorRef? FfnGateInp, FfnGateInpB;
    public DeepSeek4TensorRef? FfnGateExps, FfnGateExpsB;
    public DeepSeek4TensorRef? FfnDownExps, FfnDownExpsB;
    public DeepSeek4TensorRef? FfnUpExps, FfnUpExpsB;
}

/// <summary>
/// ALPHA/UNTESTED. Resolves every gpt-oss GGUF tensor by name against a real
/// <see cref="GgufModel"/>.
/// </summary>
public sealed unsafe class GptOssTensorSet
{
    public DeepSeek4TensorRef TokEmbd { get; }
    public DeepSeek4TensorRef OutputNorm { get; }
    public DeepSeek4TensorRef Output { get; }
    public IReadOnlyList<GptOssLayerTensors> Layers { get; }

    private GptOssTensorSet(DeepSeek4TensorRef tokEmbd, DeepSeek4TensorRef outputNorm, DeepSeek4TensorRef output, IReadOnlyList<GptOssLayerTensors> layers)
    {
        TokEmbd = tokEmbd;
        OutputNorm = outputNorm;
        Output = output;
        Layers = layers;
    }

    public static GptOssTensorSet Load(GgufModel model, GptOssHyperparams hp)
    {
        DeepSeek4TensorRef Required(string name)
        {
            var info = model.FindTensor(name)
                ?? throw new InvalidOperationException($"Missing required gpt-oss tensor: {name}");
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

        var layers = new GptOssLayerTensors[hp.NumLayer];
        for (int i = 0; i < hp.NumLayer; i++)
        {
            layers[i] = new GptOssLayerTensors
            {
                AttnNorm = Optional($"blk.{i}.attn_norm.weight"),
                AttnPostNorm = Optional($"blk.{i}.post_attention_norm.weight"),
                Wq = Optional($"blk.{i}.attn_q.weight"),
                WqB = Optional($"blk.{i}.attn_q.bias"),
                Wk = Optional($"blk.{i}.attn_k.weight"),
                WkB = Optional($"blk.{i}.attn_k.bias"),
                Wv = Optional($"blk.{i}.attn_v.weight"),
                WvB = Optional($"blk.{i}.attn_v.bias"),
                Wo = Optional($"blk.{i}.attn_output.weight"),
                WoB = Optional($"blk.{i}.attn_output.bias"),
                AttnSinks = Optional($"blk.{i}.attn_sinks.weight"),

                FfnGateInp = Optional($"blk.{i}.ffn_gate_inp.weight"),
                FfnGateInpB = Optional($"blk.{i}.ffn_gate_inp.bias"),
                FfnGateExps = Optional($"blk.{i}.ffn_gate_exps.weight"),
                FfnGateExpsB = Optional($"blk.{i}.ffn_gate_exps.bias"),
                FfnDownExps = Optional($"blk.{i}.ffn_down_exps.weight"),
                FfnDownExpsB = Optional($"blk.{i}.ffn_down_exps.bias"),
                FfnUpExps = Optional($"blk.{i}.ffn_up_exps.weight"),
                FfnUpExpsB = Optional($"blk.{i}.ffn_up_exps.bias"),
            };
        }

        return new GptOssTensorSet(tokEmbd, outputNorm, output, layers);
    }
}
