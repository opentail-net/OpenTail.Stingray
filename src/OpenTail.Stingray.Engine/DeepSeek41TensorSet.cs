using System;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Tensor collection for DeepSeek-V4.1 (deepseek41) architecture.
/// Resolves layer-by-layer tensors, including MoE 384 routed experts, 8-group output LoRA,
/// compression ratio 0/1/2 weights, and dual Engram tables on layers 1 and 14.
/// </summary>
public sealed unsafe class DeepSeek41TensorSet
{
    public DeepSeek4TensorRef? TokenEmbd;
    public DeepSeek4TensorRef? OutputNorm;
    public DeepSeek4TensorRef? Output;
    public DeepSeek4TensorRef? OutputHcFn;
    public DeepSeek4TensorRef? OutputHcScale;
    public DeepSeek4TensorRef? OutputHcBase;

    public DeepSeek41LayerTensors[] Layers { get; }

    private DeepSeek41TensorSet(int numLayers)
    {
        Layers = new DeepSeek41LayerTensors[numLayers];
    }

    public static DeepSeek41TensorSet Load(GgufModel model, DeepSeek41Hyperparams hp)
    {
        var set = new DeepSeek41TensorSet(hp.NumLayerAll);

        set.TokenEmbd = Resolve(model, "token_embd.weight");
        set.OutputNorm = Resolve(model, "output_norm.weight");
        set.Output = Resolve(model, "output.weight") ?? set.TokenEmbd;

        set.OutputHcFn = Resolve(model, "output_hc_fn.weight");
        set.OutputHcScale = Resolve(model, "output_hc_scale.weight");
        set.OutputHcBase = Resolve(model, "output_hc_base.weight");

        for (int i = 0; i < hp.NumLayerAll; i++)
        {
            set.Layers[i] = LoadLayer(model, i, hp);
        }

        return set;
    }

    private static DeepSeek41LayerTensors LoadLayer(GgufModel model, int i, DeepSeek41Hyperparams hp)
    {
        var layer = new DeepSeek41LayerTensors();
        string pfx = $"blk.{i}.";

        layer.AttnNorm = Resolve(model, $"{pfx}attn_norm.weight");
        layer.AttnSinks = Resolve(model, $"{pfx}attn_sinks.weight");
        layer.WqA = Resolve(model, $"{pfx}attn_q_a.weight");
        layer.AttnQANorm = Resolve(model, $"{pfx}attn_q_a_norm.weight");
        layer.WqB = Resolve(model, $"{pfx}attn_q_b.weight");
        layer.Wkv = Resolve(model, $"{pfx}attn_kv_a.weight") ?? Resolve(model, $"{pfx}attn_kv.weight");
        layer.AttnKvNorm = Resolve(model, $"{pfx}attn_kv_a_norm.weight");
        layer.WoA = Resolve(model, $"{pfx}attn_output_a.weight") ?? Resolve(model, $"{pfx}attn_wo_a.weight");
        layer.WoB = Resolve(model, $"{pfx}attn_output_b.weight") ?? Resolve(model, $"{pfx}attn_wo_b.weight");

        layer.HcAttnFn = Resolve(model, $"{pfx}hc_attn_fn.weight");
        layer.HcAttnBase = Resolve(model, $"{pfx}hc_attn_base.weight");
        layer.HcAttnScale = Resolve(model, $"{pfx}hc_attn_scale.weight");

        layer.HcFfnFn = Resolve(model, $"{pfx}hc_ffn_fn.weight");
        layer.HcFfnBase = Resolve(model, $"{pfx}hc_ffn_base.weight");
        layer.HcFfnScale = Resolve(model, $"{pfx}hc_ffn_scale.weight");

        layer.FfnNorm = Resolve(model, $"{pfx}ffn_norm.weight");
        layer.FfnGateInp = Resolve(model, $"{pfx}ffn_gate_inp.weight");
        layer.FfnGateExps = Resolve(model, $"{pfx}ffn_gate_exps.weight");
        layer.FfnDownExps = Resolve(model, $"{pfx}ffn_down_exps.weight");
        layer.FfnUpExps = Resolve(model, $"{pfx}ffn_up_exps.weight");

        layer.FfnGateShexp = Resolve(model, $"{pfx}ffn_gate_shexp.weight");
        layer.FfnDownShexp = Resolve(model, $"{pfx}ffn_down_shexp.weight");
        layer.FfnUpShexp = Resolve(model, $"{pfx}ffn_up_shexp.weight");

        // Compressed KV tensors (for ratios 1 and 2)
        int ratio = i < hp.CompressRatios.Count ? hp.CompressRatios[i] : 0;
        if (ratio > 0)
        {
            layer.AttnCompWkv = Resolve(model, $"{pfx}attn_comp_wkv.weight");
            layer.AttnCompWgate = Resolve(model, $"{pfx}attn_comp_wgate.weight");
            layer.AttnCompApe = Resolve(model, $"{pfx}attn_comp_ape.weight");
            layer.AttnCompNorm = Resolve(model, $"{pfx}attn_comp_norm.weight");

            layer.IndexerProj = Resolve(model, $"{pfx}indexer_proj.weight");
            layer.IndexerAttnQB = Resolve(model, $"{pfx}indexer_attn_q_b.weight");
            layer.IndexerCompWkv = Resolve(model, $"{pfx}indexer_comp_wkv.weight");
            layer.IndexerCompWgate = Resolve(model, $"{pfx}indexer_comp_wgate.weight");
            layer.IndexerCompApe = Resolve(model, $"{pfx}indexer_comp_ape.weight");
            layer.IndexerCompNorm = Resolve(model, $"{pfx}indexer_comp_norm.weight");
        }

        // Engram tensors on layer 1 and 14
        if (i == 1 || i == 14)
        {
            layer.EngramTable = Resolve(model, $"{pfx}engram.weight") ?? Resolve(model, $"{pfx}engram_table.weight");
            layer.EngramProj = Resolve(model, $"{pfx}engram.proj.weight") ?? Resolve(model, $"{pfx}engram_proj.weight");
        }

        return layer;
    }

    private static DeepSeek4TensorRef? Resolve(GgufModel model, string name)
    {
        var tensor = model.FindTensor(name);
        if (!tensor.HasValue) return null;
        return new DeepSeek4TensorRef(name, tensor.Value, (byte*)model.GetTensorDataPtr(tensor.Value));
    }
}

/// <summary>
/// Layer tensors for DeepSeek-V4.1 including optional Engram weights.
/// </summary>
public sealed unsafe class DeepSeek41LayerTensors
{
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

    public DeepSeek4TensorRef? FfnNorm;
    public DeepSeek4TensorRef? FfnGateInp;
    public DeepSeek4TensorRef? FfnGateExps;
    public DeepSeek4TensorRef? FfnDownExps;
    public DeepSeek4TensorRef? FfnUpExps;
    public DeepSeek4TensorRef? FfnGateShexp;
    public DeepSeek4TensorRef? FfnDownShexp;
    public DeepSeek4TensorRef? FfnUpShexp;

    public DeepSeek4TensorRef? AttnCompWkv;
    public DeepSeek4TensorRef? AttnCompWgate;
    public DeepSeek4TensorRef? AttnCompApe;
    public DeepSeek4TensorRef? AttnCompNorm;

    public DeepSeek4TensorRef? IndexerProj;
    public DeepSeek4TensorRef? IndexerAttnQB;
    public DeepSeek4TensorRef? IndexerCompWkv;
    public DeepSeek4TensorRef? IndexerCompWgate;
    public DeepSeek4TensorRef? IndexerCompApe;
    public DeepSeek4TensorRef? IndexerCompNorm;

    // Engram tensors (present on layers 1 and 14)
    public DeepSeek4TensorRef? EngramTable;
    public DeepSeek4TensorRef? EngramProj;
}
