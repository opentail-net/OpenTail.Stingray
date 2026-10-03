using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine;

// ============================================================================================
// ALPHA / UNTESTED -- Qwen 3.8 Flash Next ("qwen4exp") GGUF tensor resolution.
//
// Status as of 2026-10-03: maps exact GGUF tensor names from llama.cpp
// src/models/qwen4exp.cpp (lines 171-304) for all 48 layers (36 GDN + 12 QSA), 4-stream
// GatedResidual hyper-connections, PLE dilated conv & embeddings, and 512-expert MoE.
// ============================================================================================

/// <summary>
/// A resolved GGUF tensor reference with shape, data type, and pointer into the mapped file.
/// </summary>
public readonly unsafe struct Qwen4ExpTensorRef
{
    public readonly string Name;
    public readonly GgufTensorInfo Info;
    public readonly DType DType;
    public readonly byte* DataPtr;

    public Qwen4ExpTensorRef(string name, GgufTensorInfo info, byte* dataPtr)
    {
        Name = name;
        Info = info;
        DType = info.DType;
        DataPtr = dataPtr;
    }
}

/// <summary>
/// Holds all resolved tensor references for a single Qwen4Exp layer block.
/// Depending on whether the layer is recurrent (GDN) or full attention (QSA),
/// and whether it hosts the PLE block, appropriate fields are populated.
/// </summary>
public sealed unsafe class Qwen4ExpLayerTensors
{
    public int LayerIndex { get; init; }
    public bool IsRecurrent { get; init; }
    public bool IsPle { get; init; }

    // Hyper-Connections (present on every layer: mixer and MoE)
    public Qwen4ExpTensorRef? HcAttnNorm;
    public Qwen4ExpTensorRef? HcAttnDown;
    public Qwen4ExpTensorRef? HcAttnUp;
    public Qwen4ExpTensorRef? HcAttnInject;
    public Qwen4ExpTensorRef? HcFfnNorm;
    public Qwen4ExpTensorRef? HcFfnDown;
    public Qwen4ExpTensorRef? HcFfnUp;
    public Qwen4ExpTensorRef? HcFfnInject;

    // QSA (Full Attention) tensors (when !IsRecurrent)
    public Qwen4ExpTensorRef? AttnQ;
    public Qwen4ExpTensorRef? AttnK;
    public Qwen4ExpTensorRef? AttnV;
    public Qwen4ExpTensorRef? AttnOut;
    public Qwen4ExpTensorRef? AttnQNorm;
    public Qwen4ExpTensorRef? AttnKNorm;
    public Qwen4ExpTensorRef? IndexQProj;
    public Qwen4ExpTensorRef? IndexKProj;
    public Qwen4ExpTensorRef? IndexQNorm;
    public Qwen4ExpTensorRef? IndexKNorm;

    // GDN (Linear Attention) tensors (when IsRecurrent)
    public Qwen4ExpTensorRef? AttnQkv;
    public Qwen4ExpTensorRef? AttnGate;
    public Qwen4ExpTensorRef? SsmConv1d;
    public Qwen4ExpTensorRef? SsmDt;
    public Qwen4ExpTensorRef? SsmA;
    public Qwen4ExpTensorRef? SsmBeta;
    public Qwen4ExpTensorRef? SsmAlpha;
    public Qwen4ExpTensorRef? SsmNorm;
    public Qwen4ExpTensorRef? SsmOut;

    // PLE (Per-Layer Embedding) tensors (when IsPle)
    public Qwen4ExpTensorRef? PleKey;
    public Qwen4ExpTensorRef? PleValue;
    public Qwen4ExpTensorRef? PleNormKey;
    public Qwen4ExpTensorRef? PleNormQuery;
    public Qwen4ExpTensorRef? PleNormConv;
    public Qwen4ExpTensorRef? PleConv1d;

    // MoE tensors (present on all layers)
    public Qwen4ExpTensorRef? FfnGateInp;
    public Qwen4ExpTensorRef? FfnDownExps;
    public Qwen4ExpTensorRef? FfnGateExps;
    public Qwen4ExpTensorRef? FfnUpExps;
    public Qwen4ExpTensorRef? FfnGateUpExps;
    public Qwen4ExpTensorRef? FfnGateInpShexp;
    public Qwen4ExpTensorRef? FfnGateShexp;
    public Qwen4ExpTensorRef? FfnUpShexp;
    public Qwen4ExpTensorRef? FfnDownShexp;
}

/// <summary>
/// Resolves and holds the complete set of tensors for a Qwen4Exp model from a <see cref="GgufModel"/>.
/// </summary>
public sealed unsafe class Qwen4ExpTensorSet
{
    public Qwen4ExpTensorRef TokEmbd { get; }
    public Qwen4ExpTensorRef Output { get; }
    public Qwen4ExpTensorRef HcHeadNorm { get; }
    public Qwen4ExpTensorRef HcHeadDown { get; }
    public Qwen4ExpTensorRef HcHeadUp { get; }
    public Qwen4ExpTensorRef? PerLayerTokEmbd { get; }
    public IReadOnlyList<Qwen4ExpLayerTensors> Layers { get; }

    public Qwen4ExpTensorSet(
        Qwen4ExpTensorRef tokEmbd,
        Qwen4ExpTensorRef output,
        Qwen4ExpTensorRef hcHeadNorm,
        Qwen4ExpTensorRef hcHeadDown,
        Qwen4ExpTensorRef hcHeadUp,
        Qwen4ExpTensorRef? perLayerTokEmbd,
        IReadOnlyList<Qwen4ExpLayerTensors> layers)
    {
        TokEmbd = tokEmbd;
        Output = output;
        HcHeadNorm = hcHeadNorm;
        HcHeadDown = hcHeadDown;
        HcHeadUp = hcHeadUp;
        PerLayerTokEmbd = perLayerTokEmbd;
        Layers = layers;
    }

    /// <summary>
    /// Resolves all Qwen4Exp tensors from the given GGUF model according to the hyperparameters.
    /// </summary>
    public static Qwen4ExpTensorSet Load(GgufModel model, Qwen4ExpHyperparams hp, int numLayers)
    {
        var tokEmbd = Require(model, "token_embd.weight");
        var output = Find(model, "output.weight") ?? tokEmbd;
        var hcHeadNorm = Require(model, "hc_head_norm.weight");
        var hcHeadDown = Require(model, "hc_head_down.weight");
        var hcHeadUp = Require(model, "hc_head_up.weight");
        var perLayerTokEmbd = Find(model, "per_layer_token_embd.weight");

        var layers = new List<Qwen4ExpLayerTensors>(numLayers);
        for (int il = 0; il < numLayers; il++)
        {
            bool isRecr = hp.IsRecurrentLayer(il);
            bool isPle = hp.IsPleLayer(il);

            var layer = new Qwen4ExpLayerTensors
            {
                LayerIndex = il,
                IsRecurrent = isRecr,
                IsPle = isPle,

                // Hyper-connections
                HcAttnNorm = Find(model, $"blk.{il}.hc_attn_norm.weight"),
                HcAttnDown = Find(model, $"blk.{il}.hc_attn_down.weight"),
                HcAttnUp = Find(model, $"blk.{il}.hc_attn_up.weight"),
                HcAttnInject = Find(model, $"blk.{il}.hc_attn_inject.weight"),
                HcFfnNorm = Find(model, $"blk.{il}.hc_ffn_norm.weight"),
                HcFfnDown = Find(model, $"blk.{il}.hc_ffn_down.weight"),
                HcFfnUp = Find(model, $"blk.{il}.hc_ffn_up.weight"),
                HcFfnInject = Find(model, $"blk.{il}.hc_ffn_inject.weight"),

                // MoE
                FfnGateInp = Find(model, $"blk.{il}.ffn_gate_inp.weight"),
                FfnDownExps = Find(model, $"blk.{il}.ffn_down_exps.weight"),
                FfnGateExps = Find(model, $"blk.{il}.ffn_gate_exps.weight"),
                FfnUpExps = Find(model, $"blk.{il}.ffn_up_exps.weight"),
                FfnGateUpExps = Find(model, $"blk.{il}.ffn_gate_up_exps.weight"),
                FfnGateInpShexp = Find(model, $"blk.{il}.ffn_gate_inp_shexp.weight"),
                FfnGateShexp = Find(model, $"blk.{il}.ffn_gate_shexp.weight"),
                FfnUpShexp = Find(model, $"blk.{il}.ffn_up_shexp.weight"),
                FfnDownShexp = Find(model, $"blk.{il}.ffn_down_shexp.weight"),
            };

            if (isRecr)
            {
                // GDN (Linear Attention)
                layer.AttnQkv = Find(model, $"blk.{il}.attn_qkv.weight");
                layer.AttnGate = Find(model, $"blk.{il}.attn_gate.weight");
                layer.SsmConv1d = Find(model, $"blk.{il}.ssm_conv1d.weight");
                layer.SsmDt = Find(model, $"blk.{il}.ssm_dt.bias");
                layer.SsmA = Find(model, $"blk.{il}.ssm_a");
                layer.SsmBeta = Find(model, $"blk.{il}.ssm_beta.weight");
                layer.SsmAlpha = Find(model, $"blk.{il}.ssm_alpha.weight");
                layer.SsmNorm = Find(model, $"blk.{il}.ssm_norm.weight");
                layer.SsmOut = Find(model, $"blk.{il}.ssm_out.weight");
            }
            else
            {
                // QSA (Full Attention)
                layer.AttnQ = Find(model, $"blk.{il}.attn_q.weight");
                layer.AttnK = Find(model, $"blk.{il}.attn_k.weight");
                layer.AttnV = Find(model, $"blk.{il}.attn_v.weight");
                layer.AttnOut = Find(model, $"blk.{il}.attn_output.weight");
                layer.AttnQNorm = Find(model, $"blk.{il}.attn_q_norm.weight");
                layer.AttnKNorm = Find(model, $"blk.{il}.attn_k_norm.weight");
                layer.IndexQProj = Find(model, $"blk.{il}.index_q_proj.weight");
                layer.IndexKProj = Find(model, $"blk.{il}.index_k_proj.weight");
                layer.IndexQNorm = Find(model, $"blk.{il}.index_q_norm.weight");
                layer.IndexKNorm = Find(model, $"blk.{il}.index_k_norm.weight");
            }

            if (isPle)
            {
                // PLE
                layer.PleKey = Find(model, $"blk.{il}.ple_key.weight");
                layer.PleValue = Find(model, $"blk.{il}.ple_value.weight");
                layer.PleNormKey = Find(model, $"blk.{il}.ple_norm_key.weight");
                layer.PleNormQuery = Find(model, $"blk.{il}.ple_norm_query.weight");
                layer.PleNormConv = Find(model, $"blk.{il}.ple_norm_conv.weight");
                layer.PleConv1d = Find(model, $"blk.{il}.ple_conv1d.weight");
            }

            layers.Add(layer);
        }

        return new Qwen4ExpTensorSet(
            tokEmbd, output, hcHeadNorm, hcHeadDown, hcHeadUp, perLayerTokEmbd, layers);
    }

    private static Qwen4ExpTensorRef Require(GgufModel model, string name)
    {
        var tensor = Find(model, name);
        if (tensor == null)
        {
            throw new InvalidOperationException($"Required Qwen4Exp tensor '{name}' not found in GGUF model.");
        }
        return tensor.Value;
    }

    private static Qwen4ExpTensorRef? Find(GgufModel model, string name)
    {
        var info = model.FindTensor(name);
        if (!info.HasValue) return null;
        byte* ptr = model.GetTensorDataPtr(info.Value);
        return new Qwen4ExpTensorRef(name, info.Value, ptr);
    }
}
