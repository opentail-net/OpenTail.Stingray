using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Diffusion.DiffusionGemma;

/// <summary>
/// Base class for resolved per-layer tensors in DiffusionGemma.
/// Gemma-4 heterogeneous MoE architecture: 25 Sliding Window Attention (SWA) layers and 5 Full Attention layers.
/// </summary>
public abstract unsafe class DiffusionGemmaLayerTensorsBase
{
    public required int LayerIndex { get; init; }
    public abstract bool IsFullAttention { get; }
    public abstract int QHeads { get; }
    public abstract int KvHeads { get; }
    public abstract int HeadDim { get; }
    public abstract int RopeDim { get; }

    // Attention projections and norms
    public required DeepSeek4TensorRef AttnNorm { get; init; }
    public required DeepSeek4TensorRef Wq { get; init; }
    public required DeepSeek4TensorRef Wk { get; init; }
    public abstract DeepSeek4TensorRef? Wv { get; }
    public required DeepSeek4TensorRef AttnQNorm { get; init; }
    public required DeepSeek4TensorRef AttnKNorm { get; init; }
    public required DeepSeek4TensorRef Wo { get; init; }
    public required DeepSeek4TensorRef PostAttnNorm { get; init; }

    // Dense FFN
    public required DeepSeek4TensorRef FfnNorm { get; init; }
    public required DeepSeek4TensorRef FfnGate { get; init; }
    public required DeepSeek4TensorRef FfnUp { get; init; }
    public required DeepSeek4TensorRef FfnDown { get; init; }
    public DeepSeek4TensorRef? PostFfwNorm1 { get; init; }

    // MoE Router and Experts (nullable for synthetic models without MoE)
    public DeepSeek4TensorRef? PreFfwNorm2 { get; init; }
    public DeepSeek4TensorRef? FfnGateInp { get; init; }
    public DeepSeek4TensorRef? FfnGateInpScale { get; init; }
    public DeepSeek4TensorRef? FfnGateUpExps { get; init; }
    public DeepSeek4TensorRef? FfnDownExps { get; init; }
    public DeepSeek4TensorRef? FfnDownExpsScale { get; init; }
    public DeepSeek4TensorRef? PostFfwNorm2 { get; init; }

    // Combined FFN post-norm and layer output scales
    public DeepSeek4TensorRef? PostFfwNorm { get; init; }
    public DeepSeek4TensorRef? LayerOutputScale { get; init; }
    public DeepSeek4TensorRef? EncLayerOutputScale { get; init; }

    public float LayerOutputScaleValue => LayerOutputScale is { } s ? DiffusionGemmaTensorSet.ReadScalarFloat(s) : 1.0f;
    public float EncLayerOutputScaleValue => EncLayerOutputScale is { } s ? DiffusionGemmaTensorSet.ReadScalarFloat(s) : 1.0f;

    public float GetExpertDownScale(int expert)
    {
        if (FfnDownExpsScale is not { } scale) return 1.0f;
        if (scale.DType == DType.Float32)
            return ((float*)scale.DataPtr)[expert];
        if (scale.DType == DType.Float16)
            return (float)((Half*)scale.DataPtr)[expert];
        float val = 0f;
        int blockSize = DTypeInfo.BytesPerBlock(scale.DType);
        byte* src = scale.DataPtr + (long)expert * blockSize;
        Dequantize.ToFloat32(new ReadOnlySpan<byte>(src, blockSize), new Span<float>(&val, 1), scale.DType, 1);
        return val;
    }
}

/// <summary>
/// Sliding-Window Attention (SWA) layer tensors.
/// Dedicated attn_v.weight is required.
/// </summary>
public sealed unsafe class DiffusionGemmaSlidingLayerTensors : DiffusionGemmaLayerTensorsBase
{
    private readonly DeepSeek4TensorRef _wv;

    public DiffusionGemmaSlidingLayerTensors(DeepSeek4TensorRef wv)
    {
        _wv = wv;
    }

    public override bool IsFullAttention => false;
    public override int QHeads => 16;
    public override int KvHeads => 8;
    public override int HeadDim => 256;
    public override int RopeDim => 256;
    public override DeepSeek4TensorRef? Wv => _wv;
}

/// <summary>
/// Full Attention layer tensors.
/// Dedicated attn_v.weight is absent (V is derived from raw K before RoPE).
/// </summary>
public sealed unsafe class DiffusionGemmaFullLayerTensors : DiffusionGemmaLayerTensorsBase
{
    public override bool IsFullAttention => true;
    public override int QHeads => 16;
    public override int KvHeads => 2;
    public override int HeadDim => 512;
    public override int RopeDim => 512;
    public override DeepSeek4TensorRef? Wv => null;
}

/// <summary>
/// Strict tensor loader for DiffusionGemma representing the 692-tensor GGUF contract.
/// </summary>
public sealed unsafe class DiffusionGemmaTensorSet
{
    public DeepSeek4TensorRef TokEmbd { get; }
    public DeepSeek4TensorRef OutputNorm { get; }
    public DeepSeek4TensorRef Output { get; }
    public DeepSeek4TensorRef RopeFreqs { get; }

    public DeepSeek4TensorRef SelfCondPreNorm { get; }
    public DeepSeek4TensorRef SelfCondGate { get; }
    public DeepSeek4TensorRef SelfCondUp { get; }
    public DeepSeek4TensorRef SelfCondDown { get; }

    public IReadOnlyList<DiffusionGemmaLayerTensorsBase> Layers { get; }

    public DiffusionGemmaTensorSet(
        DeepSeek4TensorRef tokEmbd,
        DeepSeek4TensorRef outputNorm,
        DeepSeek4TensorRef output,
        DeepSeek4TensorRef ropeFreqs,
        DeepSeek4TensorRef selfCondPreNorm,
        DeepSeek4TensorRef selfCondGate,
        DeepSeek4TensorRef selfCondUp,
        DeepSeek4TensorRef selfCondDown,
        IReadOnlyList<DiffusionGemmaLayerTensorsBase> layers)
    {
        TokEmbd = tokEmbd;
        OutputNorm = outputNorm;
        Output = output;
        RopeFreqs = ropeFreqs;
        SelfCondPreNorm = selfCondPreNorm;
        SelfCondGate = selfCondGate;
        SelfCondUp = selfCondUp;
        SelfCondDown = selfCondDown;
        Layers = layers;
    }

    public static DeepSeek4TensorRef Required(GgufModel model, string name)
    {
        var info = model.FindTensor(name) ?? model.FindTensor(name + ".weight");
        if (info is null)
        {
            throw new InvalidOperationException($"Missing required DiffusionGemma tensor: {name}");
        }
        return new DeepSeek4TensorRef(info.Value.Name, info.Value, model.GetTensorDataPtr(info.Value));
    }

    public static DeepSeek4TensorRef? Optional(GgufModel model, string name)
    {
        var info = model.FindTensor(name) ?? model.FindTensor(name + ".weight");
        if (info is null) return null;
        return new DeepSeek4TensorRef(info.Value.Name, info.Value, model.GetTensorDataPtr(info.Value));
    }

    public static float ReadScalarFloat(DeepSeek4TensorRef t)
    {
        if (t.DType == DType.Float32) return *(float*)t.DataPtr;
        if (t.DType == DType.Float16) return (float)*(Half*)t.DataPtr;
        float val = 0f;
        int blockSize = DTypeInfo.BytesPerBlock(t.DType);
        Dequantize.ToFloat32(new ReadOnlySpan<byte>(t.DataPtr, blockSize), new Span<float>(&val, 1), t.DType, 1);
        return val;
    }

    public static void ValidateShape(DeepSeek4TensorRef tensor, params long[] expected)
    {
        var dims = tensor.Info.Dimensions;
        if (dims.Length != expected.Length)
        {
            throw new InvalidOperationException(
                $"Tensor '{tensor.Name}' has {dims.Length} dimensions ([{string.Join(", ", dims)}]), expected {expected.Length} ([{string.Join(", ", expected)}]).");
        }
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] > 0 && (long)dims[i] != expected[i])
            {
                throw new InvalidOperationException(
                    $"Tensor '{tensor.Name}' dimension {i} is {dims[i]}, expected {expected[i]}.");
            }
        }
    }

    /// <summary>
    /// Loads and strictly validates all DiffusionGemma tensors from the given GGUF model.
    /// </summary>
    public static DiffusionGemmaTensorSet Load(GgufModel model, DiffusionGemmaConfig config)
    {
        var tokEmbd = Required(model, "token_embd.weight");
        ValidateShape(tokEmbd, config.HiddenDim, config.VocabSize);

        var outputNorm = Required(model, "output_norm.weight");
        ValidateShape(outputNorm, config.HiddenDim);

        var output = Optional(model, "output.weight") ?? tokEmbd;
        if (output.DataPtr != tokEmbd.DataPtr)
        {
            ValidateShape(output, config.HiddenDim, config.VocabSize);
        }

        var ropeFreqs = Required(model, "rope_freqs.weight");

        var selfCondPreNorm = Required(model, "self_cond_pre_norm.weight");
        ValidateShape(selfCondPreNorm, config.HiddenDim);

        var selfCondGate = Required(model, "self_cond_gate.weight");
        ValidateShape(selfCondGate, config.HiddenDim, config.SelfCondIntermediateDim);

        var selfCondUp = Required(model, "self_cond_up.weight");
        ValidateShape(selfCondUp, config.HiddenDim, config.SelfCondIntermediateDim);

        var selfCondDown = Required(model, "self_cond_down.weight");
        ValidateShape(selfCondDown, config.SelfCondIntermediateDim, config.HiddenDim);

        var layers = new List<DiffusionGemmaLayerTensorsBase>(config.NumLayers);
        for (int i = 0; i < config.NumLayers; i++)
        {
            bool isFull = config.IsFullAttention(i);

            int qDim = isFull ? config.FullNumQHeads * config.FullHeadDim : config.SlidingNumQHeads * config.SlidingHeadDim;
            int kvDim = isFull ? config.FullNumKvHeads * config.FullHeadDim : config.SlidingNumKvHeads * config.SlidingHeadDim;
            int headDim = isFull ? config.FullHeadDim : config.SlidingHeadDim;

            var attnNorm = Required(model, $"blk.{i}.attn_norm.weight");
            ValidateShape(attnNorm, config.HiddenDim);

            var attnQ = Required(model, $"blk.{i}.attn_q.weight");
            ValidateShape(attnQ, config.HiddenDim, qDim);

            var attnK = Required(model, $"blk.{i}.attn_k.weight");
            ValidateShape(attnK, config.HiddenDim, kvDim);

            var attnQNorm = Required(model, $"blk.{i}.attn_q_norm.weight");
            ValidateShape(attnQNorm, headDim);

            var attnKNorm = Required(model, $"blk.{i}.attn_k_norm.weight");
            ValidateShape(attnKNorm, headDim);

            var attnOutput = Required(model, $"blk.{i}.attn_output.weight");
            ValidateShape(attnOutput, qDim, config.HiddenDim);

            var postAttnNorm = Required(model, $"blk.{i}.post_attention_norm.weight");
            ValidateShape(postAttnNorm, config.HiddenDim);

            var ffnNorm = Required(model, $"blk.{i}.ffn_norm.weight");
            ValidateShape(ffnNorm, config.HiddenDim);

            var ffnGate = Required(model, $"blk.{i}.ffn_gate.weight");
            ValidateShape(ffnGate, config.HiddenDim, config.DenseIntermediateDim);

            var ffnUp = Required(model, $"blk.{i}.ffn_up.weight");
            ValidateShape(ffnUp, config.HiddenDim, config.DenseIntermediateDim);

            var ffnDown = Required(model, $"blk.{i}.ffn_down.weight");
            ValidateShape(ffnDown, config.DenseIntermediateDim, config.HiddenDim);

            var postFfwNorm1 = Required(model, $"blk.{i}.post_ffw_norm_1.weight");
            ValidateShape(postFfwNorm1, config.HiddenDim);

            var preFfwNorm2 = Required(model, $"blk.{i}.pre_ffw_norm_2.weight");
            ValidateShape(preFfwNorm2, config.HiddenDim);

            var ffnGateInp = Required(model, $"blk.{i}.ffn_gate_inp.weight");
            ValidateShape(ffnGateInp, config.HiddenDim, config.NumExperts);

            var ffnGateInpScale = Required(model, $"blk.{i}.ffn_gate_inp.scale");
            ValidateShape(ffnGateInpScale, config.HiddenDim);

            var ffnGateUpExps = Required(model, $"blk.{i}.ffn_gate_up_exps.weight");
            if (ffnGateUpExps.Info.NDimensions == 3)
            {
                ValidateShape(ffnGateUpExps, config.HiddenDim, config.ExpertIntermediateDim * 2, config.NumExperts);
            }
            else if (ffnGateUpExps.Info.NDimensions == 2)
            {
                ValidateShape(ffnGateUpExps, config.HiddenDim, (long)config.ExpertIntermediateDim * 2 * config.NumExperts);
            }
            else
            {
                throw new InvalidOperationException($"ffn_gate_up_exps.weight must be 2D or 3D, got {ffnGateUpExps.Info.NDimensions}D");
            }

            var ffnDownExps = Required(model, $"blk.{i}.ffn_down_exps.weight");
            if (ffnDownExps.Info.NDimensions == 3)
            {
                ValidateShape(ffnDownExps, config.ExpertIntermediateDim, config.HiddenDim, config.NumExperts);
            }
            else if (ffnDownExps.Info.NDimensions == 2)
            {
                ValidateShape(ffnDownExps, config.ExpertIntermediateDim, (long)config.HiddenDim * config.NumExperts);
            }
            else
            {
                throw new InvalidOperationException($"ffn_down_exps.weight must be 2D or 3D, got {ffnDownExps.Info.NDimensions}D");
            }

            var ffnDownExpsScale = Required(model, $"blk.{i}.ffn_down_exps.scale");
            ValidateShape(ffnDownExpsScale, config.NumExperts);

            var postFfwNorm2 = Required(model, $"blk.{i}.post_ffw_norm_2.weight");
            ValidateShape(postFfwNorm2, config.HiddenDim);

            var postFfwNorm = Required(model, $"blk.{i}.post_ffw_norm.weight");
            ValidateShape(postFfwNorm, config.HiddenDim);

            var layerOutputScale = Required(model, $"blk.{i}.layer_output_scale");
            ValidateShape(layerOutputScale, 1);

            var encLayerOutputScale = Required(model, $"blk.{i}.enc_layer_output_scale");
            ValidateShape(encLayerOutputScale, 1);

            DiffusionGemmaLayerTensorsBase layer;
            if (isFull)
            {
                // Full attention layers must NOT have attn_v.weight
                var vInfo = model.FindTensor($"blk.{i}.attn_v.weight") ?? model.FindTensor($"blk.{i}.attn_v");
                if (vInfo is not null)
                {
                    throw new InvalidOperationException(
                        $"Full attention layer {i} must NOT have attn_v.weight (V is derived from raw K before RoPE).");
                }

                layer = new DiffusionGemmaFullLayerTensors
                {
                    LayerIndex = i,
                    AttnNorm = attnNorm,
                    Wq = attnQ,
                    Wk = attnK,
                    AttnQNorm = attnQNorm,
                    AttnKNorm = attnKNorm,
                    Wo = attnOutput,
                    PostAttnNorm = postAttnNorm,
                    FfnNorm = ffnNorm,
                    FfnGate = ffnGate,
                    FfnUp = ffnUp,
                    FfnDown = ffnDown,
                    PostFfwNorm1 = postFfwNorm1,
                    PreFfwNorm2 = preFfwNorm2,
                    FfnGateInp = ffnGateInp,
                    FfnGateInpScale = ffnGateInpScale,
                    FfnGateUpExps = ffnGateUpExps,
                    FfnDownExps = ffnDownExps,
                    FfnDownExpsScale = ffnDownExpsScale,
                    PostFfwNorm2 = postFfwNorm2,
                    PostFfwNorm = postFfwNorm,
                    LayerOutputScale = layerOutputScale,
                    EncLayerOutputScale = encLayerOutputScale,
                };
            }
            else
            {
                // SWA layers require attn_v.weight
                var attnV = Required(model, $"blk.{i}.attn_v.weight");
                ValidateShape(attnV, config.HiddenDim, kvDim);

                layer = new DiffusionGemmaSlidingLayerTensors(attnV)
                {
                    LayerIndex = i,
                    AttnNorm = attnNorm,
                    Wq = attnQ,
                    Wk = attnK,
                    AttnQNorm = attnQNorm,
                    AttnKNorm = attnKNorm,
                    Wo = attnOutput,
                    PostAttnNorm = postAttnNorm,
                    FfnNorm = ffnNorm,
                    FfnGate = ffnGate,
                    FfnUp = ffnUp,
                    FfnDown = ffnDown,
                    PostFfwNorm1 = postFfwNorm1,
                    PreFfwNorm2 = preFfwNorm2,
                    FfnGateInp = ffnGateInp,
                    FfnGateInpScale = ffnGateInpScale,
                    FfnGateUpExps = ffnGateUpExps,
                    FfnDownExps = ffnDownExps,
                    FfnDownExpsScale = ffnDownExpsScale,
                    PostFfwNorm2 = postFfwNorm2,
                    PostFfwNorm = postFfwNorm,
                    LayerOutputScale = layerOutputScale,
                    EncLayerOutputScale = encLayerOutputScale,
                };
            }

            layers.Add(layer);
        }

        return new DiffusionGemmaTensorSet(
            tokEmbd, outputNorm, output, ropeFreqs,
            selfCondPreNorm, selfCondGate, selfCondUp, selfCondDown,
            layers);
    }
}
