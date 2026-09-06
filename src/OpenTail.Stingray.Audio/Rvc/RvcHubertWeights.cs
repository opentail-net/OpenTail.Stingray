
namespace OpenTail.Stingray.Audio.Rvc;

/// <summary>
/// Real weight loader for RVC's bundled HuBERT-base content encoder (`support_hubert_base/*` in
/// the packed `rvc-f16.gguf` checkpoint). Standard fairseq/HuggingFace Wav2Vec2-family
/// architecture (HuBERT reuses Wav2Vec2's encoder verbatim): a 7-stage 1D conv feature extractor
/// (GroupNorm after only the first conv layer -- `FirstLayerGroupNorm` convention), a feature
/// projection (LayerNorm + Linear 512-&gt;768), a weight-normalized positional conv
/// (`dim=2` weight norm, confirmed from real tensor shapes: `pos_conv.0.weight_g` is `[128]`
/// matching only the kernel dimension, not a per-output-channel norm), and 12 standard
/// post-LayerNorm transformer blocks (`self_attn`/`fc1`/`fc2`, matching
/// `examples/audio.cpp/src/models/rvc/hubert.cpp`'s real `rvc_hubert_config()`/
/// `rvc_hubert_binding()`).
///
/// <para>Real tensor names confirmed directly from `rvc-f16.gguf`'s own `audiocpp.tensor_names`
/// metadata array (the tensors themselves carry opaque `_audiocpp.NNNN` names --
/// `general.architecture=audiocpp`/`tensor_name_format=native` -- the real names only exist in
/// that metadata array, paired by index with `GgufModel.Tensors`; see
/// `RvcTensorNameDumpDebugTest`), not guessed from the C++ reference's config alone.</para>
///
/// <para><b>NOT YET implemented</b>: the actual forward pass (conv feature extraction, weight-norm
/// reconstruction application, transformer blocks). This class is weight-loading plumbing only,
/// the first concrete step of the RVC port -- see docs/audio-review-progress.md's RVC section.</para>
/// </summary>
public sealed class RvcHubertWeights
{
    public const int HiddenDim = 768;
    public const int IntermediateDim = 3072;
    public const int NumLayers = 12;
    public const int NumHeads = 12;
    public const int ConvPosKernel = 128;
    public const int ConvPosGroups = 16;
    public static readonly int[] ConvDim = [512, 512, 512, 512, 512, 512, 512];
    public static readonly int[] ConvKernel = [10, 3, 3, 3, 3, 2, 2];
    public static readonly int[] ConvStride = [5, 2, 2, 2, 2, 2, 2];

    // Feature extractor: 7 conv1d layers, GroupNorm only after layer 0.
    public float[][] ConvWeights { get; } = new float[7][]; // [outCh, inCh, kernel] per layer
    public float[] ConvLayer0GroupNormWeight { get; }
    public float[] ConvLayer0GroupNormBias { get; }

    // Feature projection.
    public float[] LayerNormWeight { get; }
    public float[] LayerNormBias { get; }
    public float[] PostExtractProjWeight { get; } // [768, 512]
    public float[] PostExtractProjBias { get; }

    // Positional conv (weight-normalized, dim=2 -- see class doc comment).
    public float[] PosConvBias { get; }
    public float[] PosConvWeightG { get; } // [128] (kernel-dim only)
    public float[] PosConvWeightV { get; } // [768, 48, 128] real shape (out, in_per_group, kernel)
    /// <summary>Real reconstructed positional conv weight: weight_v * (weight_g / ||weight_v||_dim2),
    /// computed once at load time so the forward pass never needs to redo the weight-norm math.</summary>
    public float[] PosConvWeight { get; }

    public float[] EncoderLayerNormWeight { get; }
    public float[] EncoderLayerNormBias { get; }

    public RvcHubertLayerWeights[] Layers { get; }

    private readonly RvcPackedTensorSource _source;

    public RvcHubertWeights(RvcPackedTensorSource source)
    {
        _source = source;
        for (int i = 0; i < 7; i++)
            ConvWeights[i] = GetTensor(source, $"support_hubert_base/feature_extractor.conv_layers.{i}.0.weight");
        ConvLayer0GroupNormWeight = GetTensor(source, "support_hubert_base/feature_extractor.conv_layers.0.2.weight");
        ConvLayer0GroupNormBias = GetTensor(source, "support_hubert_base/feature_extractor.conv_layers.0.2.bias");

        LayerNormWeight = GetTensor(source, "support_hubert_base/layer_norm.weight");
        LayerNormBias = GetTensor(source, "support_hubert_base/layer_norm.bias");
        PostExtractProjWeight = GetTensor(source, "support_hubert_base/post_extract_proj.weight");
        PostExtractProjBias = GetTensor(source, "support_hubert_base/post_extract_proj.bias");

        PosConvBias = GetTensor(source, "support_hubert_base/encoder.pos_conv.0.bias");
        PosConvWeightG = GetTensor(source, "support_hubert_base/encoder.pos_conv.0.weight_g");
        PosConvWeightV = GetTensor(source, "support_hubert_base/encoder.pos_conv.0.weight_v");
        PosConvWeight = ReconstructWeightNormDim2(PosConvWeightG, PosConvWeightV, outCh: HiddenDim, inPerGroup: HiddenDim / ConvPosGroups, kernel: ConvPosKernel);

        EncoderLayerNormWeight = GetTensor(source, "support_hubert_base/encoder.layer_norm.weight");
        EncoderLayerNormBias = GetTensor(source, "support_hubert_base/encoder.layer_norm.bias");

        Layers = new RvcHubertLayerWeights[NumLayers];
        for (int i = 0; i < NumLayers; i++)
            Layers[i] = new RvcHubertLayerWeights(source, i);
    }

    /// <summary>
    /// Real PyTorch `weight_norm(dim=2)` reconstruction: for weight shape [out, in_per_group,
    /// kernel], the norm is computed by reducing over dims 0 and 1 (everything EXCEPT dim 2), so
    /// `weight_g` has one scale value per kernel position. `weight = weight_v * (weight_g /
    /// ||weight_v[:,:,k]||)` for each kernel index k.
    /// </summary>
    private static float[] ReconstructWeightNormDim2(float[] weightG, float[] weightV, int outCh, int inPerGroup, int kernel)
    {
        var norms = new double[kernel];
        for (int k = 0; k < kernel; k++)
        {
            double sumSq = 0;
            for (int o = 0; o < outCh; o++)
                for (int ic = 0; ic < inPerGroup; ic++)
                {
                    float v = weightV[(o * inPerGroup + ic) * kernel + k];
                    sumSq += (double)v * v;
                }
            norms[k] = Math.Sqrt(sumSq);
        }

        var result = new float[weightV.Length];
        for (int o = 0; o < outCh; o++)
            for (int ic = 0; ic < inPerGroup; ic++)
                for (int k = 0; k < kernel; k++)
                {
                    int idx = (o * inPerGroup + ic) * kernel + k;
                    double scale = norms[k] > 1e-12 ? weightG[k] / norms[k] : 0.0;
                    result[idx] = (float)(weightV[idx] * scale);
                }
        return result;
    }

    internal static float[] GetTensor(RvcPackedTensorSource source, string name) => source.GetTensor(name);
}

/// <summary>One post-LayerNorm HuBERT/Wav2Vec2 transformer block: standard self-attn (with bias) + 2-layer FFN, both LayerNorm-after (PostNorm), matching `rvc_hubert_config().encoder_layer_norm_order = PostNorm`.</summary>
public sealed class RvcHubertLayerWeights
{
    public float[] AttnQWeight { get; }
    public float[] AttnQBias { get; }
    public float[] AttnKWeight { get; }
    public float[] AttnKBias { get; }
    public float[] AttnVWeight { get; }
    public float[] AttnVBias { get; }
    public float[] AttnOutWeight { get; }
    public float[] AttnOutBias { get; }
    public float[] SelfAttnLayerNormWeight { get; }
    public float[] SelfAttnLayerNormBias { get; }
    public float[] Fc1Weight { get; }
    public float[] Fc1Bias { get; }
    public float[] Fc2Weight { get; }
    public float[] Fc2Bias { get; }
    public float[] FinalLayerNormWeight { get; }
    public float[] FinalLayerNormBias { get; }

    public RvcHubertLayerWeights(RvcPackedTensorSource source, int i)
    {
        string p = $"support_hubert_base/encoder.layers.{i}";
        AttnQWeight = RvcHubertWeights.GetTensor(source, $"{p}.self_attn.q_proj.weight");
        AttnQBias = RvcHubertWeights.GetTensor(source, $"{p}.self_attn.q_proj.bias");
        AttnKWeight = RvcHubertWeights.GetTensor(source, $"{p}.self_attn.k_proj.weight");
        AttnKBias = RvcHubertWeights.GetTensor(source, $"{p}.self_attn.k_proj.bias");
        AttnVWeight = RvcHubertWeights.GetTensor(source, $"{p}.self_attn.v_proj.weight");
        AttnVBias = RvcHubertWeights.GetTensor(source, $"{p}.self_attn.v_proj.bias");
        AttnOutWeight = RvcHubertWeights.GetTensor(source, $"{p}.self_attn.out_proj.weight");
        AttnOutBias = RvcHubertWeights.GetTensor(source, $"{p}.self_attn.out_proj.bias");
        SelfAttnLayerNormWeight = RvcHubertWeights.GetTensor(source, $"{p}.self_attn_layer_norm.weight");
        SelfAttnLayerNormBias = RvcHubertWeights.GetTensor(source, $"{p}.self_attn_layer_norm.bias");
        Fc1Weight = RvcHubertWeights.GetTensor(source, $"{p}.fc1.weight");
        Fc1Bias = RvcHubertWeights.GetTensor(source, $"{p}.fc1.bias");
        Fc2Weight = RvcHubertWeights.GetTensor(source, $"{p}.fc2.weight");
        Fc2Bias = RvcHubertWeights.GetTensor(source, $"{p}.fc2.bias");
        FinalLayerNormWeight = RvcHubertWeights.GetTensor(source, $"{p}.final_layer_norm.weight");
        FinalLayerNormBias = RvcHubertWeights.GetTensor(source, $"{p}.final_layer_norm.bias");
    }
}
