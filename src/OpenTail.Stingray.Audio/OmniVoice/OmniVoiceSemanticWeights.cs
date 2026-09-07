
namespace OpenTail.Stingray.Audio.OmniVoice;

/// <summary>
/// Real weight loader for OmniVoice's semantic audio tokenizer (`semantic_model.*` in
/// `audio_tokenizer/model.safetensors`), a standard HuggingFace `HubertModel` -- the same
/// fairseq/HuggingFace Wav2Vec2-family architecture already golden-verified in this codebase for
/// RVC (<see cref="Rvc.RvcHubertWeights"/>), confirmed via a real tensor dump to have the EXACT
/// same hyperparameters (`conv_dim=[512]*7`, `conv_kernel=[10,3,3,3,3,2,2]`,
/// `conv_stride=[5,2,2,2,2,2,2]`, `hidden_size=768`, 12 layers, 12 heads,
/// `num_conv_pos_embeddings=128`, `num_conv_pos_embedding_groups=16`). Two real naming/format
/// differences from RVC's checkpoint (not guessed, confirmed from the real tensor dump):
/// <list type="bullet">
/// <item>Layer-0's GroupNorm submodule is named `feature_extractor.conv_layers.0.layer_norm`
/// (HF's own `Wav2Vec2GroupNormConvLayer` class attribute is literally called `layer_norm` even
/// though it holds an `nn.GroupNorm` -- confirmed by `feat_extract_norm: "group"` in the real
/// `config.json`, so this is NOT a `do_stable_layer_norm` variant despite the attribute name).</item>
/// <item>The positional conv's weight-norm uses the NEWER PyTorch `parametrizations.weight.
/// {original0,original1}` naming (`original0`=g, `original1`=v) instead of RVC's older
/// `weight_g`/`weight_v` naming -- same underlying `dim=2` weight-norm math either way.</item>
/// </list>
/// This class is weight-loading plumbing only; the forward pass is expected to be directly
/// adaptable from <see cref="Rvc.RvcHubertEncoder"/> given the identical architecture -- not yet
/// written.
/// </summary>
public sealed class OmniVoiceSemanticWeights
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

    public float[][] ConvWeights { get; } = new float[7][];
    public float[] ConvLayer0GroupNormWeight { get; }
    public float[] ConvLayer0GroupNormBias { get; }

    public float[] FeatureProjLayerNormWeight { get; }
    public float[] FeatureProjLayerNormBias { get; }
    public float[] FeatureProjWeight { get; } // [768, 512]
    public float[] FeatureProjBias { get; }

    public float[] PosConvBias { get; }
    public float[] PosConvWeight { get; } // real reconstructed weight, dim=2 weight-norm

    public float[] EncoderLayerNormWeight { get; }
    public float[] EncoderLayerNormBias { get; }

    public OmniVoiceSemanticLayerWeights[] Layers { get; } = new OmniVoiceSemanticLayerWeights[NumLayers];

    public OmniVoiceSemanticWeights(SafetensorsLoader loader) : this(loader.ReadF32)
    {
    }

    /// <summary>
    /// Real, generalized constructor, added 2026-09-07 so this same class can be reused for
    /// Higgs Audio TTS's real semantic encoder -- confirmed BYTE-FOR-BYTE architecturally
    /// identical to OmniVoice's (same real HuBERT-base hyperparameters: `conv_dim=[512]*7`,
    /// `conv_kernel=[10,3,3,3,3,2,2]`, `conv_stride=[5,2,2,2,2,2,2]`, `hidden_size=768`, 12
    /// layers, `num_conv_pos_embeddings=128`/`groups=16`, and the SAME
    /// `parametrizations.weight.original0/1` positional-conv weight-norm naming/formula --
    /// verified directly against `higgs_audio_tts/codec.cpp`'s own `kSemantic*` constants and
    /// `load_hubert_semantic_model_weights`/`effective_semantic_pos_conv_weight`, not assumed).
    /// `get` receives the bare `semantic_model.*`-relative tensor name; the caller supplies
    /// whatever real prefix its own checkpoint format needs (OmniVoice: none, reads directly from
    /// its safetensors; Higgs: `"tied.embedding.modality_embeddings.0.model." + name`, its real
    /// packed-GGUF codec prefix).
    /// </summary>
    public OmniVoiceSemanticWeights(Func<string, float[]> get)
    {
        for (int i = 0; i < 7; i++)
            ConvWeights[i] = get($"semantic_model.feature_extractor.conv_layers.{i}.conv.weight");
        ConvLayer0GroupNormWeight = get("semantic_model.feature_extractor.conv_layers.0.layer_norm.weight");
        ConvLayer0GroupNormBias = get("semantic_model.feature_extractor.conv_layers.0.layer_norm.bias");

        FeatureProjLayerNormWeight = get("semantic_model.feature_projection.layer_norm.weight");
        FeatureProjLayerNormBias = get("semantic_model.feature_projection.layer_norm.bias");
        FeatureProjWeight = get("semantic_model.feature_projection.projection.weight");
        FeatureProjBias = get("semantic_model.feature_projection.projection.bias");

        PosConvBias = get("semantic_model.encoder.pos_conv_embed.conv.bias");
        var posConvG = get("semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0");
        var posConvV = get("semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1");
        PosConvWeight = ReconstructWeightNormDim2(posConvG, posConvV, outCh: HiddenDim, inPerGroup: HiddenDim / ConvPosGroups, kernel: ConvPosKernel);

        EncoderLayerNormWeight = get("semantic_model.encoder.layer_norm.weight");
        EncoderLayerNormBias = get("semantic_model.encoder.layer_norm.bias");

        for (int i = 0; i < NumLayers; i++)
            Layers[i] = new OmniVoiceSemanticLayerWeights(get, i);
    }

    /// <summary>Real PyTorch `weight_norm(dim=2)` reconstruction -- identical math to
    /// <see cref="Rvc.RvcHubertWeights"/>'s own reconstruction, just fed from the newer
    /// `parametrizations.weight.original0/1` tensor names instead of `weight_g`/`weight_v`.
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
}

/// <summary>One post-LayerNorm HuBERT/Wav2Vec2 transformer block -- identical shape to
/// <see cref="Rvc.RvcHubertLayerWeights"/>.</summary>
public sealed class OmniVoiceSemanticLayerWeights
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

    public OmniVoiceSemanticLayerWeights(SafetensorsLoader loader, int i) : this(loader.ReadF32, i)
    {
    }

    public OmniVoiceSemanticLayerWeights(Func<string, float[]> get, int i)
    {
        string p = $"semantic_model.encoder.layers.{i}";
        AttnQWeight = get($"{p}.attention.q_proj.weight");
        AttnQBias = get($"{p}.attention.q_proj.bias");
        AttnKWeight = get($"{p}.attention.k_proj.weight");
        AttnKBias = get($"{p}.attention.k_proj.bias");
        AttnVWeight = get($"{p}.attention.v_proj.weight");
        AttnVBias = get($"{p}.attention.v_proj.bias");
        AttnOutWeight = get($"{p}.attention.out_proj.weight");
        AttnOutBias = get($"{p}.attention.out_proj.bias");
        SelfAttnLayerNormWeight = get($"{p}.layer_norm.weight");
        SelfAttnLayerNormBias = get($"{p}.layer_norm.bias");
        Fc1Weight = get($"{p}.feed_forward.intermediate_dense.weight");
        Fc1Bias = get($"{p}.feed_forward.intermediate_dense.bias");
        Fc2Weight = get($"{p}.feed_forward.output_dense.weight");
        Fc2Bias = get($"{p}.feed_forward.output_dense.bias");
        FinalLayerNormWeight = get($"{p}.final_layer_norm.weight");
        FinalLayerNormBias = get($"{p}.final_layer_norm.bias");
    }
}
