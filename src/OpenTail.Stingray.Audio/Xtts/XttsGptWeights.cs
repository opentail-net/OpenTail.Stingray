
namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 GPT2 trunk weights (`gpt.gpt.*`), loaded from `model.safetensors` (converted from
/// the real `coqui/XTTS-v2` `model.pth` via `scratch-llamacpp-ref/xtts_convert_to_safetensors.py`).
///
/// <para>Confirmed against the real `coqui-ai-TTS` source
/// (`TTS/tts/layers/tortoise/autoregressive.py`'s `build_hf_gpt_transformer`): `gpt.gpt` is a
/// PLAIN, STANDARD HuggingFace `GPT2Model` (`transformers.GPT2Model`) -- its own token/positional
/// embeddings (`wte`/`wpe`) are deleted and handled externally by XTTS's own
/// `gpt.text_embedding`/`gpt.mel_embedding`/`gpt.*_pos_embedding`, but the `h.N.*` transformer
/// blocks themselves are 100% vanilla GPT2 decoder math -- no custom modification.</para>
///
/// <para><b>Critical, easy-to-get-backwards detail</b>: HF GPT2 uses `Conv1D`
/// (`transformers.pytorch_utils.Conv1D`), NOT `nn.Linear`, for `c_attn`/`c_proj`/`mlp.c_fc`/
/// `mlp.c_proj`. `Conv1D`'s real weight is stored `[in_features, out_features]` (the OPPOSITE of
/// `nn.Linear`'s `[out,in]`, and the opposite of every other pipeline's convention in this
/// codebase) -- confirmed directly from the real safetensors header (`c_attn.weight` shape
/// `(1024, 3072)` for a 1024-in/3072-out projection). <b>Transposed to this codebase's usual
/// `[out,in]` row-major layout HERE, at load time</b>, so every downstream matvec kernel
/// (<see cref="VitsAttentionKernels.Conv1x1"/> et al) can be used unmodified with the same
/// `weight[outIdx*inFeatures+inIdx]` indexing every other pipeline already assumes.</para>
/// </summary>
public sealed class XttsGptWeights
{
    public const int ModelDim = 1024;
    public const int NumHeads = 16;
    public const int HeadDim = ModelDim / NumHeads;
    public const int NumLayers = 30;
    public const int FfnDim = ModelDim * 4;

    public XttsGptLayerWeights[] Layers { get; } = new XttsGptLayerWeights[NumLayers];
    public float[] FinalNormWeight { get; }
    public float[] FinalNormBias { get; }

    public XttsGptWeights(SafetensorsLoader loader)
    {
        for (int i = 0; i < NumLayers; i++)
            Layers[i] = new XttsGptLayerWeights(loader, $"gpt.gpt.h.{i}");

        FinalNormWeight = loader.ReadF32("gpt.gpt.ln_f.weight");
        FinalNormBias = loader.ReadF32("gpt.gpt.ln_f.bias");
    }

    /// <summary>Reads a real HF `Conv1D` weight ([inFeatures,outFeatures] real storage order) and transposes it to this codebase's usual [outFeatures,inFeatures] row-major layout.</summary>
    internal static float[] ReadConv1DWeightTransposed(SafetensorsLoader loader, string name, out int inFeatures, out int outFeatures)
    {
        var raw = loader.ReadF32(name);
        int[] shape = loader.GetShape(name);
        inFeatures = shape[0];
        outFeatures = shape[1];

        var transposed = new float[raw.Length];
        for (int i = 0; i < inFeatures; i++)
            for (int o = 0; o < outFeatures; o++)
                transposed[o * inFeatures + i] = raw[i * outFeatures + o];
        return transposed;
    }
}

/// <summary>One GPT2 decoder block: ln_1 -> attn (c_attn fused QKV, causal self-attn, c_proj) -> +residual -> ln_2 -> mlp (c_fc, GELU("gelu_new"/tanh-approx, HF GPT2Config's real default activation_function -- not overridden), c_proj) -> +residual.</summary>
public sealed class XttsGptLayerWeights
{
    public float[] Ln1Weight { get; }
    public float[] Ln1Bias { get; }

    public float[] AttnCAttnWeight { get; } // [3*ModelDim, ModelDim] transposed, fused q/k/v
    public float[] AttnCAttnBias { get; }
    public float[] AttnCProjWeight { get; } // [ModelDim, ModelDim] transposed
    public float[] AttnCProjBias { get; }

    public float[] Ln2Weight { get; }
    public float[] Ln2Bias { get; }

    public float[] MlpCFcWeight { get; } // [FfnDim, ModelDim] transposed
    public float[] MlpCFcBias { get; }
    public float[] MlpCProjWeight { get; } // [ModelDim, FfnDim] transposed
    public float[] MlpCProjBias { get; }

    public XttsGptLayerWeights(SafetensorsLoader loader, string prefix)
    {
        Ln1Weight = loader.ReadF32($"{prefix}.ln_1.weight");
        Ln1Bias = loader.ReadF32($"{prefix}.ln_1.bias");

        AttnCAttnWeight = XttsGptWeights.ReadConv1DWeightTransposed(loader, $"{prefix}.attn.c_attn.weight", out _, out _);
        AttnCAttnBias = loader.ReadF32($"{prefix}.attn.c_attn.bias");
        AttnCProjWeight = XttsGptWeights.ReadConv1DWeightTransposed(loader, $"{prefix}.attn.c_proj.weight", out _, out _);
        AttnCProjBias = loader.ReadF32($"{prefix}.attn.c_proj.bias");

        Ln2Weight = loader.ReadF32($"{prefix}.ln_2.weight");
        Ln2Bias = loader.ReadF32($"{prefix}.ln_2.bias");

        MlpCFcWeight = XttsGptWeights.ReadConv1DWeightTransposed(loader, $"{prefix}.mlp.c_fc.weight", out _, out _);
        MlpCFcBias = loader.ReadF32($"{prefix}.mlp.c_fc.bias");
        MlpCProjWeight = XttsGptWeights.ReadConv1DWeightTransposed(loader, $"{prefix}.mlp.c_proj.weight", out _, out _);
        MlpCProjBias = loader.ReadF32($"{prefix}.mlp.c_proj.bias");
    }
}
