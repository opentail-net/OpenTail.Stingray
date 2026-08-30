
namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 speaker/style conditioning weights: `gpt.conditioning_encoder.*` (a real
/// `ConditioningEncoder`, `TTS/tts/layers/tortoise/autoregressive.py`) feeding
/// `gpt.conditioning_perceiver.*` (a real `PerceiverResampler`,
/// `TTS/tts/layers/xtts/perceiver_encoder.py`) -- confirmed from the actual source, not guessed:
/// mel-spectrogram (80-dim) reference audio -&gt; conv1x1 init -&gt; 6x self-attention `AttentionBlock`
/// (real construction: `ConditioningEncoder(80, model_dim=1024, num_attn_heads=16)`, `attn_blocks`
/// defaults to 6, NOT overridden) -&gt; Perceiver Resampler (32 learned latents cross-attending to
/// [latents ++ conditioning-encoder-output], `depth=2, dim_head=64, heads=8, ff_mult=4`) -&gt;
/// `[32, 1024]` conditioning sequence prefixed onto the GPT2's real input.
/// </summary>
public sealed class XttsConditioningWeights
{
    public const int MelDim = 80;
    public const int ModelDim = XttsGptWeights.ModelDim; // 1024
    public const int EncoderAttnBlocks = 6;
    public const int EncoderHeads = 16;

    public const int PerceiverDepth = 2;
    public const int PerceiverNumLatents = 32;
    public const int PerceiverDimHead = 64;
    public const int PerceiverHeads = 8;
    public const int PerceiverFfnInner = 2730; // int(ModelDim * 4 * 2 / 3), real FeedForward's dim_inner

    public float[] EncoderInitWeight { get; } // conv1x1, 80 -> 1024
    public float[] EncoderInitBias { get; }
    public XttsAttentionBlockWeights[] EncoderBlocks { get; } = new XttsAttentionBlockWeights[EncoderAttnBlocks];

    public float[] PerceiverLatents { get; } // [32, 1024] learned latent seed
    public XttsPerceiverLayerWeights[] PerceiverLayers { get; } = new XttsPerceiverLayerWeights[PerceiverDepth];
    public float[] PerceiverNormGamma { get; } // final RMSNorm

    public XttsConditioningWeights(SafetensorsLoader loader)
    {
        EncoderInitWeight = loader.ReadF32("gpt.conditioning_encoder.init.weight");
        EncoderInitBias = loader.ReadF32("gpt.conditioning_encoder.init.bias");
        for (int i = 0; i < EncoderAttnBlocks; i++)
            EncoderBlocks[i] = new XttsAttentionBlockWeights(loader, $"gpt.conditioning_encoder.attn.{i}");

        PerceiverLatents = loader.ReadF32("gpt.conditioning_perceiver.latents");
        for (int i = 0; i < PerceiverDepth; i++)
            PerceiverLayers[i] = new XttsPerceiverLayerWeights(loader, $"gpt.conditioning_perceiver.layers.{i}");
        PerceiverNormGamma = loader.ReadF32("gpt.conditioning_perceiver.norm.gamma");
    }
}

/// <summary>Real `AttentionBlock` (GroupNorm(32 groups) -> qkv conv1x1 -> self-attn, 16 heads, no mask -> proj_out conv1x1), residual added to the NORMALIZED input (`tortoise_norm=False`, confirmed from source -- NOT the raw pre-norm input, an easy detail to get backwards).</summary>
public sealed class XttsAttentionBlockWeights
{
    public float[] NormWeight { get; } // GroupNorm affine gamma, per-channel [1024]
    public float[] NormBias { get; }
    public float[] QkvWeight { get; } // conv1x1, 1024 -> 3072
    public float[] QkvBias { get; }
    public float[] ProjOutWeight { get; } // conv1x1, 1024 -> 1024
    public float[] ProjOutBias { get; }

    public XttsAttentionBlockWeights(SafetensorsLoader loader, string prefix)
    {
        NormWeight = loader.ReadF32($"{prefix}.norm.weight");
        NormBias = loader.ReadF32($"{prefix}.norm.bias");
        QkvWeight = loader.ReadF32($"{prefix}.qkv.weight");
        QkvBias = loader.ReadF32($"{prefix}.qkv.bias");
        ProjOutWeight = loader.ReadF32($"{prefix}.proj_out.weight");
        ProjOutBias = loader.ReadF32($"{prefix}.proj_out.bias");
    }
}

/// <summary>Real Perceiver Resampler layer: cross-attention (no bias, `to_q`/`to_kv`/`to_out`, `cross_attn_include_queries=True` -- context is [latents ++ input]) + GEGLU FeedForward, both with a plain residual add (no pre-norm inside the layer -- only the FINAL output gets RMSNorm'd, confirmed from source).</summary>
public sealed class XttsPerceiverLayerWeights
{
    public float[] ToQWeight { get; } // 1024 -> 512 (dim_head*heads), no bias
    public float[] ToKvWeight { get; } // 1024 -> 1024 (2 * dim_head*heads), no bias
    public float[] ToOutWeight { get; } // 512 -> 1024, no bias

    public float[] Ffn0Weight { get; } // 1024 -> 5460 (2*2730), no bias? (real nn.Linear default HAS bias -- confirmed via real tensor names below)
    public float[] Ffn0Bias { get; }
    public float[] Ffn2Weight { get; } // 2730 -> 1024
    public float[] Ffn2Bias { get; }

    public XttsPerceiverLayerWeights(SafetensorsLoader loader, string prefix)
    {
        ToQWeight = loader.ReadF32($"{prefix}.0.to_q.weight");
        ToKvWeight = loader.ReadF32($"{prefix}.0.to_kv.weight");
        ToOutWeight = loader.ReadF32($"{prefix}.0.to_out.weight");

        Ffn0Weight = loader.ReadF32($"{prefix}.1.0.weight");
        Ffn0Bias = loader.ReadF32($"{prefix}.1.0.bias");
        Ffn2Weight = loader.ReadF32($"{prefix}.1.2.weight");
        Ffn2Bias = loader.ReadF32($"{prefix}.1.2.bias");
    }
}
