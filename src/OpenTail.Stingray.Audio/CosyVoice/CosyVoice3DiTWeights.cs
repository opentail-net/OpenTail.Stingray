using System;
using System.IO;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// CosyVoice3's flow-matching DiT estimator (`decoder.estimator.*` in `models/cosyvoice3/
/// CosyVoice3-2512_F16.gguf`) -- confirmed this session to be, tensor-for-tensor, the SAME
/// architecture as `F5TTS/F5TtsWeights.cs`'s DiT (`transformer_blocks.{i}.{attn_norm.linear,
/// attn.to_q/k/v, attn.to_out.0, ff.ff.0.0, ff.ff.2}`, `time_embed.time_mlp.0/2`, `norm_out.
/// linear`, `proj_out` -- byte-identical tensor-name suffixes, just prefixed with `decoder.
/// estimator.`), with identical hyperparameters (hidden=1024, heads=16, head_dim=64, ffn=2048,
/// depth=22 -- all read directly from real tensor shapes and the GGUF's own `decoder.
/// estimator.depth`/`decoder.estimator.heads` metadata, not assumed) -- only `MelDim` differs
/// (80 here vs F5's 100). This is architecturally NOT the Chatterbox/CosyVoice2 Conformer flow
/// encoder at all; see docs/audio-review-progress.md's CosyVoice3 section for the full
/// derivation and why `S3GenConformerKernels` doesn't apply here.
///
/// Only the confirmed-identical pieces are loaded/ported so far: the 22-block transformer
/// backbone, timestep embedding, and final norm_out/proj_out. The `input_embed.proj`+
/// `conv_pos_embed` stage's exact input composition (what real tensors are concatenated into
/// the 320-dim vector fed to `proj`) is NOT yet confirmed against `examples/cosyvoice.cpp`'s
/// real estimator-input construction, so it is deliberately NOT implemented here rather than
/// guessed -- see <see cref="CosyVoice3DiTModel"/>'s doc comment.
/// </summary>
public sealed class CosyVoice3DiTWeights
{
    public const int HiddenDim = 1024;
    public const int NumHeads = 16;
    public const int HeadDim = HiddenDim / NumHeads; // 64
    public const int FfnDim = 2048;
    public const int MelDim = 80;
    public const int TimeFreqDim = 256;
    public const int ConvPosKernel = 31;
    public const int ConvPosGroups = 16;

    public int NumLayers { get; }

    public float[] InputProjWeight { get; }  // [1024, concatDim=320]
    public float[] InputProjBias { get; }
    public float[] ConvPos1Weight { get; }   // grouped conv1d, [1024, HiddenDim/groups, 31]
    public float[] ConvPos1Bias { get; }
    public float[] ConvPos2Weight { get; }
    public float[] ConvPos2Bias { get; }

    public float[] TimeMlp0Weight { get; }
    public float[] TimeMlp0Bias { get; }
    public float[] TimeMlp2Weight { get; }
    public float[] TimeMlp2Bias { get; }

    public float[] NormOutLinearWeight { get; }  // [2048, 1024]
    public float[] NormOutLinearBias { get; }
    public float[] ProjOutWeight { get; }        // [80, 1024]
    public float[] ProjOutBias { get; }

    public CosyVoice3DiTBlockWeights[] Blocks { get; }

    public CosyVoice3DiTWeights(GgufModel model)
    {
        NumLayers = model.Metadata.TryGetValue("decoder.estimator.depth", out var d) ? Convert.ToInt32(d) : 22;

        InputProjWeight = GetTensor(model, "decoder.estimator.input_embed.proj.weight");
        InputProjBias = GetTensor(model, "decoder.estimator.input_embed.proj.bias");
        ConvPos1Weight = GetTensor(model, "decoder.estimator.input_embed.conv_pos_embed.conv1.0.weight");
        ConvPos1Bias = GetTensor(model, "decoder.estimator.input_embed.conv_pos_embed.conv1.0.bias");
        ConvPos2Weight = GetTensor(model, "decoder.estimator.input_embed.conv_pos_embed.conv2.0.weight");
        ConvPos2Bias = GetTensor(model, "decoder.estimator.input_embed.conv_pos_embed.conv2.0.bias");

        TimeMlp0Weight = GetTensor(model, "decoder.estimator.time_embed.time_mlp.0.weight");
        TimeMlp0Bias = GetTensor(model, "decoder.estimator.time_embed.time_mlp.0.bias");
        TimeMlp2Weight = GetTensor(model, "decoder.estimator.time_embed.time_mlp.2.weight");
        TimeMlp2Bias = GetTensor(model, "decoder.estimator.time_embed.time_mlp.2.bias");

        NormOutLinearWeight = GetTensor(model, "decoder.estimator.norm_out.linear.weight");
        NormOutLinearBias = GetTensor(model, "decoder.estimator.norm_out.linear.bias");
        ProjOutWeight = GetTensor(model, "decoder.estimator.proj_out.weight");
        ProjOutBias = GetTensor(model, "decoder.estimator.proj_out.bias");

        Blocks = new CosyVoice3DiTBlockWeights[NumLayers];
        for (int i = 0; i < NumLayers; i++)
            Blocks[i] = new CosyVoice3DiTBlockWeights(model, i);
    }

    internal static float[] GetTensor(GgufModel model, string name)
    {
        var info = model.FindTensor(name) ?? throw new InvalidDataException($"CosyVoice3 GGUF missing required tensor '{name}'.");
        var bytes = model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Cpu.Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }
}

/// <summary>One DiT block: AdaLN-Zero self-attention (RoPE) + gated GELU FFN -- identical structure to <see cref="F5TTS.F5DiTBlockWeights"/>, ported separately since that class is coupled to <c>F5TtsWeights</c>' tensor source.</summary>
public sealed class CosyVoice3DiTBlockWeights
{
    public float[] AttnNormLinearWeight { get; }
    public float[] AttnNormLinearBias { get; }
    public float[] ToQWeight { get; }
    public float[] ToQBias { get; }
    public float[] ToKWeight { get; }
    public float[] ToKBias { get; }
    public float[] ToVWeight { get; }
    public float[] ToVBias { get; }
    public float[] ToOutWeight { get; }
    public float[] ToOutBias { get; }
    public float[] FfInWeight { get; }
    public float[] FfInBias { get; }
    public float[] FfOutWeight { get; }
    public float[] FfOutBias { get; }

    public CosyVoice3DiTBlockWeights(GgufModel model, int i)
    {
        string p = $"decoder.estimator.transformer_blocks.{i}";
        AttnNormLinearWeight = CosyVoice3DiTWeights.GetTensor(model, $"{p}.attn_norm.linear.weight");
        AttnNormLinearBias = CosyVoice3DiTWeights.GetTensor(model, $"{p}.attn_norm.linear.bias");
        ToQWeight = CosyVoice3DiTWeights.GetTensor(model, $"{p}.attn.to_q.weight");
        ToQBias = CosyVoice3DiTWeights.GetTensor(model, $"{p}.attn.to_q.bias");
        ToKWeight = CosyVoice3DiTWeights.GetTensor(model, $"{p}.attn.to_k.weight");
        ToKBias = CosyVoice3DiTWeights.GetTensor(model, $"{p}.attn.to_k.bias");
        ToVWeight = CosyVoice3DiTWeights.GetTensor(model, $"{p}.attn.to_v.weight");
        ToVBias = CosyVoice3DiTWeights.GetTensor(model, $"{p}.attn.to_v.bias");
        ToOutWeight = CosyVoice3DiTWeights.GetTensor(model, $"{p}.attn.to_out.0.weight");
        ToOutBias = CosyVoice3DiTWeights.GetTensor(model, $"{p}.attn.to_out.0.bias");
        FfInWeight = CosyVoice3DiTWeights.GetTensor(model, $"{p}.ff.ff.0.0.weight");
        FfInBias = CosyVoice3DiTWeights.GetTensor(model, $"{p}.ff.ff.0.0.bias");
        FfOutWeight = CosyVoice3DiTWeights.GetTensor(model, $"{p}.ff.ff.2.weight");
        FfOutBias = CosyVoice3DiTWeights.GetTensor(model, $"{p}.ff.ff.2.bias");
    }
}
