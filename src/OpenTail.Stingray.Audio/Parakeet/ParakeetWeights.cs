using System;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Parakeet;

/// <summary>
/// Container for NVIDIA NeMo Parakeet FastConformer CTC ASR GGUF weights (`general.architecture
/// = canary_ctc`). Full architecture spec, tensor naming, and BN-fold formula verified against
/// `examples/crispasr/src/canary_ctc.cpp` and `src/core/fastconformer.h` -- see
/// docs/audio-review-progress.md's Parakeet section for the full derivation. Eagerly dequantizes
/// every tensor to float32 in file storage order (same convention as Chatterbox/Kokoro weights).
/// </summary>
public sealed class ParakeetWeights : IDisposable
{
    public GgufModel Model { get; }

    public int NumLayers { get; } = 24;
    public int HiddenDim { get; } = 1024;
    public int NumHeads { get; } = 8;
    public int HeadDim { get; } = 128;
    public int FfDim { get; } = 4096;
    public int ConvKernel { get; } = 9;
    public int VocabSize { get; } = 1024;
    public int BlankTokenId { get; } = 1024;
    public int SubsampleFactor { get; } = 8;
    public int SubsampleChannels { get; } = 256;
    public int NMels { get; } = 80;
    public int NFft { get; } = 512;
    public int WinLength { get; } = 400;
    public int HopLength { get; } = 160;
    public int SampleRate { get; } = 16000;

    public const float LayerNormEps = 1e-5f;

    // --- Subsampling front-end (dw_striding, 8x) ---
    public float[] PreConv0Weight { get; }  // [3,3,1,256] full conv2d, in=1
    public float[] PreConv0Bias { get; }
    public float[] PreConv2Weight { get; }  // [3,3,1,256] depthwise
    public float[] PreConv2Bias { get; }
    public float[] PreConv3Weight { get; }  // [1,1,256,256] pointwise
    public float[] PreConv3Bias { get; }
    public float[] PreConv5Weight { get; }  // [3,3,1,256] depthwise
    public float[] PreConv5Bias { get; }
    public float[] PreConv6Weight { get; }  // [1,1,256,256] pointwise
    public float[] PreConv6Bias { get; }
    public float[] PreOutWeight { get; }    // [2560, 1024]
    public float[] PreOutBias { get; }

    // --- Mel preprocessing (shipped in checkpoint, don't recompute) ---
    public float[] MelFilterbank { get; }   // [257, 80]
    public float[] MelWindow { get; }       // [400]

    // --- CTC head ---
    public float[] CtcWeight { get; }       // [1024, vocab+1]
    public float[]? CtcBias { get; }

    public ParakeetConformerLayer[] Layers { get; }

    public ParakeetWeights(string ggufPath)
    {
        if (!File.Exists(ggufPath))
            throw new FileNotFoundException($"Parakeet GGUF model file not found: {ggufPath}");

        Model = GgufModel.Open(ggufPath);

        NumLayers = GetInt("canary_ctc.n_layers", NumLayers);
        HiddenDim = GetInt("canary_ctc.d_model", HiddenDim);
        NumHeads = GetInt("canary_ctc.n_heads", NumHeads);
        HeadDim = GetInt("canary_ctc.head_dim", HiddenDim / NumHeads);
        FfDim = GetInt("canary_ctc.ff_dim", FfDim);
        ConvKernel = GetInt("canary_ctc.conv_kernel", ConvKernel);
        VocabSize = GetInt("canary_ctc.vocab_size", VocabSize);
        BlankTokenId = GetInt("canary_ctc.blank_id", BlankTokenId);
        SubsampleFactor = GetInt("canary_ctc.subsampling_factor", SubsampleFactor);
        SubsampleChannels = GetInt("canary_ctc.subsampling_channels", SubsampleChannels);
        NMels = GetInt("canary_ctc.n_mels", NMels);
        NFft = GetInt("canary_ctc.n_fft", NFft);
        WinLength = GetInt("canary_ctc.win_length", WinLength);
        HopLength = GetInt("canary_ctc.hop_length", HopLength);
        SampleRate = GetInt("canary_ctc.sample_rate", SampleRate);

        PreConv0Weight = GetTensor("encoder.pre.conv.0.weight");
        PreConv0Bias = GetTensor("encoder.pre.conv.0.bias");
        PreConv2Weight = GetTensor("encoder.pre.conv.2.weight");
        PreConv2Bias = GetTensor("encoder.pre.conv.2.bias");
        PreConv3Weight = GetTensor("encoder.pre.conv.3.weight");
        PreConv3Bias = GetTensor("encoder.pre.conv.3.bias");
        PreConv5Weight = GetTensor("encoder.pre.conv.5.weight");
        PreConv5Bias = GetTensor("encoder.pre.conv.5.bias");
        PreConv6Weight = GetTensor("encoder.pre.conv.6.weight");
        PreConv6Bias = GetTensor("encoder.pre.conv.6.bias");
        PreOutWeight = GetTensor("encoder.pre.out.weight");
        PreOutBias = GetTensor("encoder.pre.out.bias");

        MelFilterbank = GetTensor("preprocessor.fb");
        MelWindow = GetTensor("preprocessor.window");

        CtcWeight = GetTensor("ctc.weight");
        CtcBias = TryGetTensor("ctc.bias");

        Layers = new ParakeetConformerLayer[NumLayers];
        for (int i = 0; i < NumLayers; i++)
            Layers[i] = new ParakeetConformerLayer(this, $"encoder.layers.{i}", ConvKernel);
    }

    private int GetInt(string key, int fallback) =>
        Model.Metadata.TryGetValue(key, out var v) ? Convert.ToInt32(v) : fallback;

    /// <summary>Loads and dequantizes a required tensor by exact GGUF name to a flat float[] in file storage order.</summary>
    public float[] GetTensor(string name)
    {
        var info = Model.FindTensor(name) ?? throw new InvalidDataException($"Parakeet GGUF missing required tensor '{name}'.");
        return DequantTensor(Model, info);
    }

    public float[]? TryGetTensor(string name)
    {
        var info = Model.FindTensor(name);
        return info is null ? null : DequantTensor(Model, info.Value);
    }

    private static float[] DequantTensor(GgufModel model, GgufTensorInfo info)
    {
        var bytes = model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }

    public void Dispose()
    {
        Model.Dispose();
    }
}

/// <summary>
/// One Conformer block's weights (macaron FFN1 -> rel-pos self-attn -> conv module -> FFN2 ->
/// final LayerNorm), plus the BatchNorm-folded depthwise conv weight/bias. This checkpoint's
/// q/k/v/out/ff linears carry NO bias (canary_ctc arch, confirmed against the converter);
/// `attn.pos` (the relative-position projection) also has no bias.
/// </summary>
public sealed class ParakeetConformerLayer
{
    public float[] NormFf1Weight { get; }
    public float[] NormFf1Bias { get; }
    public float[] Ff1Linear1Weight { get; }  // [d, ff]
    public float[] Ff1Linear1Bias { get; }
    public float[] Ff1Linear2Weight { get; }  // [ff, d]
    public float[] Ff1Linear2Bias { get; }

    public float[] NormAttnWeight { get; }
    public float[] NormAttnBias { get; }
    public float[] AttnQWeight { get; }
    public float[]? AttnQBias { get; }
    public float[] AttnKWeight { get; }
    public float[]? AttnKBias { get; }
    public float[] AttnVWeight { get; }
    public float[]? AttnVBias { get; }
    public float[] AttnOutWeight { get; }
    public float[]? AttnOutBias { get; }
    public float[] AttnPosWeight { get; }
    public float[] AttnPosBiasU { get; }  // [head_dim, n_heads]
    public float[] AttnPosBiasV { get; }  // [head_dim, n_heads]

    public float[] NormConvWeight { get; }
    public float[] NormConvBias { get; }
    public float[] ConvPw1Weight { get; }   // [d, 2d]
    public float[] ConvPw1Bias { get; }
    /// <summary>Depthwise conv weight, BatchNorm-folded at load time (raw checkpoint ships unfused BN tensors -- see docs/audio-review-progress.md).</summary>
    public float[] ConvDwWeight { get; }    // [K, d]
    public float[] ConvDwBias { get; }
    public float[] ConvPw2Weight { get; }   // [d, d]
    public float[] ConvPw2Bias { get; }

    public float[] NormFf2Weight { get; }
    public float[] NormFf2Bias { get; }
    public float[] Ff2Linear1Weight { get; }
    public float[] Ff2Linear1Bias { get; }
    public float[] Ff2Linear2Weight { get; }
    public float[] Ff2Linear2Bias { get; }

    public float[] NormOutWeight { get; }
    public float[] NormOutBias { get; }

    public ParakeetConformerLayer(ParakeetWeights w, string prefix, int convKernel)
    {
        NormFf1Weight = w.GetTensor($"{prefix}.norm_ff1.weight");
        NormFf1Bias = w.GetTensor($"{prefix}.norm_ff1.bias");
        Ff1Linear1Weight = w.GetTensor($"{prefix}.ff1.linear1.weight");
        Ff1Linear1Bias = w.GetTensor($"{prefix}.ff1.linear1.bias");
        Ff1Linear2Weight = w.GetTensor($"{prefix}.ff1.linear2.weight");
        Ff1Linear2Bias = w.GetTensor($"{prefix}.ff1.linear2.bias");

        NormAttnWeight = w.GetTensor($"{prefix}.norm_attn.weight");
        NormAttnBias = w.GetTensor($"{prefix}.norm_attn.bias");
        AttnQWeight = w.GetTensor($"{prefix}.attn.q.weight");
        AttnQBias = w.TryGetTensor($"{prefix}.attn.q.bias");
        AttnKWeight = w.GetTensor($"{prefix}.attn.k.weight");
        AttnKBias = w.TryGetTensor($"{prefix}.attn.k.bias");
        AttnVWeight = w.GetTensor($"{prefix}.attn.v.weight");
        AttnVBias = w.TryGetTensor($"{prefix}.attn.v.bias");
        AttnOutWeight = w.GetTensor($"{prefix}.attn.out.weight");
        AttnOutBias = w.TryGetTensor($"{prefix}.attn.out.bias");
        AttnPosWeight = w.GetTensor($"{prefix}.attn.pos.weight");
        AttnPosBiasU = w.GetTensor($"{prefix}.attn.pos_bias_u");
        AttnPosBiasV = w.GetTensor($"{prefix}.attn.pos_bias_v");

        NormConvWeight = w.GetTensor($"{prefix}.norm_conv.weight");
        NormConvBias = w.GetTensor($"{prefix}.norm_conv.bias");
        ConvPw1Weight = w.GetTensor($"{prefix}.conv.pw1.weight");
        ConvPw1Bias = w.GetTensor($"{prefix}.conv.pw1.bias");
        ConvPw2Weight = w.GetTensor($"{prefix}.conv.pw2.weight");
        ConvPw2Bias = w.GetTensor($"{prefix}.conv.pw2.bias");

        var dwWeightRaw = w.GetTensor($"{prefix}.conv.dw.weight");  // [K, 1, d] storage order
        var dwBiasRaw = w.GetTensor($"{prefix}.conv.dw.bias");
        var bnWeight = w.GetTensor($"{prefix}.conv.bn.weight");
        var bnBias = w.GetTensor($"{prefix}.conv.bn.bias");
        var bnMean = w.GetTensor($"{prefix}.conv.bn.running_mean");
        var bnVar = w.GetTensor($"{prefix}.conv.bn.running_var");
        (ConvDwWeight, ConvDwBias) = FoldBatchNorm(dwWeightRaw, dwBiasRaw, bnWeight, bnBias, bnMean, bnVar, w.HiddenDim, convKernel);

        NormFf2Weight = w.GetTensor($"{prefix}.norm_ff2.weight");
        NormFf2Bias = w.GetTensor($"{prefix}.norm_ff2.bias");
        Ff2Linear1Weight = w.GetTensor($"{prefix}.ff2.linear1.weight");
        Ff2Linear1Bias = w.GetTensor($"{prefix}.ff2.linear1.bias");
        Ff2Linear2Weight = w.GetTensor($"{prefix}.ff2.linear2.weight");
        Ff2Linear2Bias = w.GetTensor($"{prefix}.ff2.linear2.bias");

        NormOutWeight = w.GetTensor($"{prefix}.norm_out.weight");
        NormOutBias = w.GetTensor($"{prefix}.norm_out.bias");
    }

    /// <summary>
    /// Folds BatchNorm1d (applied after the depthwise conv, before SiLU, in the original NeMo
    /// module) into the depthwise conv's weight/bias, matching `canary_ctc.cpp`'s
    /// `cc_fold_batchnorm` exactly: `s[c] = bn_w[c] / sqrt(bn_var[c] + eps)`,
    /// `w_folded[k,c] = w[k,c] * s[c]`, `b_folded[c] = s[c]*orig_b[c] - bn_mean[c]*s[c] + bn_b[c]`.
    /// </summary>
    private static (float[] weight, float[] bias) FoldBatchNorm(
        float[] dwWeight, float[] dwBias, float[] bnWeight, float[] bnBias, float[] bnMean, float[] bnVar,
        int channels, int kernel)
    {
        const float eps = 1e-5f; // BN eps, per canary_ctc.cpp's cc_fold_batchnorm (numerically same as LayerNormEps, tracked separately since they're conceptually distinct)
        var s = new float[channels];
        for (int c = 0; c < channels; c++)
            s[c] = bnWeight[c] / MathF.Sqrt(bnVar[c] + eps);

        var foldedWeight = new float[dwWeight.Length];
        // GGML ne=[K,1,d] (K fastest-varying) -> flat index = k + c*K, confirmed against
        // canary_ctc.cpp's cc_fold_batchnorm: `w_f32[ki + c * K] *= s[c]` (an earlier version
        // of this port assumed k*channels+c, the wrong order -- caught by reading the actual
        // fold loop's indexing instead of guessing from the GGUF dims listing alone).
        for (int c = 0; c < channels; c++)
            for (int k = 0; k < kernel; k++)
                foldedWeight[c * kernel + k] = dwWeight[c * kernel + k] * s[c];

        var foldedBias = new float[channels];
        for (int c = 0; c < channels; c++)
            foldedBias[c] = s[c] * dwBias[c] - bnMean[c] * s[c] + bnBias[c];

        return (foldedWeight, foldedBias);
    }
}
