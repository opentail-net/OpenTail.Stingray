
namespace OpenTail.Stingray.Audio.Rvc;

/// <summary>
/// Real weight loader for RVC's bundled RMVPE pitch extractor (`support_rmvpe/*` in the packed
/// `rvc-f16.gguf`). Real 5-level ResNet-style U-Net (encoder/intermediate/decoder) + bidirectional
/// GRU + sigmoid 360-class pitch head -- see docs/audio-review-progress.md's RMVPE section for the
/// full architecture derivation from `examples/audio.cpp/src/framework/modules/pitch_extractors/
/// rmvpe_pitch_extractor.cpp` (not guessed). This class is weight-loading plumbing only; the
/// forward pass (RvcRmvpeEncoder) is a separate, not-yet-written next step.
/// </summary>
public sealed class RvcRmvpeWeights
{
    public const int MelBins = 128;
    public const int FeatureDim = 384; // 3 (final conv out channels) * 128 (mel bins)
    public const int GruHiddenDim = 256;
    public const int NumPitchClasses = 360;

    public RvcBatchNorm2d EncoderInputBn { get; }
    public RvcResUNetLevel[] EncoderLevels { get; } = new RvcResUNetLevel[5];
    public RvcResUNetLevel[] IntermediateLevels { get; } = new RvcResUNetLevel[4];
    public RvcDecoderLevel[] DecoderLevels { get; } = new RvcDecoderLevel[5];

    public float[] CnnWeight { get; } // [3, 16, 3, 3] real PyTorch (out, in, kh, kw)
    public float[] CnnBias { get; }

    public RvcGruWeights GruForward { get; }
    public RvcGruWeights GruReverse { get; }

    public float[] FcOutWeight { get; } // [360, 512]
    public float[] FcOutBias { get; }

    public RvcRmvpeWeights(RvcPackedTensorSource source)
    {
        EncoderInputBn = new RvcBatchNorm2d(source, "support_rmvpe/unet.encoder.bn");

        int channels = 1, outChannels = 16;
        for (int level = 0; level < 5; level++)
        {
            EncoderLevels[level] = new RvcResUNetLevel(source, $"support_rmvpe/unet.encoder.layers.{level}.conv", channels, outChannels);
            channels = outChannels;
            outChannels *= 2;
        }

        channels = 256;
        int interOut = 512;
        for (int level = 0; level < 4; level++)
        {
            IntermediateLevels[level] = new RvcResUNetLevel(source, $"support_rmvpe/unet.intermediate.layers.{level}.conv", channels, interOut);
            channels = interOut;
        }

        channels = 512;
        for (int level = 0; level < 5; level++)
        {
            int decOut = channels / 2;
            DecoderLevels[level] = new RvcDecoderLevel(source, $"support_rmvpe/unet.decoder.layers.{level}", inChannels: channels, upOutChannels: decOut);
            channels = decOut;
        }

        CnnWeight = source.GetTensor("support_rmvpe/cnn.weight");
        CnnBias = source.GetTensor("support_rmvpe/cnn.bias");

        GruForward = new RvcGruWeights(source, "support_rmvpe/fc.0.gru", reverse: false);
        GruReverse = new RvcGruWeights(source, "support_rmvpe/fc.0.gru", reverse: true);

        FcOutWeight = source.GetTensor("support_rmvpe/fc.1.weight");
        FcOutBias = source.GetTensor("support_rmvpe/fc.1.bias");
    }
}

/// <summary>Real PyTorch BatchNorm2d (inference-mode: running_mean/running_var, no batch statistics computed) -- `num_batches_tracked` is training-only bookkeeping, safely ignored at inference.</summary>
public sealed class RvcBatchNorm2d
{
    public float[] Weight { get; }
    public float[] Bias { get; }
    public float[] RunningMean { get; }
    public float[] RunningVar { get; }

    public RvcBatchNorm2d(RvcPackedTensorSource source, string prefix)
    {
        Weight = source.GetTensor($"{prefix}.weight");
        Bias = source.GetTensor($"{prefix}.bias");
        RunningMean = source.GetTensor($"{prefix}.running_mean");
        RunningVar = source.GetTensor($"{prefix}.running_var");
    }
}

/// <summary>One real ResNet BasicBlock: conv(3x3,pad1)-BN-ReLU-conv(3x3,pad1)-BN, plus an optional 1x1 conv+bias shortcut (present only when in_channels != out_channels, i.e. the first block of a level).</summary>
public sealed class RvcResBlock
{
    public float[] Conv0Weight { get; } // [out, in, 3, 3]
    public RvcBatchNorm2d Bn1 { get; }
    public float[] Conv3Weight { get; } // [out, out, 3, 3]
    public RvcBatchNorm2d Bn4 { get; }
    public float[]? ShortcutWeight { get; } // [out, in, 1, 1], null when in==out
    public float[]? ShortcutBias { get; }

    public RvcResBlock(RvcPackedTensorSource source, string prefix, int inChannels, int outChannels)
    {
        Conv0Weight = source.GetTensor($"{prefix}.conv.0.weight");
        Bn1 = new RvcBatchNorm2d(source, $"{prefix}.conv.1");
        Conv3Weight = source.GetTensor($"{prefix}.conv.3.weight");
        Bn4 = new RvcBatchNorm2d(source, $"{prefix}.conv.4");
        if (inChannels != outChannels)
        {
            ShortcutWeight = source.GetTensor($"{prefix}.shortcut.weight");
            ShortcutBias = source.GetTensor($"{prefix}.shortcut.bias");
        }
    }
}

/// <summary>One encoder/intermediate U-Net level: 4 stacked ResBlocks (only the first changes channel count).</summary>
public sealed class RvcResUNetLevel
{
    public RvcResBlock[] Blocks { get; } = new RvcResBlock[4];

    public RvcResUNetLevel(RvcPackedTensorSource source, string prefix, int inChannels, int outChannels)
    {
        int ch = inChannels;
        for (int b = 0; b < 4; b++)
        {
            Blocks[b] = new RvcResBlock(source, $"{prefix}.{b}", ch, outChannels);
            ch = outChannels;
        }
    }
}

/// <summary>One decoder level: ConvTranspose2d(2x2,stride2) upsample + BN + ReLU (`conv1`), then 4 ResBlocks over the concatenated [upsampled, skip] channels (`conv2`).</summary>
public sealed class RvcDecoderLevel
{
    public float[] UpsampleWeight { get; } // real PyTorch ConvTranspose2d weight [in, out, 2, 2]
    public RvcBatchNorm2d UpsampleBn { get; }
    public RvcResBlock[] Conv2Blocks { get; } = new RvcResBlock[4];

    public RvcDecoderLevel(RvcPackedTensorSource source, string prefix, int inChannels, int upOutChannels)
    {
        UpsampleWeight = source.GetTensor($"{prefix}.conv1.0.weight");
        UpsampleBn = new RvcBatchNorm2d(source, $"{prefix}.conv1.1");

        // After upsample+concat-with-skip, channel count is upOutChannels*2 (skip has the same
        // channel count as the upsampled tensor, matching encoder/decoder symmetry).
        int concatChannels = upOutChannels * 2;
        int ch = concatChannels;
        for (int b = 0; b < 4; b++)
        {
            Conv2Blocks[b] = new RvcResBlock(source, $"{prefix}.conv2.{b}", ch, upOutChannels);
            ch = upOutChannels;
        }
    }
}

/// <summary>Real PyTorch GRU cell weights for one direction (fairseq/PyTorch convention: combined [3*hidden, input] and [3*hidden, hidden] weight matrices for the r/z/n gates, real `_reverse`-suffixed tensors for the backward direction).</summary>
public sealed class RvcGruWeights
{
    public float[] WeightIh { get; } // [3*hidden, input]
    public float[] BiasIh { get; }
    public float[] WeightHh { get; } // [3*hidden, hidden]
    public float[] BiasHh { get; }

    public RvcGruWeights(RvcPackedTensorSource source, string prefix, bool reverse)
    {
        string suffix = reverse ? "_reverse" : "";
        WeightIh = source.GetTensor($"{prefix}.weight_ih_l0{suffix}");
        BiasIh = source.GetTensor($"{prefix}.bias_ih_l0{suffix}");
        WeightHh = source.GetTensor($"{prefix}.weight_hh_l0{suffix}");
        BiasHh = source.GetTensor($"{prefix}.bias_hh_l0{suffix}");
    }
}
