namespace OpenTail.Stingray.Audio.VoxCpm2;

public sealed class VoxCpm2Conv1dWeights
{
    public required float[] Weight { get; init; } // regular: [out,in,kernel] row-major; depthwise: [out,1,kernel]
    public required float[] Bias { get; init; }
    public required int InChannels { get; init; }
    public required int OutChannels { get; init; }
    public required int Kernel { get; init; }
    public required bool Depthwise { get; init; }
}

public sealed class VoxCpm2ConvTranspose1dWeights
{
    public required float[] Weight { get; init; } // [in,out,kernel] row-major
    public required float[] Bias { get; init; }
    public required int InChannels { get; init; }
    public required int OutChannels { get; init; }
    public required int Kernel { get; init; }
}

public sealed class VoxCpm2ResidualUnitWeights
{
    public required float[] Snake1Alpha { get; init; }
    public required VoxCpm2Conv1dWeights Conv1 { get; init; } // depthwise, kernel 7
    public required float[] Snake2Alpha { get; init; }
    public required VoxCpm2Conv1dWeights Conv2 { get; init; } // pointwise, kernel 1
}

public sealed class VoxCpm2DecoderBlockWeights
{
    public required float[] SrCondScale { get; init; }
    public required float[] SrCondBias { get; init; }
    public required float[] SnakeAlpha { get; init; }
    public required VoxCpm2ConvTranspose1dWeights Upsample { get; init; }
    public required VoxCpm2ResidualUnitWeights[] Res { get; init; } // dilations 1,3,9
    public required int InputChannels { get; init; }
    public required int OutputChannels { get; init; }
    public required int Stride { get; init; }
}

/// <summary>
/// Real weight loader for VoxCPM2's AudioVAE decoder, ported from
/// `examples/audio.cpp/src/models/voxcpm2/audiovae.cpp`'s `load_vae_weights`/`load_wn_conv1d`/
/// `load_wn_conv_transpose1d`/`load_sr_condition`/`load_residual_unit` (not guessed). Decode-only
/// (this pipeline never re-encodes a reference clip through the AudioVAE encoder inline -- see
/// <see cref="VoxCpm2AudioVaeDecoder"/>'s doc comment). Real, non-obvious detail: `weight_norm`'s
/// magnitude parameter (`weight_g`) is per-OUTPUT-channel for a Conv1d (`dim0=out_channels`) but
/// per-INPUT-channel for a ConvTranspose1d (PyTorch's real `weight_norm(dim=0)` on a
/// `ConvTranspose1d` module normalizes over the weight's own dim 0, which for that module type is
/// `in_channels`, not `out_channels` -- a real, easy-to-get-backwards distinction, confirmed
/// directly against the reference's separate `load_wn_conv1d`/`load_wn_conv_transpose1d` shapes,
/// not assumed to be symmetric).
/// </summary>
public sealed class VoxCpm2AudioVaeDecoderWeights
{
    public required VoxCpm2Conv1dWeights DecoderFirstDepthwise { get; init; } // latent_dim -> latent_dim, k=7, depthwise
    public required VoxCpm2Conv1dWeights DecoderFirstPointwise { get; init; } // latent_dim -> decoder_dim, k=1
    public required VoxCpm2DecoderBlockWeights[] DecoderBlocks { get; init; }
    public required float[] DecoderFinalSnakeAlpha { get; init; }
    public required VoxCpm2Conv1dWeights DecoderFinalConv { get; init; } // final_channels -> 1, k=7

    public static VoxCpm2AudioVaeDecoderWeights Load(VoxCpm2AudioVaeConfig config, Func<string, float[]> get)
    {
        VoxCpm2Conv1dWeights LoadConv1d(string prefix, int outChannels, int inChannels, int kernel, bool depthwise)
        {
            int storedIn = depthwise ? 1 : inChannels;
            var v = get($"{prefix}.weight_v"); // [outChannels, storedIn, kernel]
            var g = get($"{prefix}.weight_g"); // [outChannels]
            var weight = ReconstructWeightNorm(g, v, dim0: outChannels, dim1: storedIn, kernel: kernel);
            return new VoxCpm2Conv1dWeights
            {
                Weight = weight,
                Bias = get($"{prefix}.bias"),
                InChannels = inChannels,
                OutChannels = outChannels,
                Kernel = kernel,
                Depthwise = depthwise,
            };
        }

        VoxCpm2ConvTranspose1dWeights LoadConvTranspose1d(string prefix, int inChannels, int outChannels, int kernel)
        {
            var v = get($"{prefix}.weight_v"); // [inChannels, outChannels, kernel]
            var g = get($"{prefix}.weight_g"); // [inChannels]
            var weight = ReconstructWeightNorm(g, v, dim0: inChannels, dim1: outChannels, kernel: kernel);
            return new VoxCpm2ConvTranspose1dWeights
            {
                Weight = weight,
                Bias = get($"{prefix}.bias"),
                InChannels = inChannels,
                OutChannels = outChannels,
                Kernel = kernel,
            };
        }

        VoxCpm2ResidualUnitWeights LoadResidualUnit(string prefix, int channels) => new()
        {
            Snake1Alpha = get($"{prefix}.block.0.alpha"),
            Conv1 = LoadConv1d($"{prefix}.block.1", channels, channels, kernel: 7, depthwise: true),
            Snake2Alpha = get($"{prefix}.block.2.alpha"),
            Conv2 = LoadConv1d($"{prefix}.block.3", channels, channels, kernel: 1, depthwise: false),
        };

        int bucket = config.SampleRateBucket();
        int buckets = config.SampleRateBucketCount;

        (float[] Scale, float[] Bias) LoadSrCondition(string prefix, int channels)
        {
            var scaleTable = get($"{prefix}.scale_embed.weight"); // [buckets, channels]
            var biasTable = get($"{prefix}.bias_embed.weight");
            var scale = new float[channels];
            var bias = new float[channels];
            Array.Copy(scaleTable, bucket * channels, scale, 0, channels);
            Array.Copy(biasTable, bucket * channels, bias, 0, channels);
            return (scale, bias);
        }

        var decoderFirstDepthwise = LoadConv1d("decoder.model.0", config.LatentDim, config.LatentDim, kernel: 7, depthwise: true);
        var decoderFirstPointwise = LoadConv1d("decoder.model.1", config.DecoderDim, config.LatentDim, kernel: 1, depthwise: false);

        var decoderBlocks = new VoxCpm2DecoderBlockWeights[config.DecoderRates.Length];
        for (int i = 0; i < config.DecoderRates.Length; i++)
        {
            int inputChannels = config.DecoderDim >> i;
            int outputChannels = config.DecoderDim >> (i + 1);
            int stride = config.DecoderRates[i];
            int modelIndex = i + 2;
            string blockPrefix = $"decoder.model.{modelIndex}";
            var (srScale, srBias) = LoadSrCondition($"decoder.sr_cond_model.{modelIndex}", inputChannels);

            decoderBlocks[i] = new VoxCpm2DecoderBlockWeights
            {
                SrCondScale = srScale,
                SrCondBias = srBias,
                SnakeAlpha = get($"{blockPrefix}.block.0.alpha"),
                Upsample = LoadConvTranspose1d($"{blockPrefix}.block.1", inputChannels, outputChannels, kernel: 2 * stride),
                Res =
                [
                    LoadResidualUnit($"{blockPrefix}.block.2", outputChannels),
                    LoadResidualUnit($"{blockPrefix}.block.3", outputChannels),
                    LoadResidualUnit($"{blockPrefix}.block.4", outputChannels),
                ],
                InputChannels = inputChannels,
                OutputChannels = outputChannels,
                Stride = stride,
            };
        }

        int finalChannels = config.DecoderDim >> config.DecoderRates.Length;
        var finalSnakeAlpha = get($"decoder.model.{config.DecoderRates.Length + 2}.alpha");
        var finalConv = LoadConv1d($"decoder.model.{config.DecoderRates.Length + 3}", 1, finalChannels, kernel: 7, depthwise: false);

        return new VoxCpm2AudioVaeDecoderWeights
        {
            DecoderFirstDepthwise = decoderFirstDepthwise,
            DecoderFirstPointwise = decoderFirstPointwise,
            DecoderBlocks = decoderBlocks,
            DecoderFinalSnakeAlpha = finalSnakeAlpha,
            DecoderFinalConv = finalConv,
        };
    }

    /// <summary>Real PyTorch `weight_norm(dim=0)` reconstruction: `weight = v * (g / ||v||)`, norm
    /// computed per `dim0` slice across the remaining `dim1*kernel` elements (double precision,
    /// matching the reference's `fold_weight_norm` exactly).</summary>
    private static float[] ReconstructWeightNorm(float[] g, float[] v, int dim0, int dim1, int kernel)
    {
        var weight = new float[dim0 * dim1 * kernel];
        int rest = dim1 * kernel;
        for (int d0 = 0; d0 < dim0; d0++)
        {
            int baseIdx = d0 * rest;
            double normSq = 0;
            for (int i = 0; i < rest; i++) normSq += (double)v[baseIdx + i] * v[baseIdx + i];
            float scale = (float)(g[d0] / Math.Sqrt(normSq));
            for (int i = 0; i < rest; i++) weight[baseIdx + i] = v[baseIdx + i] * scale;
        }
        return weight;
    }
}
