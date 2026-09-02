
namespace OpenTail.Stingray.Audio.MusicGen;

/// <summary>
/// Weight loader for MusicGen's audio codec, EnCodec 32kHz (`audio_encoder.*` prefix in
/// musicgen-small's own `model.safetensors`). Only the DECODER path is loaded -- MusicGen
/// generation never runs EnCodec's own encoder (that only matters for encoding real reference
/// audio, which text-to-music generation never needs).
///
/// <para>Real tensor layout confirmed against the checkpoint's own safetensors header
/// (2026-09-02): `audio_encoder.quantizer.layers.{0..3}.codebook.embed` (`[2048,128]` per
/// codebook, real RVQ codebook vectors -- NOT `embed_avg`/`cluster_size`, which are
/// training-only EMA state, unused at inference); `audio_encoder.decoder.layers.{i}.*` with
/// `weight_g`/`weight_v` real PyTorch `weight_norm`-wrapped conv pairs throughout (`norm_type:
/// "weight_norm"` in config) -- folded once at load via <see cref="FoldConvWeight"/> (same
/// convention as <see cref="Parler.DacWeights"/>). Layer index layout, derived from the real
/// tensor shapes (see docs/062-musicgen-implementation-plan.md): 0=initial conv(128-&gt;1024,k7),
/// 1=2-layer LSTM(1024, with a real residual/skip connection), {3,6,9,12}=upsampling
/// ConvTranspose1d (ratios 8,5,4,4 in that order), {4,7,10,13}=one residual block each (real
/// `num_residual_layers=1`, dilation=1 -- NOT DAC's 3-dilation stack), 15=final
/// conv(64-&gt;1,k7). Implicit (parameter-free) ELU activations sit between every stage at the
/// odd-numbered gaps (2,5,8,11,14) -- matched structurally in <see cref="EncodecDecoder"/>, not
/// represented as tensors here.</para>
/// </summary>
public sealed class EncodecDecoderWeights
{
    public const int LatentDim = MusicGenConfig.EncodecHiddenSize; // 128
    public static readonly int[] Ratios = MusicGenConfig.EncodecUpsamplingRatios; // [8,5,4,4], decoder order
    public static readonly int[] ChannelsPerStage = [1024, 512, 256, 128, 64]; // channel count entering each of the 5 top-level stages (initial conv output, then after each upsample)

    public float[][] Codebooks { get; } = new float[MusicGenConfig.NumCodebooks][]; // [codebook][2048*128]

    public float[] InitConvWeight { get; }
    public float[] InitConvBias { get; }

    public EncodecLstmWeights Lstm { get; }

    public EncodecUpsampleStageWeights[] Stages { get; } = new EncodecUpsampleStageWeights[4];

    public float[] OutConvWeight { get; }
    public float[] OutConvBias { get; }

    public EncodecDecoderWeights(SafetensorsLoader loader)
    {
        for (int q = 0; q < MusicGenConfig.NumCodebooks; q++)
            Codebooks[q] = loader.ReadF32($"audio_encoder.quantizer.layers.{q}.codebook.embed");

        InitConvWeight = FoldConvWeight(loader, "audio_encoder.decoder.layers.0.conv");
        InitConvBias = loader.ReadF32("audio_encoder.decoder.layers.0.conv.bias");

        Lstm = new EncodecLstmWeights
        {
            WeightIhL0 = loader.ReadF32("audio_encoder.decoder.layers.1.lstm.weight_ih_l0"),
            WeightHhL0 = loader.ReadF32("audio_encoder.decoder.layers.1.lstm.weight_hh_l0"),
            BiasIhL0 = loader.ReadF32("audio_encoder.decoder.layers.1.lstm.bias_ih_l0"),
            BiasHhL0 = loader.ReadF32("audio_encoder.decoder.layers.1.lstm.bias_hh_l0"),
            WeightIhL1 = loader.ReadF32("audio_encoder.decoder.layers.1.lstm.weight_ih_l1"),
            WeightHhL1 = loader.ReadF32("audio_encoder.decoder.layers.1.lstm.weight_hh_l1"),
            BiasIhL1 = loader.ReadF32("audio_encoder.decoder.layers.1.lstm.bias_ih_l1"),
            BiasHhL1 = loader.ReadF32("audio_encoder.decoder.layers.1.lstm.bias_hh_l1"),
        };

        int[] upsampleLayerIndex = [3, 6, 9, 12];
        int[] blockLayerIndex = [4, 7, 10, 13];
        for (int i = 0; i < 4; i++)
        {
            string up = $"audio_encoder.decoder.layers.{upsampleLayerIndex[i]}.conv";
            string blk = $"audio_encoder.decoder.layers.{blockLayerIndex[i]}.block";
            Stages[i] = new EncodecUpsampleStageWeights
            {
                UpsampleWeight = FoldConvWeight(loader, up),
                UpsampleBias = loader.ReadF32($"{up}.bias"),
                ResBlockConv0Weight = FoldConvWeight(loader, $"{blk}.1.conv"),
                ResBlockConv0Bias = loader.ReadF32($"{blk}.1.conv.bias"),
                ResBlockConv1Weight = FoldConvWeight(loader, $"{blk}.3.conv"),
                ResBlockConv1Bias = loader.ReadF32($"{blk}.3.conv.bias"),
            };
        }

        OutConvWeight = FoldConvWeight(loader, "audio_encoder.decoder.layers.15.conv");
        OutConvBias = loader.ReadF32("audio_encoder.decoder.layers.15.conv.bias");
    }

    /// <summary>Folds `weight_g`(`[outCh,1,1]`)*`weight_v`(`[outCh,inCh,K]`)/||v[outCh,:,:]||_2 into a plain conv weight -- same PyTorch `weight_norm` (dim=0) convention as <see cref="Parler.DacWeights"/>'s identically-named helper.</summary>
    private static float[] FoldConvWeight(SafetensorsLoader loader, string prefix)
    {
        var g = loader.ReadF32($"{prefix}.weight_g");
        var v = loader.ReadF32($"{prefix}.weight_v");
        int[] vShape = loader.GetShape($"{prefix}.weight_v");
        int outCh = vShape[0];
        int perChannel = v.Length / outCh;

        var folded = new float[v.Length];
        for (int o = 0; o < outCh; o++)
        {
            double sumSq = 0;
            int baseIdx = o * perChannel;
            for (int j = 0; j < perChannel; j++) { double vv = v[baseIdx + j]; sumSq += vv * vv; }
            float norm = (float)Math.Sqrt(sumSq);
            float scale = norm > 1e-12f ? g[o] / norm : 0f;
            for (int j = 0; j < perChannel; j++) folded[baseIdx + j] = v[baseIdx + j] * scale;
        }
        return folded;
    }
}

public sealed class EncodecLstmWeights
{
    public required float[] WeightIhL0 { get; init; }
    public required float[] WeightHhL0 { get; init; }
    public required float[] BiasIhL0 { get; init; }
    public required float[] BiasHhL0 { get; init; }
    public required float[] WeightIhL1 { get; init; }
    public required float[] WeightHhL1 { get; init; }
    public required float[] BiasIhL1 { get; init; }
    public required float[] BiasHhL1 { get; init; }
}

public sealed class EncodecUpsampleStageWeights
{
    public required float[] UpsampleWeight { get; init; } // ConvTranspose1d, [inCh, outCh, K] flat
    public required float[] UpsampleBias { get; init; }
    public required float[] ResBlockConv0Weight { get; init; } // dim -> dim/compress, k=3
    public required float[] ResBlockConv0Bias { get; init; }
    public required float[] ResBlockConv1Weight { get; init; } // dim/compress -> dim, k=1
    public required float[] ResBlockConv1Bias { get; init; }
}
