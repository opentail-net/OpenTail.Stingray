
namespace OpenTail.Stingray.Audio.MusicGen;

/// <summary>
/// Weight loader for MusicGen's audio codec, EnCodec 32kHz (`audio_encoder.*` prefix in
/// musicgen-small's own `model.safetensors`). Only the DECODER path is loaded -- MusicGen
/// generation never runs EnCodec's own encoder.
///
/// <para>Real tensor layout confirmed against the checkpoint's own safetensors header
/// (2026-09-02): `audio_encoder.quantizer.layers.{0..3}.codebook.embed` (`[2048,128]` per
/// codebook, real RVQ codebook vectors -- NOT `embed_avg`/`cluster_size`, training-only EMA
/// state unused at inference); `audio_encoder.decoder.layers.{i}.*` with `weight_g`/`weight_v`
/// real PyTorch `weight_norm`-wrapped conv pairs throughout. Layer index layout: 0=initial
/// conv(128-&gt;1024,k7), 1=2-layer LSTM, {3,6,9,12}=upsampling ConvTranspose1d (ratios
/// 8,5,4,4), {4,7,10,13}=one residual block each, 15=final conv(64-&gt;1,k7).</para>
///
/// <para><b>DRY pass, 2026-09-02</b>: forward-pass math moved to the shared, ratio-parameterized
/// <see cref="Primitives.EncodecDecoderKernels"/> once AudioGen's separately-trained 16kHz EnCodec
/// turned out to share the identical layer skeleton, differing only in per-stage ratios -- see
/// that class's doc comment.</para>
/// </summary>
public static class MusicGenEncodecDecoderWeights
{
    public static Primitives.EncodecDecoderWeights Load(SafetensorsLoader loader)
    {
        var codebooks = new float[MusicGenConfig.NumCodebooks][];
        for (int q = 0; q < MusicGenConfig.NumCodebooks; q++)
            codebooks[q] = loader.ReadF32($"audio_encoder.quantizer.layers.{q}.codebook.embed");

        var lstm = new EncodecLstmWeights
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
        var stages = new EncodecUpsampleStageWeights[4];
        for (int i = 0; i < 4; i++)
        {
            string up = $"audio_encoder.decoder.layers.{upsampleLayerIndex[i]}.conv";
            string blk = $"audio_encoder.decoder.layers.{blockLayerIndex[i]}.block";
            stages[i] = new EncodecUpsampleStageWeights
            {
                UpsampleWeight = Primitives.EncodecDecoderWeights.FoldConvWeight(loader, $"{up}.weight_g", $"{up}.weight_v"),
                UpsampleBias = loader.ReadF32($"{up}.bias"),
                ResBlockConv0Weight = Primitives.EncodecDecoderWeights.FoldConvWeight(loader, $"{blk}.1.conv.weight_g", $"{blk}.1.conv.weight_v"),
                ResBlockConv0Bias = loader.ReadF32($"{blk}.1.conv.bias"),
                ResBlockConv1Weight = Primitives.EncodecDecoderWeights.FoldConvWeight(loader, $"{blk}.3.conv.weight_g", $"{blk}.3.conv.weight_v"),
                ResBlockConv1Bias = loader.ReadF32($"{blk}.3.conv.bias"),
            };
        }

        return new Primitives.EncodecDecoderWeights
        {
            LatentDim = MusicGenConfig.EncodecHiddenSize,
            Ratios = MusicGenConfig.EncodecUpsamplingRatios,
            ChannelsPerStage = Primitives.EncodecDecoderWeights.DefaultChannelsPerStage(MusicGenConfig.EncodecNumFilters, MusicGenConfig.EncodecUpsamplingRatios.Length),
            Codebooks = codebooks,
            InitConvWeight = Primitives.EncodecDecoderWeights.FoldConvWeight(loader, "audio_encoder.decoder.layers.0.conv.weight_g", "audio_encoder.decoder.layers.0.conv.weight_v"),
            InitConvBias = loader.ReadF32("audio_encoder.decoder.layers.0.conv.bias"),
            Lstm = lstm,
            Stages = stages,
            OutConvWeight = Primitives.EncodecDecoderWeights.FoldConvWeight(loader, "audio_encoder.decoder.layers.15.conv.weight_g", "audio_encoder.decoder.layers.15.conv.weight_v"),
            OutConvBias = loader.ReadF32("audio_encoder.decoder.layers.15.conv.bias"),
        };
    }
}
