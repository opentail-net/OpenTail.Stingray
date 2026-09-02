
namespace OpenTail.Stingray.Audio.AudioGen;

/// <summary>
/// Weight loader for AudioGen's audio codec: a SEPARATELY-TRAINED 16kHz EnCodec for environmental
/// sound (real native AudioCraft "compression" checkpoint, converted to safetensors -- see
/// docs/063-audiogen-implementation-plan.md), NOT the same weights as MusicGen's 32kHz music
/// codec despite sharing the identical layer skeleton (see
/// <see cref="Primitives.EncodecDecoderKernels"/>'s doc comment). Only the DECODER path is
/// loaded.
///
/// <para>Real tensor layout confirmed against `compression_state_dict.bin`'s own embedded
/// training config (`seanet.ratios: [8,5,4,2]`, `seanet.n_filters: 64`,
/// `seanet.n_residual_layers: 1`, `seanet.compress: 2`, `seanet.dimension: 128`, `rvq.bins:
/// 2048`, `decoder.trim_right_ratio: 1.0`, `decoder.final_activation: null`) -- structurally
/// identical to MusicGen's EnCodec, but native AudioCraft tensor names differ from HF's remap:
/// `decoder.model.{i}.conv.conv.weight_g/weight_v` (an extra `.conv.conv.` nesting from
/// `SConv1d` wrapping `NormConv1d` wrapping raw `nn.Conv1d`), `decoder.model.{i}.convtr.convtr.*`
/// for the transpose convs, `quantizer.vq.layers.{q}._codebook.embed` for the RVQ codebooks
/// (leading underscore, real `ResidualVectorQuantization`/`VectorQuantization` naming).</para>
/// </summary>
public static class AudioGenEncodecDecoderWeights
{
    public static Primitives.EncodecDecoderWeights Load(SafetensorsLoader loader)
    {
        var codebooks = new float[AudioGenConfig.NumCodebooks][];
        for (int q = 0; q < AudioGenConfig.NumCodebooks; q++)
            codebooks[q] = loader.ReadF32($"quantizer.vq.layers.{q}._codebook.embed");

        var lstm = new EncodecLstmWeights
        {
            WeightIhL0 = loader.ReadF32("decoder.model.1.lstm.weight_ih_l0"),
            WeightHhL0 = loader.ReadF32("decoder.model.1.lstm.weight_hh_l0"),
            BiasIhL0 = loader.ReadF32("decoder.model.1.lstm.bias_ih_l0"),
            BiasHhL0 = loader.ReadF32("decoder.model.1.lstm.bias_hh_l0"),
            WeightIhL1 = loader.ReadF32("decoder.model.1.lstm.weight_ih_l1"),
            WeightHhL1 = loader.ReadF32("decoder.model.1.lstm.weight_hh_l1"),
            BiasIhL1 = loader.ReadF32("decoder.model.1.lstm.bias_ih_l1"),
            BiasHhL1 = loader.ReadF32("decoder.model.1.lstm.bias_hh_l1"),
        };

        int[] upsampleLayerIndex = [3, 6, 9, 12];
        int[] blockLayerIndex = [4, 7, 10, 13];
        var stages = new EncodecUpsampleStageWeights[4];
        for (int i = 0; i < 4; i++)
        {
            string up = $"decoder.model.{upsampleLayerIndex[i]}.convtr.convtr";
            string blk = $"decoder.model.{blockLayerIndex[i]}.block";
            stages[i] = new EncodecUpsampleStageWeights
            {
                UpsampleWeight = Primitives.EncodecDecoderWeights.FoldConvWeight(loader, $"{up}.weight_g", $"{up}.weight_v"),
                UpsampleBias = loader.ReadF32($"{up}.bias"),
                ResBlockConv0Weight = Primitives.EncodecDecoderWeights.FoldConvWeight(loader, $"{blk}.1.conv.conv.weight_g", $"{blk}.1.conv.conv.weight_v"),
                ResBlockConv0Bias = loader.ReadF32($"{blk}.1.conv.conv.bias"),
                ResBlockConv1Weight = Primitives.EncodecDecoderWeights.FoldConvWeight(loader, $"{blk}.3.conv.conv.weight_g", $"{blk}.3.conv.conv.weight_v"),
                ResBlockConv1Bias = loader.ReadF32($"{blk}.3.conv.conv.bias"),
            };
        }

        return new Primitives.EncodecDecoderWeights
        {
            LatentDim = AudioGenConfig.EncodecHiddenSize,
            Ratios = AudioGenConfig.EncodecUpsamplingRatios,
            ChannelsPerStage = Primitives.EncodecDecoderWeights.DefaultChannelsPerStage(AudioGenConfig.EncodecNumFilters, AudioGenConfig.EncodecUpsamplingRatios.Length),
            Codebooks = codebooks,
            InitConvWeight = Primitives.EncodecDecoderWeights.FoldConvWeight(loader, "decoder.model.0.conv.conv.weight_g", "decoder.model.0.conv.conv.weight_v"),
            InitConvBias = loader.ReadF32("decoder.model.0.conv.conv.bias"),
            Lstm = lstm,
            Stages = stages,
            OutConvWeight = Primitives.EncodecDecoderWeights.FoldConvWeight(loader, "decoder.model.15.conv.conv.weight_g", "decoder.model.15.conv.conv.weight_v"),
            OutConvBias = loader.ReadF32("decoder.model.15.conv.conv.bias"),
        };
    }
}
