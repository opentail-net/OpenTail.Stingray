namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>Real weights for VibeVoice's acoustic-latent-to-waveform tokenizer DECODER, ported
/// from `tokenizer_audio.cpp`'s `load_decoder` (not guessed) -- the TTS-side counterpart to
/// <see cref="VibeVoiceTokenizerEncoderWeights"/> (ASR only ever encodes; TTS only ever decodes,
/// but both directions share the exact same real `VibeVoiceConvNeXtBlock`-based stage
/// architecture and the SAME real causal-conv padding convention, `extra_padding_for_conv1d`).
/// Real structure: a stem stride-1 causal Conv1d (`VaeDim -&gt; topChannels`, kernel 7), then
/// `ratios.Length` stages each starting with a causal `ConvTranspose1d` upsample (kernel
/// `ratio*2`, stride `ratio`, real all-trim-from-the-right causal crop -- confirmed real
/// `kTokenizerConvTransposeTrimRightRatio=1.0`, i.e. `padding_left=0`, unlike a symmetric
/// PyTorch `ConvTranspose1d`) followed by that stage's real ConvNeXt blocks; an optional final
/// channel RMSNorm; then a stride-1 causal Conv1d head (kernel 7) projecting to
/// `config.Channels` (the output waveform channel count, 1 for mono).</summary>
public sealed class VibeVoiceTokenizerDecoderWeights
{
    public required float[][][] StemWeight { get; init; } // [topChannels][VaeDim][kernel]
    public required float[] StemBias { get; init; }

    public required float[][][][] UpsampleWeights { get; init; } // [stage][inCh][outCh][kernel] (ConvTranspose1d layout)
    public required float[][] UpsampleBiases { get; init; } // [stage][outCh]
    public required int[] UpsampleRatios { get; init; }

    public required VibeVoiceConvNeXtBlockWeights[][] Stages { get; init; } // [stage][block]
    public float[]? FinalNormWeight { get; init; }

    public required float[][][] HeadWeight { get; init; } // [channels][decoderNFilters][kernel]
    public required float[] HeadBias { get; init; }

    public static VibeVoiceTokenizerDecoderWeights Load(VibeVoiceTokenizerConfig config, int decoderNFilters,
        int[] decoderRatios, int[] decoderDepths, string prefix, Func<string, float[]> get)
    {
        const int kernelSize = 7;
        const int lastKernelSize = 7;

        int stageCount = decoderDepths.Length;
        if (stageCount != decoderRatios.Length + 1)
            throw new InvalidDataException("VibeVoice tokenizer decoder depths/ratios mismatch.");

        int topChannels = decoderNFilters * (1 << (stageCount - 1));
        var stemWeight = LoadConvTranspose1dLayoutAsConv1d(get, $"{prefix}.upsample_layers.0.0.conv.conv", topChannels, config.VaeDim, kernelSize);
        var stemBias = get($"{prefix}.upsample_layers.0.0.conv.conv.bias");

        var upsampleWeights = new float[decoderRatios.Length][][][];
        var upsampleBiases = new float[decoderRatios.Length][];
        for (int i = 0; i < decoderRatios.Length; i++)
        {
            int inCh = decoderNFilters * (1 << (stageCount - 1 - i));
            int outCh = decoderNFilters * (1 << (stageCount - 2 - i));
            int kernel = decoderRatios[i] * 2;
            string p = $"{prefix}.upsample_layers.{i + 1}.0.convtr.convtr";
            upsampleWeights[i] = LoadConvTranspose1dWeight(get, p, inCh, outCh, kernel);
            upsampleBiases[i] = get($"{p}.bias");
        }

        var stages = new VibeVoiceConvNeXtBlockWeights[stageCount][];
        for (int stage = 0; stage < stageCount; stage++)
        {
            int channels = decoderNFilters * (1 << (stageCount - 1 - stage));
            int depth = decoderDepths[stage];
            var blocks = new VibeVoiceConvNeXtBlockWeights[depth];
            for (int block = 0; block < depth; block++)
            {
                string bp = $"{prefix}.stages.{stage}.{block}";
                blocks[block] = new VibeVoiceConvNeXtBlockWeights
                {
                    NormWeight = get($"{bp}.norm.weight"),
                    MixerWeight = LoadDepthwiseConv1dWeight(get, $"{bp}.mixer.conv.conv.conv", channels, kernelSize),
                    MixerBias = get($"{bp}.mixer.conv.conv.conv.bias"),
                    Gamma = get($"{bp}.gamma"),
                    FfnNormWeight = get($"{bp}.ffn_norm.weight"),
                    FfnLinear1Weight = get($"{bp}.ffn.linear1.weight"),
                    FfnLinear1Bias = get($"{bp}.ffn.linear1.bias"),
                    FfnLinear2Weight = get($"{bp}.ffn.linear2.weight"),
                    FfnLinear2Bias = get($"{bp}.ffn.linear2.bias"),
                    FfnGamma = get($"{bp}.ffn_gamma"),
                };
            }
            stages[stage] = blocks;
        }

        float[]? finalNorm = config.DisableLastNorm ? null : get($"{prefix}.norm.weight");

        var headWeight = LoadConvTranspose1dLayoutAsConv1d(get, $"{prefix}.head.conv.conv", config.Channels, decoderNFilters, lastKernelSize);
        var headBias = get($"{prefix}.head.conv.conv.bias");

        return new VibeVoiceTokenizerDecoderWeights
        {
            StemWeight = stemWeight,
            StemBias = stemBias,
            UpsampleWeights = upsampleWeights,
            UpsampleBiases = upsampleBiases,
            UpsampleRatios = decoderRatios,
            Stages = stages,
            FinalNormWeight = finalNorm,
            HeadWeight = headWeight,
            HeadBias = headBias,
        };
    }

    // Real Conv1d weight layout [outCh, inCh, kernel] row-major (stem/head are ordinary Conv1d,
    // not ConvTranspose1d, despite the stem living under the checkpoint's "upsample_layers.0"
    // prefix -- confirmed via load_conv1d being used for both in the real reference).
    private static float[][][] LoadConvTranspose1dLayoutAsConv1d(Func<string, float[]> get, string prefix, int outCh, int inCh, int kernel)
    {
        var flat = get($"{prefix}.weight");
        var w = new float[outCh][][];
        for (int oc = 0; oc < outCh; oc++)
        {
            w[oc] = new float[inCh][];
            for (int ic = 0; ic < inCh; ic++)
            {
                var row = new float[kernel];
                Array.Copy(flat, (oc * inCh + ic) * kernel, row, 0, kernel);
                w[oc][ic] = row;
            }
        }
        return w;
    }

    // Real ConvTranspose1d weight layout [inCh, outCh, kernel] row-major.
    private static float[][][] LoadConvTranspose1dWeight(Func<string, float[]> get, string prefix, int inCh, int outCh, int kernel)
    {
        var flat = get($"{prefix}.weight");
        var w = new float[inCh][][];
        for (int ic = 0; ic < inCh; ic++)
        {
            w[ic] = new float[outCh][];
            for (int oc = 0; oc < outCh; oc++)
            {
                var row = new float[kernel];
                Array.Copy(flat, (ic * outCh + oc) * kernel, row, 0, kernel);
                w[ic][oc] = row;
            }
        }
        return w;
    }

    private static float[][] LoadDepthwiseConv1dWeight(Func<string, float[]> get, string prefix, int channels, int kernel)
    {
        var flat = get($"{prefix}.weight");
        var w = new float[channels][];
        for (int c = 0; c < channels; c++)
        {
            var row = new float[kernel];
            Array.Copy(flat, c * kernel, row, 0, kernel);
            w[c] = row;
        }
        return w;
    }
}
