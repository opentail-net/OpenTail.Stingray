namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>Real VibeVoice tokenizer config fields (`acoustic_tokenizer_config`/
/// `semantic_tokenizer_config` in the checkpoint's `config.json`), parsed the same way
/// `assets.cpp`'s `parse_tokenizer_config` does (not guessed). Encoder-relevant fields only --
/// this pipeline only ever ENCODES audio for ASR (never decodes back to audio, unlike the
/// acoustic tokenizer's TTS-side use), so decoder-side fields are omitted.</summary>
public sealed class VibeVoiceTokenizerConfig
{
    public required int Channels { get; init; }
    public required int VaeDim { get; init; }
    public required int EncoderNFilters { get; init; }
    public required int[] EncoderRatios { get; init; } // real config order (largest-stride-first); reversed internally to match the reference
    public required int[] EncoderDepths { get; init; } // parsed from "N-N-N..." string
    public required bool DisableLastNorm { get; init; }
    public required float LayerNormEps { get; init; }
    /// <summary>Real `fix_std` -- only meaningful for the acoustic tokenizer (Gaussian VAE
    /// reparameterization scale, see <see cref="VibeVoiceAcousticLatentSampler"/>); unused by the
    /// semantic tokenizer.</summary>
    public float FixStd { get; init; }
}

/// <summary>One VibeVoice tokenizer encoder's real weights (acoustic OR semantic -- same
/// architecture, different config/checkpoint prefix), ported from `speech_tokenizer.cpp`'s
/// `load_encoder` (not guessed). Real structure: `downsample_layers[0]` is a stride-1 causal
/// Conv1d (kernel 7, channels-&gt;encoder_n_filters); `downsample_layers[1..]` are stride-`ratio`
/// causal Conv1ds (kernel `ratio*2`) doubling channel width each time; each stage
/// `stages[i]` is `EncoderDepths[i]` real ConvNeXt-1D blocks (<see cref="VibeVoiceConvNeXtBlock"/>)
/// at that stage's channel width; an optional final channel RMSNorm; then a stride-1 causal
/// Conv1d head (kernel 7) projecting to `VaeDim`.</summary>
public sealed class VibeVoiceTokenizerEncoderWeights
{
    public required float[][][][] DownsampleWeights { get; init; } // [stage][outCh][inCh][kernel]
    public required float[][] DownsampleBiases { get; init; } // [stage][outCh]
    public required int[] DownsampleStrides { get; init; }
    public required VibeVoiceConvNeXtBlockWeights[][] Stages { get; init; } // [stage][block]
    public float[]? FinalNormWeight { get; init; }
    public required float[][][] HeadWeight { get; init; } // [outCh=VaeDim][inCh][kernel]
    public required float[] HeadBias { get; init; }

    public static VibeVoiceTokenizerEncoderWeights Load(VibeVoiceTokenizerConfig config, string prefix, Func<string, float[]> get)
    {
        const int kernelSize = 7;
        const int lastKernelSize = 7;

        // Real reference: ratios come from config in one order, reversed once at load time.
        var ratios = (int[])config.EncoderRatios.Clone();
        Array.Reverse(ratios);
        int stageCount = config.EncoderDepths.Length;
        if (stageCount != ratios.Length + 1)
            throw new InvalidDataException("VibeVoice tokenizer encoder depths/ratios mismatch.");

        var downsampleWeights = new float[stageCount][][][];
        var downsampleBiases = new float[stageCount][];
        var downsampleStrides = new int[stageCount];

        downsampleWeights[0] = LoadConv1dWeight(get, $"{prefix}.downsample_layers.0.0.conv.conv", config.EncoderNFilters, config.Channels, kernelSize);
        downsampleBiases[0] = get($"{prefix}.downsample_layers.0.0.conv.conv.bias");
        downsampleStrides[0] = 1;

        for (int i = 0; i < ratios.Length; i++)
        {
            int inCh = config.EncoderNFilters * (1 << i);
            int outCh = config.EncoderNFilters * (1 << (i + 1));
            int kernel = ratios[i] * 2;
            string p = $"{prefix}.downsample_layers.{i + 1}.0.conv.conv";
            downsampleWeights[i + 1] = LoadConv1dWeight(get, p, outCh, inCh, kernel);
            downsampleBiases[i + 1] = get($"{p}.bias");
            downsampleStrides[i + 1] = ratios[i];
        }

        var stages = new VibeVoiceConvNeXtBlockWeights[stageCount][];
        for (int stage = 0; stage < stageCount; stage++)
        {
            int channels = config.EncoderNFilters * (1 << stage);
            int depth = config.EncoderDepths[stage];
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

        int finalChannels = config.EncoderNFilters * (1 << (stageCount - 1));
        float[]? finalNorm = config.DisableLastNorm ? null : get($"{prefix}.norm.weight");

        var headWeight = LoadConv1dWeight(get, $"{prefix}.head.conv.conv", config.VaeDim, finalChannels, lastKernelSize);
        var headBias = get($"{prefix}.head.conv.conv.bias");

        return new VibeVoiceTokenizerEncoderWeights
        {
            DownsampleWeights = downsampleWeights,
            DownsampleBiases = downsampleBiases,
            DownsampleStrides = downsampleStrides,
            Stages = stages,
            FinalNormWeight = finalNorm,
            HeadWeight = headWeight,
            HeadBias = headBias,
        };
    }

    private static float[][][] LoadConv1dWeight(Func<string, float[]> get, string prefix, int outCh, int inCh, int kernel)
    {
        var flat = get($"{prefix}.weight"); // [outCh, inCh, kernel] row-major
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

    private static float[][] LoadDepthwiseConv1dWeight(Func<string, float[]> get, string prefix, int channels, int kernel)
    {
        var flat = get($"{prefix}.weight"); // [channels, 1, kernel] row-major
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
