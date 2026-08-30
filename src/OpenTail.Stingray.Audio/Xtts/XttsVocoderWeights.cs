
namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 vocoder weights: `hifigan_decoder.waveform_decoder`, a real `HifiganGenerator`
/// (`TTS/vocoder/models/hifigan_generator.py`) -- confirmed real construction args from
/// `HifiDecoder.__init__`: `in_channels=1024(decoder_input_dim), out_channels=1,
/// resblock_type="1", resblock_dilation_sizes=[[1,3,5]]*3, resblock_kernel_sizes=[3,7,11],
/// upsample_kernel_sizes=[16,16,4,4], upsample_initial_channel=512, upsample_rates=[8,8,2,2],
/// cond_channels=512(d_vector_dim), conv_pre_weight_norm=False, conv_post_weight_norm=False,
/// conv_post_bias=False, cond_in_each_up_layer=True`.
///
/// <para>Same `ResBlock1` topology (3 conv PAIRS, dilations (1,3,5) cycling regardless of the
/// resblock's own kernel size) already confirmed and implemented for MeloTTS/MMS-TTS -- reuses
/// that established pattern. NEW here: real FiLM-style global speaker conditioning --
/// `cond_layer` (512-&gt;512, applied ONCE right after `conv_pre`) and `conds.N` (512-&gt;chN, applied
/// after EACH upsample stage, `cond_in_each_up_layer=True`) -- both real, plain (non-weight-
/// normed) `Conv1d`s with bias, confirmed from the real source's own construction
/// (`nn.Conv1d(cond_channels, ..., 1)`, no `weight_norm` wrapper unlike `conv_pre`/`ups`/
/// `resblocks`).</para>
///
/// <para>Weight-norm status: `conv_pre`/`conv_post` are ALREADY FUSED (real construction passes
/// `conv_pre_weight_norm=False, conv_post_weight_norm=False` -- plain `.weight`/`.bias`, `conv_post`
/// has NO bias at all, `conv_post_bias=False`). `ups.N`/`resblocks.N.convs1/2.N` ship as real
/// `nn.utils.parametrizations.weight_norm` (`.parametrizations.weight.original0`/`original1`,
/// the newer PyTorch parametrization API) -- folded via the same formula this codebase already
/// uses for CosyVoice's HiFT decoder (`CosyVoiceHiftWeights.GetFoldedConvWeight`).</para>
/// </summary>
public sealed class XttsVocoderWeights
{
    public const int InChannels = 1024;
    public const int OutChannels = 1;
    public const int CondChannels = 512;
    public const int UpsampleInitialChannel = 512;
    public static readonly int[] UpsampleRates = [8, 8, 2, 2];
    public static readonly int[] UpsampleKernelSizes = [16, 16, 4, 4];
    public static readonly int[] ResblockKernelSizes = [3, 7, 11];

    public float[] ConvPreWeight { get; } // [512,1024,7], plain (no weight_norm)
    public float[] ConvPreBias { get; }
    public float[] CondLayerWeight { get; } // conv1x1, 512->512, WITH bias
    public float[] CondLayerBias { get; }

    public float[][] UpsWeight { get; } // folded, [numStages]
    public float[][] UpsBias { get; }
    public float[][] CondsWeight { get; } // conv1x1 per upsample stage, 512->chN, WITH bias
    public float[][] CondsBias { get; }

    public XttsVocResBlockWeights[] ResBlocks { get; } // numStages * numKernels

    public float[] ConvPostWeight { get; } // plain, NO bias (conv_post_bias=False)

    public XttsVocoderWeights(SafetensorsLoader loader, string prefix)
    {
        ConvPreWeight = loader.ReadF32($"{prefix}.conv_pre.weight");
        ConvPreBias = loader.ReadF32($"{prefix}.conv_pre.bias");
        CondLayerWeight = loader.ReadF32($"{prefix}.cond_layer.weight");
        CondLayerBias = loader.ReadF32($"{prefix}.cond_layer.bias");

        int numStages = UpsampleRates.Length;
        UpsWeight = new float[numStages][];
        UpsBias = new float[numStages][];
        CondsWeight = new float[numStages][];
        CondsBias = new float[numStages][];
        for (int i = 0; i < numStages; i++)
        {
            UpsWeight[i] = FoldConvWeight(loader, $"{prefix}.ups.{i}");
            UpsBias[i] = loader.ReadF32($"{prefix}.ups.{i}.bias");
            CondsWeight[i] = loader.ReadF32($"{prefix}.conds.{i}.weight");
            CondsBias[i] = loader.ReadF32($"{prefix}.conds.{i}.bias");
        }

        int numKernels = ResblockKernelSizes.Length;
        ResBlocks = new XttsVocResBlockWeights[numStages * numKernels];
        for (int i = 0; i < ResBlocks.Length; i++)
            ResBlocks[i] = new XttsVocResBlockWeights(loader, $"{prefix}.resblocks.{i}");

        ConvPostWeight = loader.ReadF32($"{prefix}.conv_post.weight");
    }

    /// <summary>Real `nn.utils.parametrizations.weight_norm` fold: `weight = original0[outCh] * original1[outCh,:,:] / ||original1[outCh,:,:]||_2` (dim=0 default) -- same math already used in `CosyVoiceHiftWeights.GetFoldedConvWeight`.</summary>
    internal static float[] FoldConvWeight(SafetensorsLoader loader, string prefix)
    {
        var g = loader.ReadF32($"{prefix}.parametrizations.weight.original0");
        var v = loader.ReadF32($"{prefix}.parametrizations.weight.original1");
        int[] vShape = loader.GetShape($"{prefix}.parametrizations.weight.original1");
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

/// <summary>Real ResBlock1: 3 conv PAIRS (convs1[j] at dilation (1,3,5), convs2[j] at dilation 1), both weight-normed (folded).</summary>
public sealed class XttsVocResBlockWeights
{
    public float[][] Convs1Weight { get; } = new float[3][];
    public float[][] Convs1Bias { get; } = new float[3][];
    public float[][] Convs2Weight { get; } = new float[3][];
    public float[][] Convs2Bias { get; } = new float[3][];

    public XttsVocResBlockWeights(SafetensorsLoader loader, string prefix)
    {
        for (int i = 0; i < 3; i++)
        {
            Convs1Weight[i] = XttsVocoderWeights.FoldConvWeight(loader, $"{prefix}.convs1.{i}");
            Convs1Bias[i] = loader.ReadF32($"{prefix}.convs1.{i}.bias");
            Convs2Weight[i] = XttsVocoderWeights.FoldConvWeight(loader, $"{prefix}.convs2.{i}");
            Convs2Bias[i] = loader.ReadF32($"{prefix}.convs2.{i}.bias");
        }
    }
}
