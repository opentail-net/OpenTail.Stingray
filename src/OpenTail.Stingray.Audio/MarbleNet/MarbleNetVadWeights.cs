namespace OpenTail.Stingray.Audio.MarbleNet;

/// <summary>Real per-conv weights with BatchNorm folded in at load time (no separate BN pass at
/// inference), matching `runtime.cpp`'s `make_backend_conv_bn`: `scale = bn.weight /
/// sqrt(running_var + eps)`, `folded_bias = bn.bias - running_mean * scale`, then
/// `weight[c] *= scale[c]`, `bias[c] = origBias[c] * scale[c] + folded_bias[c]` (origBias is
/// always 0 here -- every MarbleNet conv that feeds a BN layer is itself bias-free).</summary>
public sealed class MarbleNetConvBn : JasperConv1dWeights
{

    public static MarbleNetConvBn Load(
        SafetensorsLoader src, string convPrefix, string bnPrefix,
        int inChannels, int outChannels, int kernel, int stride, int dilation, int padding, bool depthwise)
    {
        const float bnEps = 1e-3f;
        var weight = src.ReadF32($"{convPrefix}.weight");
        var bnWeight = src.ReadF32($"{bnPrefix}.weight");
        var bnBias = src.ReadF32($"{bnPrefix}.bias");
        var runningMean = src.ReadF32($"{bnPrefix}.running_mean");
        var runningVar = src.ReadF32($"{bnPrefix}.running_var");

        int valuesPerOutChannel = weight.Length / outChannels;
        var foldedWeight = new float[weight.Length];
        var foldedBias = new float[outChannels];
        for (int c = 0; c < outChannels; c++)
        {
            float scale = bnWeight[c] / MathF.Sqrt(runningVar[c] + bnEps);
            float bias0 = bnBias[c] - runningMean[c] * scale;
            int baseIdx = c * valuesPerOutChannel;
            for (int i = 0; i < valuesPerOutChannel; i++)
                foldedWeight[baseIdx + i] = weight[baseIdx + i] * scale;
            foldedBias[c] = bias0; // origBias == 0
        }

        return new MarbleNetConvBn
        {
            InChannels = inChannels,
            OutChannels = outChannels,
            Kernel = kernel,
            Stride = stride,
            Dilation = dilation,
            Padding = padding,
            Depthwise = depthwise,
            Weight = foldedWeight,
            Bias = foldedBias,
        };
    }

    /// <summary>Real, deliberately BN-FREE plain conv (`runtime.cpp`'s `make_backend_conv`, NOT
    /// `make_backend_conv_bn`) -- used ONLY for a separable Jasper repeat's depthwise conv, which
    /// the reference loads with NO batch-norm folding and NO bias at all (the BN that follows a
    /// repeat is folded into the POINTWISE conv only, never the depthwise one -- confirmed by
    /// reading `load_backend_weights`'s real per-repeat call pair, not guessed).</summary>
    public static MarbleNetConvBn LoadPlainNoBias(
        SafetensorsLoader src, string convPrefix,
        int inChannels, int outChannels, int kernel, int stride, int dilation, int padding, bool depthwise)
    {
        var weight = src.ReadF32($"{convPrefix}.weight");
        return new MarbleNetConvBn
        {
            InChannels = inChannels,
            OutChannels = outChannels,
            Kernel = kernel,
            Stride = stride,
            Dilation = dilation,
            Padding = padding,
            Depthwise = depthwise,
            Weight = weight,
            Bias = new float[outChannels],
        };
    }
}

/// <summary>One real `mconv` repeat: a BN-FREE depthwise conv then a BN-folded pointwise (1x1)
/// conv, for a separable Jasper block (`runtime.cpp`'s `BackendSeparableConvBn` /
/// `load_backend_weights`'s real `make_backend_conv(depthwise)` + `make_backend_conv_bn(pointwise,
/// repeat.bn)` pair -- the repeat's single BN layer folds into the pointwise conv only).</summary>
public sealed class MarbleNetSeparableRepeat
{
    public required MarbleNetConvBn Depthwise { get; init; }
    public required MarbleNetConvBn Pointwise { get; init; }
}

/// <summary>Real Jasper block, ported from `runtime.cpp`'s `BackendJasperBlockWeights` /
/// `load_marblenet_weights`. Separable blocks alternate depthwise+pointwise+BN per repeat
/// (real safetensors index stride of 5: `mconv.{5r}`=depthwise conv, `mconv.{5r+1}`=pointwise
/// conv, `mconv.{5r+2}`=BN); non-separable blocks use one plain conv+BN per repeat (stride 3:
/// `mconv.{3r}`=conv, `mconv.{3r+1}`=BN). An optional 1x1-conv+BN residual projects the block's
/// ORIGINAL input to the output channel count before the final add.</summary>
public sealed class MarbleNetJasperBlock
{
    public required bool Separable { get; init; }
    public MarbleNetSeparableRepeat[] SeparableRepeats { get; init; } = [];
    public MarbleNetConvBn[] ConvRepeats { get; init; } = [];
    public MarbleNetConvBn? ResidualConvBn { get; init; }
}

public sealed class MarbleNetVadWeights
{
    public const int SampleRate = 16000;
    public const int NMels = 80;
    public const int NFft = 512;
    public const int WinLength = 400;
    public const int HopLength = 160;
    public const int PadTo = 2;
    public const int NumClasses = 2;

    public required float[] Window { get; init; } // [WinLength]
    public required float[] MelFilterbank { get; init; } // [NMels * (NFft/2+1)]
    public required MarbleNetJasperBlock[] Blocks { get; init; }
    /// <summary>Real decoder: a per-frame Linear (equivalently a kernel-1 conv with real bias --
    /// `decoder.layer0.weight`/`.bias`), `[NumClasses][128]`/`[NumClasses]`.</summary>
    public required float[] DecoderWeight { get; init; }
    public required float[] DecoderBias { get; init; }
    public required int OutputStride { get; init; }

    private static readonly (int Filters, int Kernel, int Repeat, int Stride, int Dilation, bool Residual, bool Separable)[] JasperConfig =
    [
        (128, 11, 1, 2, 1, false, true),
        (64, 13, 2, 1, 1, true, true),
        (64, 15, 2, 1, 1, true, true),
        (64, 17, 2, 1, 1, true, true),
        (128, 29, 1, 1, 2, false, true),
        (128, 1, 1, 1, 1, false, false),
    ];

    public static MarbleNetVadWeights Load(SafetensorsLoader src)
    {
        var window = src.ReadF32("preprocessor.featurizer.window");
        var fb = src.ReadF32("preprocessor.featurizer.fb");

        var blocks = new MarbleNetJasperBlock[JasperConfig.Length];
        int inChannels = NMels;
        int outputStride = 1;
        for (int b = 0; b < JasperConfig.Length; b++)
        {
            var cfg = JasperConfig[b];
            outputStride *= cfg.Stride;
            string blockPrefix = $"encoder.encoder.{b}";
            if (cfg.Separable)
            {
                var repeats = new MarbleNetSeparableRepeat[cfg.Repeat];
                int repeatIn = inChannels;
                for (int r = 0; r < cfg.Repeat; r++)
                {
                    int idx = r * 5;
                    int padding = cfg.Dilation * (cfg.Kernel - 1) / 2;
                    var depthwise = MarbleNetConvBn.LoadPlainNoBias(
                        src, $"{blockPrefix}.mconv.{idx}.conv",
                        repeatIn, repeatIn, cfg.Kernel, cfg.Stride, cfg.Dilation, padding, depthwise: true);
                    var pointwise = MarbleNetConvBn.Load(
                        src, $"{blockPrefix}.mconv.{idx + 1}.conv", $"{blockPrefix}.mconv.{idx + 2}",
                        repeatIn, cfg.Filters, 1, 1, 1, 0, depthwise: false);
                    repeats[r] = new MarbleNetSeparableRepeat { Depthwise = depthwise, Pointwise = pointwise };
                    repeatIn = cfg.Filters;
                }
                MarbleNetConvBn? residual = null;
                if (cfg.Residual)
                {
                    residual = MarbleNetConvBn.Load(
                        src, $"{blockPrefix}.res.0.0.conv", $"{blockPrefix}.res.0.1",
                        inChannels, cfg.Filters, 1, 1, 1, 0, depthwise: false);
                }
                blocks[b] = new MarbleNetJasperBlock { Separable = true, SeparableRepeats = repeats, ResidualConvBn = residual };
            }
            else
            {
                var repeats = new MarbleNetConvBn[cfg.Repeat];
                for (int r = 0; r < cfg.Repeat; r++)
                {
                    int idx = r * 3;
                    int padding = cfg.Dilation * (cfg.Kernel - 1) / 2;
                    repeats[r] = MarbleNetConvBn.Load(
                        src, $"{blockPrefix}.mconv.{idx}.conv", $"{blockPrefix}.mconv.{idx + 1}",
                        inChannels, cfg.Filters, cfg.Kernel, cfg.Stride, cfg.Dilation, padding, depthwise: false);
                }
                MarbleNetConvBn? residual = null;
                if (cfg.Residual)
                {
                    residual = MarbleNetConvBn.Load(
                        src, $"{blockPrefix}.res.0.0.conv", $"{blockPrefix}.res.0.1",
                        inChannels, cfg.Filters, 1, 1, 1, 0, depthwise: false);
                }
                blocks[b] = new MarbleNetJasperBlock { Separable = false, ConvRepeats = repeats, ResidualConvBn = residual };
            }
            inChannels = cfg.Filters;
        }

        return new MarbleNetVadWeights
        {
            Window = window,
            MelFilterbank = fb,
            Blocks = blocks,
            DecoderWeight = src.ReadF32("decoder.layer0.weight"),
            DecoderBias = src.ReadF32("decoder.layer0.bias"),
            OutputStride = outputStride,
        };
    }
}
