using System.Text.Json;

namespace OpenTail.Stingray.Audio.Citrinet;

/// <summary>Real per-conv weights, optionally with BatchNorm folded in at load time -- same real
/// convention as `OpenTail.Stingray.Audio.MarbleNet.MarbleNetConvBn`, confirmed identical in
/// `citrinet_asr/runtime.cpp`'s `load_backend_weights`: depthwise convs are BN-free plain convs,
/// pointwise/plain/residual convs fold their BN in.</summary>
public sealed class CitrinetConvBn : JasperConv1dWeights
{

    public static CitrinetConvBn LoadWithBn(
        Func<string, float[]> get, string convPrefix, string bnPrefix,
        int inChannels, int outChannels, int kernel, int stride, int dilation, int padding, bool depthwise)
    {
        const float bnEps = 1e-3f;
        var weight = get($"{convPrefix}.weight");
        var bnWeight = get($"{bnPrefix}.weight");
        var bnBias = get($"{bnPrefix}.bias");
        var runningMean = get($"{bnPrefix}.running_mean");
        var runningVar = get($"{bnPrefix}.running_var");

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
            foldedBias[c] = bias0;
        }

        return new CitrinetConvBn
        {
            InChannels = inChannels, OutChannels = outChannels, Kernel = kernel, Stride = stride,
            Dilation = dilation, Padding = padding, Depthwise = depthwise,
            Weight = foldedWeight, Bias = foldedBias,
        };
    }

    public static CitrinetConvBn LoadPlain(
        Func<string, float[]> get, string convPrefix,
        int inChannels, int outChannels, int kernel, int stride, int dilation, int padding, bool depthwise, bool hasBias, string? biasPrefix = null)
    {
        var weight = get($"{convPrefix}.weight");
        var bias = hasBias ? get(biasPrefix ?? $"{convPrefix}.bias") : new float[outChannels];
        return new CitrinetConvBn
        {
            InChannels = inChannels, OutChannels = outChannels, Kernel = kernel, Stride = stride,
            Dilation = dilation, Padding = padding, Depthwise = depthwise,
            Weight = weight, Bias = bias,
        };
    }
}

public sealed class CitrinetSqueezeExcite
{
    public required CitrinetConvBn Fc1 { get; init; } // channels -> hidden
    public required CitrinetConvBn Fc2 { get; init; } // hidden -> channels
}

/// <summary>One separable-conv repeat: BN-free depthwise then BN-folded pointwise (real, same
/// per-repeat BN-ownership convention as MarbleNet).</summary>
public sealed class CitrinetSeparableRepeat
{
    public required CitrinetConvBn Depthwise { get; init; }
    public required CitrinetConvBn Pointwise { get; init; }
}

public sealed class CitrinetJasperBlock
{
    public required bool Separable { get; init; }
    public CitrinetSeparableRepeat[] SeparableRepeats { get; init; } = [];
    public CitrinetConvBn[] ConvRepeats { get; init; } = [];
    public CitrinetSqueezeExcite? SqueezeExcite { get; init; }
    public CitrinetConvBn? ResidualConvBn { get; init; }
}

public sealed record CitrinetJasperBlockConfig(
    int Filters, int Kernel, int Repeat, int Stride, int Dilation,
    bool Residual, string? ResidualMode, bool Separable, bool Se, int SeReductionRatio);

public sealed class CitrinetAsrWeights
{
    public const int SampleRate = 16000;
    public const int NMels = 80;
    public const int NFft = 512;
    public const int WinLength = 400;
    public const int HopLength = 160;

    public required int PadTo { get; init; }
    public required int VocabSize { get; init; }
    public required int NumClasses { get; init; }
    public required int BlankId { get; init; }
    public required float[] Window { get; init; }
    public required float[] MelFilterbank { get; init; }
    public required CitrinetJasperBlock[] Blocks { get; init; }
    public required float[] DecoderWeight { get; init; }
    public required float[] DecoderBias { get; init; }
    public required int OutputStride { get; init; }

    /// <summary>Real config, parsed directly from the checkpoint's embedded
    /// `citrinet_256_config.json` (not hand-transcribed -- 23 real Jasper blocks, too many/too
    /// error-prone to type by hand, per `assets.cpp`'s `parse_jasper_config`).</summary>
    public static CitrinetAsrWeights Load(Func<string, float[]> get, string configJson)
    {
        using var doc = JsonDocument.Parse(configJson);
        var root = doc.RootElement;

        int padTo = root.GetProperty("pad_to").GetInt32();
        int vocabSize = root.GetProperty("vocab_size").GetInt32();
        int numClasses = root.GetProperty("num_classes").GetInt32();
        int blankId = root.GetProperty("blank_id").GetInt32();

        var blockConfigs = new List<CitrinetJasperBlockConfig>();
        foreach (var b in root.GetProperty("jasper").EnumerateArray())
        {
            blockConfigs.Add(new CitrinetJasperBlockConfig(
                Filters: b.GetProperty("filters").GetInt32(),
                Kernel: b.GetProperty("kernel").GetInt32(),
                Repeat: b.GetProperty("repeat").GetInt32(),
                Stride: b.GetProperty("stride").GetInt32(),
                Dilation: b.GetProperty("dilation").GetInt32(),
                Residual: b.GetProperty("residual").GetBoolean(),
                ResidualMode: b.TryGetProperty("residual_mode", out var rm) && rm.ValueKind == JsonValueKind.String ? rm.GetString() : null,
                Separable: b.GetProperty("separable").GetBoolean(),
                Se: b.TryGetProperty("se", out var se) && se.GetBoolean(),
                SeReductionRatio: b.TryGetProperty("se_reduction_ratio", out var srr) ? srr.GetInt32() : 8));
        }

        var window = get("preprocessor.featurizer.window");
        var fb = get("preprocessor.featurizer.fb");

        var blocks = new CitrinetJasperBlock[blockConfigs.Count];
        int inChannels = NMels;
        int outputStride = 1;
        for (int bIdx = 0; bIdx < blockConfigs.Count; bIdx++)
        {
            var cfg = blockConfigs[bIdx];
            outputStride *= cfg.Stride;
            string blockPrefix = $"encoder.encoder.{bIdx}";

            CitrinetSeparableRepeat[] separableRepeats = [];
            CitrinetConvBn[] convRepeats = [];
            if (cfg.Separable)
            {
                separableRepeats = new CitrinetSeparableRepeat[cfg.Repeat];
                int repeatIn = inChannels;
                for (int r = 0; r < cfg.Repeat; r++)
                {
                    int idx = r * 5;
                    int stride = r + 1 == cfg.Repeat ? cfg.Stride : 1;
                    int padding = cfg.Dilation * (cfg.Kernel - 1) / 2;
                    var depthwise = CitrinetConvBn.LoadPlain(
                        get, $"{blockPrefix}.mconv.{idx}.conv",
                        repeatIn, repeatIn, cfg.Kernel, stride, cfg.Dilation, padding, depthwise: true, hasBias: false);
                    var pointwise = CitrinetConvBn.LoadWithBn(
                        get, $"{blockPrefix}.mconv.{idx + 1}.conv", $"{blockPrefix}.mconv.{idx + 2}",
                        repeatIn, cfg.Filters, 1, 1, 1, 0, depthwise: false);
                    separableRepeats[r] = new CitrinetSeparableRepeat { Depthwise = depthwise, Pointwise = pointwise };
                    repeatIn = cfg.Filters;
                }
            }
            else
            {
                convRepeats = new CitrinetConvBn[cfg.Repeat];
                for (int r = 0; r < cfg.Repeat; r++)
                {
                    int idx = r * 3;
                    int stride = r + 1 == cfg.Repeat ? cfg.Stride : 1;
                    int padding = cfg.Dilation * (cfg.Kernel - 1) / 2;
                    convRepeats[r] = CitrinetConvBn.LoadWithBn(
                        get, $"{blockPrefix}.mconv.{idx}.conv", $"{blockPrefix}.mconv.{idx + 1}",
                        inChannels, cfg.Filters, cfg.Kernel, stride, cfg.Dilation, padding, depthwise: false);
                }
            }

            CitrinetSqueezeExcite? se = null;
            if (cfg.Se)
            {
                int seIndex = cfg.Repeat == 1 ? 3 : cfg.Repeat * 5 - 2;
                int hidden = Math.Max(1, cfg.Filters / cfg.SeReductionRatio);
                var fc1 = CitrinetConvBn.LoadPlain(
                    get, $"{blockPrefix}.mconv.{seIndex}.fc.0",
                    cfg.Filters, hidden, 1, 1, 1, 0, depthwise: false, hasBias: false);
                var fc2 = CitrinetConvBn.LoadPlain(
                    get, $"{blockPrefix}.mconv.{seIndex}.fc.2",
                    hidden, cfg.Filters, 1, 1, 1, 0, depthwise: false, hasBias: false);
                se = new CitrinetSqueezeExcite { Fc1 = fc1, Fc2 = fc2 };
            }

            CitrinetConvBn? residual = null;
            if (cfg.Residual)
            {
                int residualStride = cfg.ResidualMode == "stride_add" ? cfg.Stride : 1;
                residual = CitrinetConvBn.LoadWithBn(
                    get, $"{blockPrefix}.res.0.0.conv", $"{blockPrefix}.res.0.1",
                    inChannels, cfg.Filters, 1, residualStride, 1, 0, depthwise: false);
            }

            blocks[bIdx] = new CitrinetJasperBlock
            {
                Separable = cfg.Separable,
                SeparableRepeats = separableRepeats,
                ConvRepeats = convRepeats,
                SqueezeExcite = se,
                ResidualConvBn = residual,
            };
            inChannels = cfg.Filters;
        }

        return new CitrinetAsrWeights
        {
            PadTo = padTo,
            VocabSize = vocabSize,
            NumClasses = numClasses,
            BlankId = blankId,
            Window = window,
            MelFilterbank = fb,
            Blocks = blocks,
            DecoderWeight = get("decoder.decoder_layers.0.weight"),
            DecoderBias = get("decoder.decoder_layers.0.bias"),
            OutputStride = outputStride,
        };
    }
}
