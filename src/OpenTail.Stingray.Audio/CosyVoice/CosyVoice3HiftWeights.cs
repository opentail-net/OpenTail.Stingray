using System.IO;
using OpenTail.Stingray.Audio.Primitives;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// Real weights for CosyVoice3's HiFT vocoder, read directly from the bundled
/// `models/cosyvoice3/CosyVoice3-2512_F16.gguf`. Architecturally close to
/// <see cref="CosyVoiceHiftWeights"/> (CosyVoice2's, same stage shapes, same resblock/
/// source-down structure) but with two real, confirmed differences (see
/// docs/audio-review-progress.md's "CosyVoice3 flow/HiFT" entry, tensor shapes cross-checked
/// again here via `list-tensors` before writing this loader): `conv_pre` is kernel **5**
/// (`[5,80,512]` GGUF-displayed = native `[512,80,5]`), not CosyVoice2's kernel 7; and every
/// conv tensor here is already plain/pre-fused (no `parametrizations.weight.original0/1`
/// split anywhere in this GGUF) -- <see cref="CosyVoiceHiftWeights.GetFoldedConvWeight"/>'s
/// weight-norm fold must NOT be applied to this checkpoint, tensors are read directly.
/// </summary>
public sealed class CosyVoice3HiftWeights : IDisposable, IHiFTVocoderWeights
{
    public int[] UpsampleRates { get; } = [8, 5, 3];
    public int[] UpsampleKernels { get; } = [16, 11, 7];
    public int[] ResblockKernels { get; } = [3, 7, 11];
    public int[] SourceResblockKernels { get; } = [7, 7, 11];
    public int BaseChannels { get; } = 512;
    public int NbHarmonics { get; } = 8;
    public int IstftNFft { get; } = 16;
    public int IstftHopLen { get; } = 4;
    public int SampleRate { get; } = 24000;
    public int ConvPreKernel { get; } = 5;
    public int ConvPostKernel { get; } = 7;

    public float[] ConvPreWeight { get; }
    public float[] ConvPreBias { get; }
    public float[] ConvPostWeight { get; }
    public float[] ConvPostBias { get; }

    public float[][] UpWeight { get; }
    public float[][] UpBias { get; }
    public float[][] SourceDownWeight { get; }
    public float[][] SourceDownBias { get; }
    public CosyVoice3HifiResBlockWeights[] SourceResBlocks { get; }
    public CosyVoice3HifiResBlockWeights[] ResBlocks { get; }
    IHifiResBlockWeights[] IHiFTVocoderWeights.SourceResBlocks => (IHifiResBlockWeights[])SourceResBlocks;
    IHifiResBlockWeights[] IHiFTVocoderWeights.ResBlocks => (IHifiResBlockWeights[])ResBlocks;

    public CosyVoice3F0PredictorWeights F0Predictor { get; }
    IF0PredictorWeights IHiFTVocoderWeights.F0Predictor => F0Predictor;
    public float[] MSourceLinearWeight { get; }
    public float[] MSourceLinearBias { get; }

    private readonly GgufModel _model;

    public CosyVoice3HiftWeights(string ggufPath)
    {
        _model = GgufModel.Open(ggufPath);

        ConvPreWeight = GetTensor("conv_pre.weight");
        ConvPreBias = GetTensor("conv_pre.bias");
        ConvPostWeight = GetTensor("conv_post.weight");
        ConvPostBias = GetTensor("conv_post.bias");

        int numStages = UpsampleRates.Length;
        UpWeight = new float[numStages][];
        UpBias = new float[numStages][];
        SourceDownWeight = new float[numStages][];
        SourceDownBias = new float[numStages][];
        SourceResBlocks = new CosyVoice3HifiResBlockWeights[numStages];
        for (int i = 0; i < numStages; i++)
        {
            UpWeight[i] = GetTensor($"ups.{i}.weight");
            UpBias[i] = GetTensor($"ups.{i}.bias");
            SourceDownWeight[i] = GetTensor($"source_downs.{i}.weight");
            SourceDownBias[i] = GetTensor($"source_downs.{i}.bias");
            SourceResBlocks[i] = new CosyVoice3HifiResBlockWeights(this, $"source_resblocks.{i}");
        }

        int numKernels = ResblockKernels.Length;
        ResBlocks = new CosyVoice3HifiResBlockWeights[numStages * numKernels];
        for (int i = 0; i < ResBlocks.Length; i++)
            ResBlocks[i] = new CosyVoice3HifiResBlockWeights(this, $"resblocks.{i}");

        F0Predictor = new CosyVoice3F0PredictorWeights(this);
        MSourceLinearWeight = GetTensor("m_source.l_linear.weight");
        MSourceLinearBias = GetTensor("m_source.l_linear.bias");
    }

    public float[] GetTensor(string name)
    {
        var info = _model.FindTensor(name) ?? throw new InvalidDataException($"CosyVoice3 HiFT GGUF missing required tensor '{name}'.");
        var bytes = _model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }

    public void Dispose() => _model.Dispose();
}

public sealed class CosyVoice3HifiResBlockWeights : IHifiResBlockWeights
{
    public float[][] Convs1Weight { get; } = new float[3][];
    public float[][] Convs1Bias { get; } = new float[3][];
    public float[][] Convs2Weight { get; } = new float[3][];
    public float[][] Convs2Bias { get; } = new float[3][];
    public float[][] Alpha1 { get; } = new float[3][];
    public float[][] Alpha2 { get; } = new float[3][];

    public CosyVoice3HifiResBlockWeights(CosyVoice3HiftWeights w, string prefix)
    {
        for (int i = 0; i < 3; i++)
        {
            Convs1Weight[i] = w.GetTensor($"{prefix}.convs1.{i}.weight");
            Convs1Bias[i] = w.GetTensor($"{prefix}.convs1.{i}.bias");
            Convs2Weight[i] = w.GetTensor($"{prefix}.convs2.{i}.weight");
            Convs2Bias[i] = w.GetTensor($"{prefix}.convs2.{i}.bias");
            Alpha1[i] = w.GetTensor($"{prefix}.activations1.{i}.alpha");
            Alpha2[i] = w.GetTensor($"{prefix}.activations2.{i}.alpha");
        }
    }
}

public sealed class CosyVoice3F0PredictorWeights : IF0PredictorWeights
{
    public float[][] ConvWeight { get; } = new float[5][];
    public float[][] ConvBias { get; } = new float[5][];
    public float[] ClassifierWeight { get; }
    public float[] ClassifierBias { get; }

    public CosyVoice3F0PredictorWeights(CosyVoice3HiftWeights w)
    {
        int[] idx = [0, 2, 4, 6, 8];
        for (int i = 0; i < 5; i++)
        {
            ConvWeight[i] = w.GetTensor($"f0_predictor.condnet.{idx[i]}.weight");
            ConvBias[i] = w.GetTensor($"f0_predictor.condnet.{idx[i]}.bias");
        }
        ClassifierWeight = w.GetTensor("f0_predictor.classifier.weight");
        ClassifierBias = w.GetTensor("f0_predictor.classifier.bias");
    }
}
