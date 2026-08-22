using System.IO;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// Real weights for CosyVoice2's HiFT vocoder (`models/cosyvoice2_hift.safetensors`,
/// converted this session from the official `hift.pt`). Architecturally the same NSF-source +
/// ISTFTNet HiFiGAN family as `Chatterbox/ChatterboxS3GenWeights.cs`'s vocoder (S3Gen's HiFT
/// stage was itself derived from CosyVoice's -- same lineage, confirmed by real tensor shapes
/// during this session's audit matching Chatterbox's hardcoded architecture defaults exactly:
/// upsample rates [8,5,3], kernels [16,11,7], resblock kernels [3,7,11] per stage,
/// source-resblock kernels [7,7,11], base_channels=512, n_fft=16, hop=4).
///
/// Real difference from Chatterbox's checkpoint: this one's conv weights use PyTorch's newer
/// `parametrizations.weight.original0/1` weight-norm encoding (`original0`=per-output-channel
/// magnitude `g` [outCh,1,1], `original1`=direction `v` [outCh,inCh,K]) rather than being
/// pre-fused -- folded at load time into plain conv weights via `w[o,i,k] = g[o] * v[o,i,k] /
/// ||v[o,:,:]||_2` (confirmed via real tensor shapes: `original0` is always `[outCh,1,1]`,
/// the standard PyTorch `weight_norm(dim=0)` magnitude shape). `source_downs.*`/`m_source.*`/
/// `f0_predictor.classifier.*` are plain (unparametrized) convs/linears, confirmed by their
/// absence of a `parametrizations.` prefix.
/// </summary>
public sealed class CosyVoiceHiftWeights : IDisposable
{
    public SafetensorsLoader Loader { get; }

    public int[] UpsampleRates { get; } = [8, 5, 3];
    public int[] UpsampleKernels { get; } = [16, 11, 7];
    public int[] ResblockKernels { get; } = [3, 7, 11];
    public int[] SourceResblockKernels { get; } = [7, 7, 11];
    public int BaseChannels { get; } = 512;
    public int NbHarmonics { get; } = 8;
    public int IstftNFft { get; } = 16;
    public int IstftHopLen { get; } = 4;
    public int SampleRate { get; } = 24000;

    public float[] ConvPreWeight { get; }
    public float[] ConvPreBias { get; }
    public float[] ConvPostWeight { get; }
    public float[] ConvPostBias { get; }

    public float[][] UpWeight { get; }
    public float[][] UpBias { get; }
    public float[][] SourceDownWeight { get; }
    public float[][] SourceDownBias { get; }
    public CosyVoiceHifiResBlockWeights[] SourceResBlocks { get; }
    public CosyVoiceHifiResBlockWeights[] ResBlocks { get; } // numStages * numKernels

    public CosyVoiceF0PredictorWeights F0Predictor { get; }
    public float[] MSourceLinearWeight { get; }
    public float[] MSourceLinearBias { get; }

    public CosyVoiceHiftWeights(string safetensorsPath)
    {
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException($"CosyVoice HiFT model file not found: {safetensorsPath}");

        Loader = SafetensorsLoader.Open(safetensorsPath);

        ConvPreWeight = GetFoldedConvWeight("conv_pre");
        ConvPreBias = GetTensor("conv_pre.bias");
        ConvPostWeight = GetFoldedConvWeight("conv_post");
        ConvPostBias = GetTensor("conv_post.bias");

        int numStages = UpsampleRates.Length;
        UpWeight = new float[numStages][];
        UpBias = new float[numStages][];
        SourceDownWeight = new float[numStages][];
        SourceDownBias = new float[numStages][];
        SourceResBlocks = new CosyVoiceHifiResBlockWeights[numStages];
        for (int i = 0; i < numStages; i++)
        {
            UpWeight[i] = GetFoldedConvWeight($"ups.{i}");
            UpBias[i] = GetTensor($"ups.{i}.bias");
            SourceDownWeight[i] = GetTensor($"source_downs.{i}.weight"); // plain, no weight_norm
            SourceDownBias[i] = GetTensor($"source_downs.{i}.bias");
            SourceResBlocks[i] = new CosyVoiceHifiResBlockWeights(this, $"source_resblocks.{i}");
        }

        int numKernels = ResblockKernels.Length;
        ResBlocks = new CosyVoiceHifiResBlockWeights[numStages * numKernels];
        for (int i = 0; i < ResBlocks.Length; i++)
            ResBlocks[i] = new CosyVoiceHifiResBlockWeights(this, $"resblocks.{i}");

        F0Predictor = new CosyVoiceF0PredictorWeights(this);
        MSourceLinearWeight = GetTensor("m_source.l_linear.weight");
        MSourceLinearBias = GetTensor("m_source.l_linear.bias");
    }

    public float[] GetTensor(string name) => Loader.ReadF32(name);

    /// <summary>Folds PyTorch's parametrized weight_norm (original0=g magnitude [outCh,1,1], original1=v direction [outCh,inCh,K]) into a plain conv weight: w[o,i,k] = g[o] * v[o,i,k] / ||v[o,:,:]||_2.</summary>
    public float[] GetFoldedConvWeight(string prefix)
    {
        var g = GetTensor($"{prefix}.parametrizations.weight.original0");
        var v = GetTensor($"{prefix}.parametrizations.weight.original1");
        int[] vShape = Loader.GetShape($"{prefix}.parametrizations.weight.original1");
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

    public void Dispose() => Loader.Dispose();
}

/// <summary>Snake-activated HiFiGAN resblock, 3 dilated conv pairs (dilations [1,3,5]) -- same structure as `Chatterbox/ChatterboxS3GenWeights.cs`'s `ChatterboxHifiResBlockWeights`.</summary>
public sealed class CosyVoiceHifiResBlockWeights
{
    public float[][] Convs1Weight { get; } = new float[3][];
    public float[][] Convs1Bias { get; } = new float[3][];
    public float[][] Convs2Weight { get; } = new float[3][];
    public float[][] Convs2Bias { get; } = new float[3][];
    public float[][] Alpha1 { get; } = new float[3][];
    public float[][] Alpha2 { get; } = new float[3][];

    public CosyVoiceHifiResBlockWeights(CosyVoiceHiftWeights w, string prefix)
    {
        for (int i = 0; i < 3; i++)
        {
            Convs1Weight[i] = w.GetFoldedConvWeight($"{prefix}.convs1.{i}");
            Convs1Bias[i] = w.GetTensor($"{prefix}.convs1.{i}.bias");
            Convs2Weight[i] = w.GetFoldedConvWeight($"{prefix}.convs2.{i}");
            Convs2Bias[i] = w.GetTensor($"{prefix}.convs2.{i}.bias");
            Alpha1[i] = w.GetTensor($"{prefix}.activations1.{i}.alpha");
            Alpha2[i] = w.GetTensor($"{prefix}.activations2.{i}.alpha");
        }
    }
}

/// <summary>ConvRNNF0Predictor: 5 causal-ish Conv1d(k=3)+ELU stages (condnet indices 0,2,4,6,8 -- odd indices are the ELU activations, no weights) + a final linear classifier.</summary>
public sealed class CosyVoiceF0PredictorWeights
{
    public float[][] ConvWeight { get; } = new float[5][];
    public float[][] ConvBias { get; } = new float[5][];
    public float[] ClassifierWeight { get; }
    public float[] ClassifierBias { get; }

    public CosyVoiceF0PredictorWeights(CosyVoiceHiftWeights w)
    {
        int[] idx = [0, 2, 4, 6, 8];
        for (int i = 0; i < 5; i++)
        {
            ConvWeight[i] = w.GetFoldedConvWeight($"f0_predictor.condnet.{idx[i]}");
            ConvBias[i] = w.GetTensor($"f0_predictor.condnet.{idx[i]}.bias");
        }
        ClassifierWeight = w.GetTensor("f0_predictor.classifier.weight");
        ClassifierBias = w.GetTensor("f0_predictor.classifier.bias");
    }
}
