using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Universal VAE encoder: encodes RGB image [1, 3, H, W] to latent distribution [1, C, H/8, W/8].
/// Supports:
///   - 4-channel latents (Stable Diffusion 1.5, SDXL) with scaling factor 0.18215
///   - 16-channel latents (SD3, FLUX.1) with scaling factor 1.5305 / 0.3611
/// Reference: stable-diffusion.cpp:src/model/vae/auto_encoder_kl.hpp
/// </summary>
public sealed class VaeEncoder : IDisposable
{
    private readonly IWeightLoader _st;
    private readonly IComputeBackend? _backend;
    private readonly Dictionary<string, float[]> _cpuWeights = new(StringComparer.Ordinal);
    private readonly Dictionary<string, CoreTensor>? _gpuWeights;
    private bool _disposed;

    public VaeEncoder(string path) : this(SafetensorsLoader.Open(path)) { }

    public VaeEncoder(IWeightLoader st, IComputeBackend? backend = null)
    {
        _st = st;
        _backend = backend;
        if (backend is not null)
            _gpuWeights = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
    }

    private string Resolve(string name)
    {
        if (_st.Contains(name)) return name;
        if (_st.Contains("first_stage_model." + name)) return "first_stage_model." + name;
        if (_st.Contains("vae." + name)) return "vae." + name;
        return name;
    }

    /// <summary>
    /// Encodes RGB float [3, H, W] (in [0, 1]) into latent space [C, H/8, W/8].
    /// </summary>
    public float[] Encode(float[] rgb, int height, int width, int latentChannels = 4, bool sampleDeterministic = true, int seed = -1)
    {
        if (height % 8 != 0 || width % 8 != 0)
            throw new ArgumentException($"Image dimensions must be divisible by 8 (got {width}x{height})");

        int latH = height / 8;
        int latW = width / 8;

        // 1. Normalize RGB to [-1, 1]
        var x = new float[rgb.Length];
        for (int i = 0; i < x.Length; i++)
            x[i] = rgb[i] * 2.0f - 1.0f;

        // 2. conv_in: Conv2D(3 -> 128, 3x3)
        string enc = Resolve("encoder");
        string convInKey = Resolve("encoder.conv_in");
        x = ConvBlock(convInKey, x, 1, 3, height, width, 128, 3);
        int ch = 128, h = height, w = width;

        bool isCompVis = _st.Contains(Resolve("encoder.down.0.block.0.conv1.weight"));

        if (isCompVis)
        {
            // CompVis SD1.5 schema
            (x, ch, h, w) = DownBlockCompVis(x, 1, ch, h, w, $"{enc}.down.0", outCh: 128, downsample: true);
            (x, ch, h, w) = DownBlockCompVis(x, 1, ch, h, w, $"{enc}.down.1", outCh: 256, downsample: true);
            (x, ch, h, w) = DownBlockCompVis(x, 1, ch, h, w, $"{enc}.down.2", outCh: 512, downsample: true);
            (x, ch, h, w) = DownBlockCompVis(x, 1, ch, h, w, $"{enc}.down.3", outCh: 512, downsample: false);

            x = ResBlock($"{enc}.mid.block_1", x, 1, ch, h, w);
            x = MidAttnCompVis($"{enc}.mid.attn_1", x, 1, ch, h, w);
            x = ResBlock($"{enc}.mid.block_2", x, 1, ch, h, w);
        }
        else
        {
            // Diffusers schema
            (x, ch, h, w) = DownBlockDiffusers(x, 1, ch, h, w, $"{enc}.down_blocks.0", outCh: 128, downsample: true);
            (x, ch, h, w) = DownBlockDiffusers(x, 1, ch, h, w, $"{enc}.down_blocks.1", outCh: 256, downsample: true);
            (x, ch, h, w) = DownBlockDiffusers(x, 1, ch, h, w, $"{enc}.down_blocks.2", outCh: 512, downsample: true);
            (x, ch, h, w) = DownBlockDiffusers(x, 1, ch, h, w, $"{enc}.down_blocks.3", outCh: 512, downsample: false);

            x = ResBlock($"{enc}.mid_block.resnets.0", x, 1, ch, h, w);
            x = MidAttnDiffusers($"{enc}.mid_block.attentions.0", x, 1, ch, h, w);
            x = ResBlock($"{enc}.mid_block.resnets.1", x, 1, ch, h, w);
        }

        // 3. norm_out + SiLU + conv_out (512 -> 2 * latentChannels)
        string normOutKey = Resolve("encoder.norm_out");
        var (noGamma, noBeta) = GetNormWeights(normOutKey, ch);
        DiffusionOps.GroupNorm(x, noGamma, noBeta, 1, ch, h, w, 32);
        DiffusionOps.SiluInPlace(x);

        int outChans = 2 * latentChannels;
        string convOutKey = Resolve("encoder.conv_out");
        var moments = ConvBlock(convOutKey, x, 1, ch, h, w, outChans, 3);

        // 4. Extract mean and optional variance sampling
        var latent = new float[latentChannels * latH * latW];
        float scale = latentChannels == 4 ? 0.18215f : 0.3611f;

        var rng = seed >= 0 ? new Random(seed) : new Random();

        for (int c = 0; c < latentChannels; c++)
        {
            int meanOffset = c * latH * latW;
            int logvarOffset = (latentChannels + c) * latH * latW;
            int targetOffset = c * latH * latW;

            for (int p = 0; p < latH * latW; p++)
            {
                float mean = moments[meanOffset + p];
                if (sampleDeterministic)
                {
                    latent[targetOffset + p] = mean * scale;
                }
                else
                {
                    float logvar = moments[logvarOffset + p];
                    float std = MathF.Exp(0.5f * Math.Clamp(logvar, -30.0f, 20.0f));
                    double u1 = 1.0 - rng.NextDouble();
                    double u2 = 1.0 - rng.NextDouble();
                    float eps = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
                    latent[targetOffset + p] = (mean + std * eps) * scale;
                }
            }
        }

        return latent;
    }

    private (float[] outTensor, int outCh, int outH, int outW) DownBlockCompVis(float[] x, int b, int inCh, int h, int w, string prefix, int outCh, bool downsample)
    {
        x = ResBlock($"{prefix}.block.0", x, b, inCh, h, w, outCh);
        x = ResBlock($"{prefix}.block.1", x, b, outCh, h, w, outCh);

        if (downsample)
        {
            x = ConvBlock($"{prefix}.downsample.conv", x, b, outCh, h, w, outCh, 3, stride: 2, padding: 1);
            h /= 2;
            w /= 2;
        }

        return (x, outCh, h, w);
    }

    private (float[] outTensor, int outCh, int outH, int outW) DownBlockDiffusers(float[] x, int b, int inCh, int h, int w, string prefix, int outCh, bool downsample)
    {
        x = ResBlock($"{prefix}.resnets.0", x, b, inCh, h, w, outCh);
        x = ResBlock($"{prefix}.resnets.1", x, b, outCh, h, w, outCh);

        if (downsample)
        {
            x = ConvBlock($"{prefix}.downsamplers.0.conv", x, b, outCh, h, w, outCh, 3, stride: 2, padding: 1);
            h /= 2;
            w /= 2;
        }

        return (x, outCh, h, w);
    }

    private float[] ResBlock(string prefix, float[] x, int b, int inCh, int h, int w, int? outChOpt = null)
    {
        int outCh = outChOpt ?? inCh;
        var residual = x;

        // norm1 + SiLU + conv1
        var (g1, b1) = GetNormWeights($"{prefix}.norm1", inCh);
        var h1 = (float[])x.Clone();
        DiffusionOps.GroupNorm(h1, g1, b1, b, inCh, h, w, 32);
        DiffusionOps.SiluInPlace(h1);
        h1 = ConvBlock($"{prefix}.conv1", h1, b, inCh, h, w, outCh, 3);

        // norm2 + SiLU + conv2
        var (g2, b2) = GetNormWeights($"{prefix}.norm2", outCh);
        var h2 = (float[])h1.Clone();
        DiffusionOps.GroupNorm(h2, g2, b2, b, outCh, h, w, 32);
        DiffusionOps.SiluInPlace(h2);
        h2 = ConvBlock($"{prefix}.conv2", h2, b, outCh, h, w, outCh, 3);

        // Optional shortcut convolution if channel mismatch
        if (inCh != outCh)
        {
            string scKey = Resolve($"{prefix}.nin_shortcut");
            if (!_st.Contains($"{scKey}.weight"))
                scKey = Resolve($"{prefix}.conv_shortcut");
            residual = ConvBlock(scKey, x, b, inCh, h, w, outCh, 1, padding: 0);
        }

        for (int i = 0; i < h2.Length; i++)
            h2[i] += residual[i];

        return h2;
    }

    private float[] MidAttnCompVis(string prefix, float[] x, int b, int ch, int h, int w)
    {
        var (gnGamma, gnBeta) = GetNormWeights($"{prefix}.norm", ch);
        var normX = (float[])x.Clone();
        DiffusionOps.GroupNorm(normX, gnGamma, gnBeta, b, ch, h, w, 32);

        var q = ConvBlock($"{prefix}.q", normX, b, ch, h, w, ch, 1, padding: 0);
        var k = ConvBlock($"{prefix}.k", normX, b, ch, h, w, ch, 1, padding: 0);
        var v = ConvBlock($"{prefix}.v", normX, b, ch, h, w, ch, 1, padding: 0);

        var outAttn = SelfAttentionSpatial(q, k, v, ch, h, w);
        outAttn = ConvBlock($"{prefix}.proj_out", outAttn, b, ch, h, w, ch, 1, padding: 0);

        for (int i = 0; i < x.Length; i++)
            outAttn[i] += x[i];

        return outAttn;
    }

    private float[] MidAttnDiffusers(string prefix, float[] x, int b, int ch, int h, int w)
    {
        var (gnGamma, gnBeta) = GetNormWeights($"{prefix}.group_norm", ch);
        var normX = (float[])x.Clone();
        DiffusionOps.GroupNorm(normX, gnGamma, gnBeta, b, ch, h, w, 32);

        var q = ConvBlock($"{prefix}.to_q", normX, b, ch, h, w, ch, 1, padding: 0);
        var k = ConvBlock($"{prefix}.to_k", normX, b, ch, h, w, ch, 1, padding: 0);
        var v = ConvBlock($"{prefix}.to_v", normX, b, ch, h, w, ch, 1, padding: 0);

        var outAttn = SelfAttentionSpatial(q, k, v, ch, h, w);
        outAttn = ConvBlock($"{prefix}.to_out.0", outAttn, b, ch, h, w, ch, 1, padding: 0);

        for (int i = 0; i < x.Length; i++)
            outAttn[i] += x[i];

        return outAttn;
    }

    private static float[] SelfAttentionSpatial(float[] q, float[] k, float[] v, int ch, int h, int w)
    {
        int seqLen = h * w;
        float scale = 1.0f / MathF.Sqrt(ch);
        var output = new float[ch * seqLen];

        for (int i = 0; i < seqLen; i++)
        {
            var scores = new float[seqLen];
            float maxScore = float.NegativeInfinity;

            for (int j = 0; j < seqLen; j++)
            {
                float dot = 0f;
                for (int c = 0; c < ch; c++)
                    dot += q[c * seqLen + i] * k[c * seqLen + j];
                dot *= scale;
                scores[j] = dot;
                if (dot > maxScore) maxScore = dot;
            }

            float sumExp = 0f;
            for (int j = 0; j < seqLen; j++)
            {
                scores[j] = MathF.Exp(scores[j] - maxScore);
                sumExp += scores[j];
            }
            float invSum = 1f / sumExp;
            for (int j = 0; j < seqLen; j++)
                scores[j] *= invSum;

            for (int c = 0; c < ch; c++)
            {
                float sum = 0f;
                for (int j = 0; j < seqLen; j++)
                    sum += scores[j] * v[c * seqLen + j];
                output[c * seqLen + i] = sum;
            }
        }

        return output;
    }

    private float[] ConvBlock(string prefix, float[] input, int b, int inCh, int h, int w, int outCh, int kernelSize, int stride = 1, int padding = 1)
    {
        var wData = GetWeight($"{prefix}.weight");
        var bData = _st.Contains($"{prefix}.bias") ? GetWeight($"{prefix}.bias") : null;
        return DiffusionOps.Conv2D(input, wData, bData, b, inCh, h, w, outCh, kernelSize, kernelSize, stride, padding);
    }

    private (float[] gamma, float[] beta) GetNormWeights(string prefix, int channels)
    {
        string gKey = Resolve($"{prefix}.weight");
        string bKey = Resolve($"{prefix}.bias");

        var gamma = _st.Contains(gKey) ? GetWeight(gKey) : new float[channels];
        if (!_st.Contains(gKey)) Array.Fill(gamma, 1.0f);

        var beta = _st.Contains(bKey) ? GetWeight(bKey) : new float[channels];
        return (gamma, beta);
    }

    private float[] GetWeight(string key)
    {
        string resolved = Resolve(key);
        if (_cpuWeights.TryGetValue(resolved, out var cached)) return cached;
        var data = _st.ReadF32(resolved);
        _cpuWeights[resolved] = data;
        return data;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _st.Dispose();
        }
    }
}
