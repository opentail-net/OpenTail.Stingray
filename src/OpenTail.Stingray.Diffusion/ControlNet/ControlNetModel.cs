using System.Buffers;
using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.ControlNet;

/// <summary>
/// Native C# ControlNet model for Stable Diffusion 1.5.
/// Computes 13 spatial residual feature maps (12 down-blocks + 1 middle-block)
/// from structural hint conditions (Canny, Depth, OpenPose, LineArt, etc.)
/// and merges them into UNet skip connections.
/// Reference: stable-diffusion.cpp:src/model/diffusion/control.hpp:ControlNetBlock
/// </summary>
public sealed class ControlNetModel : IDisposable
{
    private readonly IWeightLoader _weights;
    private readonly string _prefix;
    private readonly IComputeBackend? _backend;
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    private readonly Dictionary<string, CoreTensor>? _gpuWeights;
    private bool _disposed;

    public ControlNetModel(IWeightLoader weights, string prefix = "", IComputeBackend? backend = null)
    {
        _weights = weights;
        _prefix = prefix;
        _backend = backend;
        if (backend is not null)
            _gpuWeights = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
    }

    public static ControlNetModel Load(string path, IComputeBackend? backend = null)
    {
        IWeightLoader loader = path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
            ? GgufWeightLoader.Open(path)
            : SafetensorsLoader.Open(path);
        return new ControlNetModel(loader, prefix: "", backend: backend);
    }

    private float[] GetWeight(string name)
    {
        string fullName = Resolve(name);
        if (_weightCache.TryGetValue(fullName, out var cached)) return cached;
        var data = _weights.ReadF32(fullName);
        _weightCache[fullName] = data;
        return data;
    }

    private float[]? TryGetWeight(string name)
    {
        string fullName = Resolve(name);
        if (_weightCache.TryGetValue(fullName, out var cached)) return cached;
        if (_weights.Contains(fullName))
        {
            var data = _weights.ReadF32(fullName);
            _weightCache[fullName] = data;
            return data;
        }
        return null;
    }

    private string Resolve(string name)
    {
        string direct = _prefix + name;
        if (_weights.Contains(direct)) return direct;
        if (_weights.Contains("control_model." + direct)) return "control_model." + direct;
        if (_weights.Contains("controlnet." + direct)) return "controlnet." + direct;
        return direct;
    }

    /// <summary>
    /// Computes ControlNet residuals from hint RGB image and current noisy latent.
    /// </summary>
    public (List<float[]> downResiduals, float[] midResidual) Forward(
        float[] latent,
        float timestep,
        float[] context,
        float[] hintRgb,
        int latH,
        int latW,
        float conditioningScale = 1.0f)
    {
        // 1. Time embedding
        var tEmb = ComputeTimeEmbedding(timestep, 320, 1280);

        // 2. Input Hint Block: downsample RGB [3, 8*latH, 8*latW] -> [320, latH, latW]
        int hintH = latH * 8;
        int hintW = latW * 8;
        var hint = (float[])hintRgb.Clone();

        // 0: Conv 3 -> 16 (stride 1)
        hint = Conv("input_hint_block.0", hint, 3, hintH, hintW, 16, 3);
        DiffusionOps.SiluInPlace(hint);

        // 2: Conv 16 -> 16 (stride 1)
        hint = Conv("input_hint_block.2", hint, 16, hintH, hintW, 16, 3);
        DiffusionOps.SiluInPlace(hint);

        // 4: Conv 16 -> 32 (stride 2)
        hint = Conv("input_hint_block.4", hint, 16, hintH, hintW, 32, 3, stride: 2);
        hintH /= 2; hintW /= 2;
        DiffusionOps.SiluInPlace(hint);

        // 6: Conv 32 -> 32 (stride 1)
        hint = Conv("input_hint_block.6", hint, 32, hintH, hintW, 32, 3);
        DiffusionOps.SiluInPlace(hint);

        // 8: Conv 32 -> 96 (stride 2)
        hint = Conv("input_hint_block.8", hint, 32, hintH, hintW, 96, 3, stride: 2);
        hintH /= 2; hintW /= 2;
        DiffusionOps.SiluInPlace(hint);

        // 10: Conv 96 -> 96 (stride 1)
        hint = Conv("input_hint_block.10", hint, 96, hintH, hintW, 96, 3);
        DiffusionOps.SiluInPlace(hint);

        // 12: Conv 96 -> 256 (stride 2)
        hint = Conv("input_hint_block.12", hint, 96, hintH, hintW, 256, 3, stride: 2);
        hintH /= 2; hintW /= 2;
        DiffusionOps.SiluInPlace(hint);

        // 14: Conv 256 -> 320 (stride 1)
        hint = Conv("input_hint_block.14", hint, 256, hintH, hintW, 320, 3);

        // 3. Input Block 0: Conv 4 -> 320 + hint
        int h = latH, w = latW;
        var cur = Conv("input_blocks.0.0", latent, 4, h, w, 320, 3);
        for (int i = 0; i < cur.Length; i++)
            cur[i] += hint[i];

        var downResiduals = new List<float[]>(12);
        downResiduals.Add(ZeroConv("zero_convs.0.0", cur, 320, h, w, conditioningScale));

        // Block 1
        cur = ResBlock("input_blocks.1.0", cur, tEmb, 320, 320, h, w);
        cur = SpatialTransformer("input_blocks.1.1", cur, context, 320, h, w);
        downResiduals.Add(ZeroConv("zero_convs.1.0", cur, 320, h, w, conditioningScale));

        // Block 2
        cur = ResBlock("input_blocks.2.0", cur, tEmb, 320, 320, h, w);
        cur = SpatialTransformer("input_blocks.2.1", cur, context, 320, h, w);
        downResiduals.Add(ZeroConv("zero_convs.2.0", cur, 320, h, w, conditioningScale));

        // Block 3: Downsample (Conv2D 320 -> 320, stride 2)
        cur = Conv("input_blocks.3.0.op", cur, 320, h, w, 320, 3, stride: 2);
        h /= 2; w /= 2;
        downResiduals.Add(ZeroConv("zero_convs.3.0", cur, 320, h, w, conditioningScale));

        // Block 4
        cur = ResBlock("input_blocks.4.0", cur, tEmb, 320, 640, h, w);
        cur = SpatialTransformer("input_blocks.4.1", cur, context, 640, h, w);
        downResiduals.Add(ZeroConv("zero_convs.4.0", cur, 640, h, w, conditioningScale));

        // Block 5
        cur = ResBlock("input_blocks.5.0", cur, tEmb, 640, 640, h, w);
        cur = SpatialTransformer("input_blocks.5.1", cur, context, 640, h, w);
        downResiduals.Add(ZeroConv("zero_convs.5.0", cur, 640, h, w, conditioningScale));

        // Block 6: Downsample (Conv2D 640 -> 640, stride 2)
        cur = Conv("input_blocks.6.0.op", cur, 640, h, w, 640, 3, stride: 2);
        h /= 2; w /= 2;
        downResiduals.Add(ZeroConv("zero_convs.6.0", cur, 640, h, w, conditioningScale));

        // Block 7
        cur = ResBlock("input_blocks.7.0", cur, tEmb, 640, 1280, h, w);
        cur = SpatialTransformer("input_blocks.7.1", cur, context, 1280, h, w);
        downResiduals.Add(ZeroConv("zero_convs.7.0", cur, 1280, h, w, conditioningScale));

        // Block 8
        cur = ResBlock("input_blocks.8.0", cur, tEmb, 1280, 1280, h, w);
        cur = SpatialTransformer("input_blocks.8.1", cur, context, 1280, h, w);
        downResiduals.Add(ZeroConv("zero_convs.8.0", cur, 1280, h, w, conditioningScale));

        // Block 9: Downsample (Conv2D 1280 -> 1280, stride 2)
        cur = Conv("input_blocks.9.0.op", cur, 1280, h, w, 1280, 3, stride: 2);
        h /= 2; w /= 2;
        downResiduals.Add(ZeroConv("zero_convs.9.0", cur, 1280, h, w, conditioningScale));

        // Block 10
        cur = ResBlock("input_blocks.10.0", cur, tEmb, 1280, 1280, h, w);
        downResiduals.Add(ZeroConv("zero_convs.10.0", cur, 1280, h, w, conditioningScale));

        // Block 11
        cur = ResBlock("input_blocks.11.0", cur, tEmb, 1280, 1280, h, w);
        downResiduals.Add(ZeroConv("zero_convs.11.0", cur, 1280, h, w, conditioningScale));

        // 4. Middle Block
        cur = ResBlock("middle_block.0", cur, tEmb, 1280, 1280, h, w);
        cur = SpatialTransformer("middle_block.1", cur, context, 1280, h, w);
        cur = ResBlock("middle_block.2", cur, tEmb, 1280, 1280, h, w);

        var midResidual = ZeroConv("middle_block_out.0", cur, 1280, h, w, conditioningScale);

        return (downResiduals, midResidual);
    }

    private float[] ZeroConv(string name, float[] x, int channels, int h, int w, float scale)
    {
        var outF = Conv(name, x, channels, h, w, channels, 1, padding: 0);
        if (scale != 1.0f)
        {
            for (int i = 0; i < outF.Length; i++)
                outF[i] *= scale;
        }
        return outF;
    }

    private float[] Conv(string name, float[] x, int inC, int h, int w, int outC, int ksize, int stride = 1, int padding = -1)
    {
        var wF = GetWeight($"{name}.weight");
        var bF = TryGetWeight($"{name}.bias");
        return DiffusionOps.Conv2D(x, wF, bF, 1, inC, h, w, outC, ksize, ksize, stride, padding);
    }

    private float[] ResBlock(string name, float[] x, float[] tEmb, int inC, int outC, int h, int w)
    {
        var (gn1Gamma, gn1Beta) = GetNormWeights($"{name}.in_layers.0", inC);
        var h1 = (float[])x.Clone();
        DiffusionOps.GroupNorm(h1, gn1Gamma, gn1Beta, 1, inC, h, w, 32);
        DiffusionOps.SiluInPlace(h1);
        h1 = Conv($"{name}.in_layers.2", h1, inC, h, w, outC, 3);

        var tProj = Linear($"{name}.emb_layers.1", tEmb, 1280, outC);
        for (int c = 0; c < outC; c++)
        {
            float bias = tProj[c];
            int offset = c * h * w;
            for (int i = 0; i < h * w; i++)
                h1[offset + i] += bias;
        }

        var (gn2Gamma, gn2Beta) = GetNormWeights($"{name}.out_layers.0", outC);
        var h2 = (float[])h1.Clone();
        DiffusionOps.GroupNorm(h2, gn2Gamma, gn2Beta, 1, outC, h, w, 32);
        DiffusionOps.SiluInPlace(h2);
        h2 = Conv($"{name}.out_layers.3", h2, outC, h, w, outC, 3);

        var residual = x;
        if (inC != outC)
        {
            string scKey = Resolve($"{name}.skip_connection");
            if (_weights.Contains($"{scKey}.weight"))
                residual = Conv($"{name}.skip_connection", x, inC, h, w, outC, 1, padding: 0);
            else if (_weights.Contains($"{scKey}.conv.weight"))
                residual = Conv($"{name}.skip_connection.conv", x, inC, h, w, outC, 1, padding: 0);
        }

        for (int i = 0; i < h2.Length; i++)
            h2[i] += residual[i];

        return h2;
    }

    private float[] SpatialTransformer(string name, float[] x, float[] context, int channels, int h, int w)
    {
        string p = $"{name}.transformer_blocks.0";
        var (gnGamma, gnBeta) = GetNormWeights($"{name}.norm", channels);
        var normX = (float[])x.Clone();
        DiffusionOps.GroupNorm(normX, gnGamma, gnBeta, 1, channels, h, w, 32);

        var projIn = Conv($"{name}.proj_in", normX, channels, h, w, channels, 1, padding: 0);

        int seqLen = h * w;
        var (n1G, n1B) = GetLayerNormWeights($"{p}.norm1", channels);
        var h1 = (float[])projIn.Clone();
        DiffusionOps.LayerNorm(h1, n1G, n1B, channels);
        var q1 = Linear($"{p}.attn1.to_q", h1, channels, channels);
        var k1 = Linear($"{p}.attn1.to_k", h1, channels, channels);
        var v1 = Linear($"{p}.attn1.to_v", h1, channels, channels);
        var attn1Out = AttentionHeads(q1, k1, v1, seqLen, seqLen, channels, 8);
        attn1Out = Linear($"{p}.attn1.to_out.0", attn1Out, channels, channels);
        for (int i = 0; i < projIn.Length; i++) projIn[i] += attn1Out[i];

        var (n2G, n2B) = GetLayerNormWeights($"{p}.norm2", channels);
        var h2 = (float[])projIn.Clone();
        DiffusionOps.LayerNorm(h2, n2G, n2B, channels);
        var q2 = Linear($"{p}.attn2.to_q", h2, channels, channels);
        var k2 = Linear($"{p}.attn2.to_k", context, 768, channels);
        var v2 = Linear($"{p}.attn2.to_v", context, 768, channels);
        var attn2Out = AttentionHeads(q2, k2, v2, seqLen, 77, channels, 8);
        attn2Out = Linear($"{p}.attn2.to_out.0", attn2Out, channels, channels);
        for (int i = 0; i < projIn.Length; i++) projIn[i] += attn2Out[i];

        var (n3G, n3B) = GetLayerNormWeights($"{p}.norm3", channels);
        var h3 = (float[])projIn.Clone();
        DiffusionOps.LayerNorm(h3, n3G, n3B, channels);
        var ff1 = LinearGeluGeGLU($"{p}.ff.net.0.proj", h3, channels, channels * 4);
        var ffOut = Linear($"{p}.ff.net.2", ff1, channels * 4, channels);
        for (int i = 0; i < projIn.Length; i++) projIn[i] += ffOut[i];

        var projOut = Conv($"{name}.proj_out", projIn, channels, h, w, channels, 1, padding: 0);
        for (int i = 0; i < x.Length; i++) x[i] += projOut[i];
        return x;
    }

    private static float[] AttentionHeads(float[] q, float[] k, float[] v, int qSeq, int kvSeq, int dim, int heads)
    {
        int headDim = dim / heads;
        float scale = 1.0f / MathF.Sqrt(headDim);
        var output = new float[qSeq * dim];

        for (int h = 0; h < heads; h++)
        {
            int hOff = h * headDim;
            for (int i = 0; i < qSeq; i++)
            {
                int qRow = i * dim + hOff;
                var scores = new float[kvSeq];
                float maxScore = float.NegativeInfinity;

                for (int j = 0; j < kvSeq; j++)
                {
                    int kRow = j * dim + hOff;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                        dot += q[qRow + d] * k[kRow + d];
                    dot *= scale;
                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }

                float sumExp = 0f;
                for (int j = 0; j < kvSeq; j++)
                {
                    scores[j] = MathF.Exp(scores[j] - maxScore);
                    sumExp += scores[j];
                }
                float invSum = 1f / sumExp;
                for (int j = 0; j < kvSeq; j++) scores[j] *= invSum;

                int outRow = i * dim + hOff;
                for (int d = 0; d < headDim; d++)
                {
                    float sum = 0f;
                    for (int j = 0; j < kvSeq; j++)
                        sum += scores[j] * v[j * dim + hOff + d];
                    output[outRow + d] = sum;
                }
            }
        }

        return output;
    }

    private float[] LinearGeluGeGLU(string name, float[] x, int inDim, int outDim)
    {
        var proj = Linear(name, x, inDim, outDim * 2);
        int rows = x.Length / inDim;
        var result = new float[rows * outDim];

        for (int r = 0; r < rows; r++)
        {
            int srcOff = r * (outDim * 2);
            int dstOff = r * outDim;
            for (int c = 0; c < outDim; c++)
            {
                float val = proj[srcOff + c];
                float gate = proj[srcOff + outDim + c];
                result[dstOff + c] = val * DiffusionOps.Gelu(gate);
            }
        }
        return result;
    }

    private float[] Linear(string name, float[] x, int inDim, int outDim)
    {
        var w = GetWeight($"{name}.weight");
        var b = TryGetWeight($"{name}.bias");
        int rows = x.Length / inDim;
        var outF = new float[rows * outDim];

        for (int r = 0; r < rows; r++)
        {
            int inOff = r * inDim;
            int outOff = r * outDim;
            for (int o = 0; o < outDim; o++)
            {
                float sum = b is not null ? b[o] : 0f;
                int wOff = o * inDim;
                for (int i = 0; i < inDim; i++)
                    sum += x[inOff + i] * w[wOff + i];
                outF[outOff + o] = sum;
            }
        }
        return outF;
    }

    private (float[] g, float[] b) GetNormWeights(string name, int dim)
    {
        var g = TryGetWeight($"{name}.weight") ?? new float[dim];
        if (!_weights.Contains(Resolve($"{name}.weight"))) Array.Fill(g, 1.0f);
        var b = TryGetWeight($"{name}.bias") ?? new float[dim];
        return (g, b);
    }

    private (float[] g, float[] b) GetLayerNormWeights(string name, int dim)
    {
        var g = TryGetWeight($"{name}.weight") ?? new float[dim];
        if (!_weights.Contains(Resolve($"{name}.weight"))) Array.Fill(g, 1.0f);
        var b = TryGetWeight($"{name}.bias") ?? new float[dim];
        return (g, b);
    }

    private float[] ComputeTimeEmbedding(float timestep, int dim, int outDim)
    {
        var emb = new float[dim];
        int half = dim / 2;
        float factor = 10000.0f;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-MathF.Log(factor) * i / half);
            emb[i] = MathF.Cos(timestep * freq);
            emb[half + i] = MathF.Sin(timestep * freq);
        }

        var t0 = Linear("time_embed.0", emb, dim, outDim);
        DiffusionOps.SiluInPlace(t0);
        return Linear("time_embed.2", t0, outDim, outDim);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _weights.Dispose();
        }
    }
}
