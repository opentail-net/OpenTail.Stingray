using System.Buffers;
using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.HunyuanVideo;

/// <summary>
/// Native C# HunyuanVideo Dual-Stream and Single-Stream Diffusion Transformer (DiT).
/// Reference: stable-diffusion.cpp:src/model/diffusion/hunyuan.hpp:HunyuanVideoModel
/// </summary>
public sealed class HunyuanVideoModel : IDisposable
{
    private readonly IWeightLoader _weights;
    private readonly string _prefix;
    private readonly IComputeBackend? _backend;
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    private readonly Dictionary<string, CoreTensor>? _gpuWeights;
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _depthDouble;
    private readonly int _depthSingle;
    private bool _disposed;

    public const int InChannels = 64;   // 16 * 2 * 2
    public const int OutChannels = 16;
    public const int TextDim = 4096;    // LLaMA-3 / Qwen2.5-VL text dimension

    public int Dim => _dim;
    public int NumHeads => _numHeads;
    public int DepthDouble => _depthDouble;
    public int DepthSingle => _depthSingle;

    public HunyuanVideoModel(
        IWeightLoader weights,
        string prefix = "",
        int dim = 3072,
        int numHeads = 24,
        int depthDouble = 20,
        int depthSingle = 0,
        IComputeBackend? backend = null)
    {
        _weights = weights;
        _prefix = prefix;
        _backend = backend;
        (_dim, _numHeads, _depthDouble, _depthSingle) = DetectConfig(weights, prefix, dim, numHeads, depthDouble, depthSingle);
        _headDim = _dim / _numHeads;
        if (backend is not null)
            _gpuWeights = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
    }

    private static (int dim, int numHeads, int depthDouble, int depthSingle) DetectConfig(
        IWeightLoader weights,
        string prefix,
        int defDim,
        int defHeads,
        int defDouble,
        int defSingle)
    {
        int detectedDouble = defDouble;
        int detectedSingle = defSingle;

        for (int i = 60; i >= 0; i--)
        {
            string key = $"{prefix}double_blocks.{i}.img_attn.qkv.weight";
            if (weights.Contains(key) || weights.Contains("model.diffusion_model." + key))
            {
                detectedDouble = i + 1;
                break;
            }
        }

        for (int i = 60; i >= 0; i--)
        {
            string key = $"{prefix}single_blocks.{i}.linear1.weight";
            if (weights.Contains(key) || weights.Contains("model.diffusion_model." + key))
            {
                detectedSingle = i + 1;
                break;
            }
        }

        int detectedDim = defDim;
        string inKey = $"{prefix}img_in.proj.weight";
        if (!weights.Contains(inKey)) inKey = "model.diffusion_model." + inKey;
        if (weights.Contains(inKey))
        {
            var w = weights.ReadF32(inKey);
            if (w.Length > 0 && InChannels > 0)
                detectedDim = w.Length / InChannels;
        }

        int detectedHeads = detectedDim == 2048 ? 16 : defHeads;
        return (detectedDim, detectedHeads, detectedDouble, detectedSingle);
    }

    private string Resolve(string name)
    {
        string direct = _prefix + name;
        if (_weights.Contains(direct)) return direct;
        if (_weights.Contains("model.diffusion_model." + direct)) return "model.diffusion_model." + direct;
        if (_weights.Contains("diffusion_model." + direct)) return "diffusion_model." + direct;
        return direct;
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

    /// <summary>
    /// Evaluates the HunyuanVideo forward pass.
    /// </summary>
    public float[] Forward(
        float[] latent,
        float timestep,
        float[] textContext,
        int numFrames,
        int latH,
        int latW)
    {
        int patchH = latH / 2;
        int patchW = latW / 2;
        int numImgTokens = numFrames * patchH * patchW;
        int numTxtTokens = textContext.Length / TextDim;

        // 1. Pack 16-channel video latent into 64-channel patches
        var packed = PackLatents(latent, numFrames, latH, latW);

        // 2. Input projections
        var imgTokens = Linear("img_in.proj", packed, InChannels, _dim);
        var txtTokens = Linear("txt_in.input_embedder", textContext, TextDim, _dim);

        // 3. Timestep embedding (sinusoidal 256 -> linear dim -> silu -> linear dim)
        var tEmb = ComputeTimestepEmbedding(timestep);

        // 4. 3D-RoPE positional frequencies
        var (cos, sin) = HunyuanVideoRoPE.Compute3DRoPE(numFrames, patchH, patchW, _headDim);

        // 5. Dual-Stream Blocks (double_blocks)
        for (int b = 0; b < _depthDouble; b++)
        {
            string p = $"double_blocks.{b}";
            (imgTokens, txtTokens) = DoubleBlock(p, imgTokens, txtTokens, tEmb, cos, sin, numImgTokens, numTxtTokens);
        }

        // 6. Single-Stream Blocks (single_blocks) if present
        if (_depthSingle > 0)
        {
            var singleTokens = ConcatSequences(txtTokens, imgTokens, numTxtTokens, numImgTokens);
            int totalSeq = numTxtTokens + numImgTokens;
            for (int b = 0; b < _depthSingle; b++)
            {
                string p = $"single_blocks.{b}";
                singleTokens = SingleBlock(p, singleTokens, tEmb, cos, sin, totalSeq);
            }
            imgTokens = singleTokens.AsSpan(numTxtTokens * _dim, numImgTokens * _dim).ToArray();
        }

        // 7. Final Layer (AdaLN + Linear dim -> 64)
        var headMod = Linear("final_layer.adaLN_modulation.1", DiffusionOpsSilu(tEmb), _dim, _dim * 2);
        var headGamma = headMod.AsSpan(0, _dim);
        var headBeta = headMod.AsSpan(_dim, _dim);

        var normed = (float[])imgTokens.Clone();
        DiffusionOps.LayerNorm(normed, headGamma, headBeta, _dim);

        var outPacked = Linear("final_layer.linear", normed, _dim, InChannels);

        // 8. Unpack patches [numTokens, 64] -> [16, numFrames, latH, latW]
        return UnpackLatents(outPacked, numFrames, latH, latW);
    }

    private (float[] imgOut, float[] txtOut) DoubleBlock(
        string prefix,
        float[] img,
        float[] txt,
        float[] tEmb,
        float[] cos,
        float[] sin,
        int numImg,
        int numTxt)
    {
        var imgMod = Linear($"{prefix}.img_mod.linear", DiffusionOpsSilu(tEmb), _dim, _dim * 6);
        var txtMod = Linear($"{prefix}.txt_mod.linear", DiffusionOpsSilu(tEmb), _dim, _dim * 6);

        var imgS1 = imgMod.AsSpan(0 * _dim, _dim);
        var imgSc1 = imgMod.AsSpan(1 * _dim, _dim);
        var imgG1 = imgMod.AsSpan(2 * _dim, _dim);
        var imgS2 = imgMod.AsSpan(3 * _dim, _dim);
        var imgSc2 = imgMod.AsSpan(4 * _dim, _dim);
        var imgG2 = imgMod.AsSpan(5 * _dim, _dim);

        var txtS1 = txtMod.AsSpan(0 * _dim, _dim);
        var txtSc1 = txtMod.AsSpan(1 * _dim, _dim);
        var txtG1 = txtMod.AsSpan(2 * _dim, _dim);
        var txtS2 = txtMod.AsSpan(3 * _dim, _dim);
        var txtSc2 = txtMod.AsSpan(4 * _dim, _dim);
        var txtG2 = txtMod.AsSpan(5 * _dim, _dim);

        var normedImg1 = Modulate(img, numImg, imgS1, imgSc1);
        var normedTxt1 = Modulate(txt, numTxt, txtS1, txtSc1);

        var (imgAttn, txtAttn) = JointAttention($"{prefix}", normedImg1, normedTxt1, cos, sin, numImg, numTxt);

        ApplyGatedResidual(img, imgAttn, numImg, imgG1);
        ApplyGatedResidual(txt, txtAttn, numTxt, txtG1);

        var normedImg2 = Modulate(img, numImg, imgS2, imgSc2);
        var normedTxt2 = Modulate(txt, numTxt, txtS2, txtSc2);

        var imgMlp = FeedForward($"{prefix}.img_mlp", normedImg2, numImg);
        var txtMlp = FeedForward($"{prefix}.txt_mlp", normedTxt2, numTxt);

        ApplyGatedResidual(img, imgMlp, numImg, imgG2);
        ApplyGatedResidual(txt, txtMlp, numTxt, txtG2);

        return (img, txt);
    }

    private (float[] imgAttn, float[] txtAttn) JointAttention(
        string prefix,
        float[] img,
        float[] txt,
        float[] cos,
        float[] sin,
        int numImg,
        int numTxt)
    {
        int totalSeq = numTxt + numImg;

        var imgQkv = Linear($"{prefix}.img_attn.qkv", img, _dim, _dim * 3);
        var txtQkv = Linear($"{prefix}.txt_attn.qkv", txt, _dim, _dim * 3);

        var (imgQ, imgK, imgV) = SplitQkv(imgQkv, numImg, _dim);
        var (txtQ, txtK, txtV) = SplitQkv(txtQkv, numTxt, _dim);

        var normImgK = TryGetWeight($"{prefix}.img_attn.norm.key_norm.weight") ?? TryGetWeight($"{prefix}.img_attn.norm.key_norm.scale");
        if (normImgK is not null) RmsNormHeads(imgK, numImg, _numHeads, _headDim, normImgK);
        var normTxtK = TryGetWeight($"{prefix}.txt_attn.norm.key_norm.weight") ?? TryGetWeight($"{prefix}.txt_attn.norm.key_norm.scale");
        if (normTxtK is not null) RmsNormHeads(txtK, numTxt, _numHeads, _headDim, normTxtK);

        var q = ConcatSequences(txtQ, imgQ, numTxt, numImg);
        var k = ConcatSequences(txtK, imgK, numTxt, numImg);
        var v = ConcatSequences(txtV, imgV, numTxt, numImg);

        int ropeSeq = Math.Min(totalSeq, cos.Length / _headDim);
        HunyuanVideoRoPE.ApplyRoPE(q, cos, sin, ropeSeq, _numHeads, _headDim);
        HunyuanVideoRoPE.ApplyRoPE(k, cos, sin, ropeSeq, _numHeads, _headDim);

        var attnOut = MultiHeadAttention(q, k, v, totalSeq, _numHeads, _headDim);

        var txtAttnSlice = attnOut.AsSpan(0, numTxt * _dim).ToArray();
        var imgAttnSlice = attnOut.AsSpan(numTxt * _dim, numImg * _dim).ToArray();

        var finalImg = Linear($"{prefix}.img_attn.proj", imgAttnSlice, _dim, _dim);
        var finalTxt = Linear($"{prefix}.txt_attn.proj", txtAttnSlice, _dim, _dim);

        return (finalImg, finalTxt);
    }

    private float[] SingleBlock(
        string prefix,
        float[] x,
        float[] tEmb,
        float[] cos,
        float[] sin,
        int totalSeq)
    {
        var mod = Linear($"{prefix}.modulation.linear", DiffusionOpsSilu(tEmb), _dim, _dim * 3);
        var s = mod.AsSpan(0 * _dim, _dim);
        var sc = mod.AsSpan(1 * _dim, _dim);
        var g = mod.AsSpan(2 * _dim, _dim);

        var normed = Modulate(x, totalSeq, s, sc);
        var qkvMlp = Linear($"{prefix}.linear1", normed, _dim, _dim * 3 + _dim * 4);

        int qkvLen = _dim * 3;
        int mlpLen = _dim * 4;
        var qkv = new float[totalSeq * qkvLen];
        var mlpUp = new float[totalSeq * mlpLen];

        for (int i = 0; i < totalSeq; i++)
        {
            int srcOff = i * (qkvLen + mlpLen);
            Array.Copy(qkvMlp, srcOff, qkv, i * qkvLen, qkvLen);
            Array.Copy(qkvMlp, srcOff + qkvLen, mlpUp, i * mlpLen, mlpLen);
        }

        var (q, k, v) = SplitQkv(qkv, totalSeq, _dim);

        int ropeSeq = Math.Min(totalSeq, cos.Length / _headDim);
        HunyuanVideoRoPE.ApplyRoPE(q, cos, sin, ropeSeq, _numHeads, _headDim);
        HunyuanVideoRoPE.ApplyRoPE(k, cos, sin, ropeSeq, _numHeads, _headDim);

        var attnOut = MultiHeadAttention(q, k, v, totalSeq, _numHeads, _headDim);

        DiffusionOps.GeluInPlace(mlpUp);

        var fused = new float[totalSeq * (_dim + mlpLen)];
        for (int i = 0; i < totalSeq; i++)
        {
            int dstOff = i * (_dim + mlpLen);
            Array.Copy(attnOut, i * _dim, fused, dstOff, _dim);
            Array.Copy(mlpUp, i * mlpLen, fused, dstOff + _dim, mlpLen);
        }

        var projOut = Linear($"{prefix}.linear2", fused, _dim + mlpLen, _dim);
        ApplyGatedResidual(x, projOut, totalSeq, g);
        return x;
    }

    private static (float[] q, float[] k, float[] v) SplitQkv(float[] qkv, int seqLen, int dim)
    {
        var q = new float[seqLen * dim];
        var k = new float[seqLen * dim];
        var v = new float[seqLen * dim];

        for (int i = 0; i < seqLen; i++)
        {
            int src = i * (dim * 3);
            int dst = i * dim;
            Array.Copy(qkv, src + 0 * dim, q, dst, dim);
            Array.Copy(qkv, src + 1 * dim, k, dst, dim);
            Array.Copy(qkv, src + 2 * dim, v, dst, dim);
        }
        return (q, k, v);
    }

    private static float[] MultiHeadAttention(float[] q, float[] k, float[] v, int seqLen, int numHeads, int headDim)
    {
        float scale = 1.0f / MathF.Sqrt(headDim);
        var output = new float[seqLen * numHeads * headDim];

        for (int h = 0; h < numHeads; h++)
        {
            int hOff = h * headDim;
            for (int i = 0; i < seqLen; i++)
            {
                int qRow = (i * numHeads + h) * headDim;
                var scores = new float[seqLen];
                float maxScore = float.NegativeInfinity;

                for (int j = 0; j < seqLen; j++)
                {
                    int kRow = (j * numHeads + h) * headDim;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                        dot += q[qRow + d] * k[kRow + d];
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
                for (int j = 0; j < seqLen; j++) scores[j] *= invSum;

                int outRow = (i * numHeads + h) * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    float sum = 0f;
                    for (int j = 0; j < seqLen; j++)
                        sum += scores[j] * v[(j * numHeads + h) * headDim + d];
                    output[outRow + d] = sum;
                }
            }
        }

        return output;
    }

    private static void RmsNormHeads(float[] qk, int seqLen, int numHeads, int headDim, float[] gamma)
    {
        for (int s = 0; s < seqLen; s++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int off = (s * numHeads + h) * headDim;
                float sumSq = 0f;
                for (int d = 0; d < headDim; d++)
                {
                    float val = qk[off + d];
                    sumSq += val * val;
                }
                float invRms = 1.0f / MathF.Sqrt(sumSq / headDim + 1e-6f);
                for (int d = 0; d < headDim; d++)
                    qk[off + d] = qk[off + d] * invRms * gamma[d % gamma.Length];
            }
        }
    }

    private static float[] ConcatSequences(float[] a, float[] b, int lenA, int lenB)
    {
        int dim = a.Length / lenA;
        var result = new float[(lenA + lenB) * dim];
        Array.Copy(a, 0, result, 0, lenA * dim);
        Array.Copy(b, 0, result, lenA * dim, lenB * dim);
        return result;
    }

    private float[] FeedForward(string prefix, float[] x, int seqLen)
    {
        int intermediateDim = _dim * 4;
        var h1 = Linear($"{prefix}.0", x, _dim, intermediateDim);
        DiffusionOps.GeluInPlace(h1);
        return Linear($"{prefix}.2", h1, intermediateDim, _dim);
    }

    private float[] Modulate(float[] x, int seqLen, ReadOnlySpan<float> shift, ReadOnlySpan<float> scale)
    {
        var outF = new float[x.Length];
        for (int i = 0; i < seqLen; i++)
        {
            int off = i * _dim;
            for (int d = 0; d < _dim; d++)
            {
                float val = x[off + d];
                outF[off + d] = val * (1.0f + scale[d]) + shift[d];
            }
        }
        return outF;
    }

    private void ApplyGatedResidual(float[] x, float[] branch, int seqLen, ReadOnlySpan<float> gate)
    {
        for (int i = 0; i < seqLen; i++)
        {
            int off = i * _dim;
            for (int d = 0; d < _dim; d++)
                x[off + d] += branch[off + d] * gate[d];
        }
    }

    private float[] ComputeTimestepEmbedding(float timestep)
    {
        var emb = new float[256];
        int half = 128;
        float factor = 10000.0f;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-MathF.Log(factor) * i / half);
            emb[i] = MathF.Cos(timestep * freq);
            emb[half + i] = MathF.Sin(timestep * freq);
        }

        var t0 = Linear("time_in.in_layer", emb, 256, _dim);
        DiffusionOps.SiluInPlace(t0);
        return Linear("time_in.out_layer", t0, _dim, _dim);
    }

    private static float[] DiffusionOpsSilu(float[] x)
    {
        var res = (float[])x.Clone();
        DiffusionOps.SiluInPlace(res);
        return res;
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

    public static float[] PackLatents(float[] latents, int numFrames, int latH, int latW)
    {
        int patchH = latH / 2;
        int patchW = latW / 2;
        int numTokens = numFrames * patchH * patchW;
        var packed = new float[numTokens * InChannels];

        for (int f = 0; f < numFrames; f++)
        {
            for (int ph = 0; ph < patchH; ph++)
            {
                for (int pw = 0; pw < patchW; pw++)
                {
                    int tokenIdx = (f * patchH + ph) * patchW + pw;
                    int tokenOff = tokenIdx * InChannels;
                    int chanOffset = 0;

                    for (int c = 0; c < OutChannels; c++)
                    {
                        for (int dy = 0; dy < 2; dy++)
                        {
                            for (int dx = 0; dx < 2; dx++)
                            {
                                int y = ph * 2 + dy;
                                int x = pw * 2 + dx;
                                int srcIdx = ((c * numFrames + f) * latH + y) * latW + x;
                                packed[tokenOff + chanOffset++] = latents[srcIdx];
                            }
                        }
                    }
                }
            }
        }
        return packed;
    }

    public static float[] UnpackLatents(float[] packed, int numFrames, int latH, int latW)
    {
        int patchH = latH / 2;
        int patchW = latW / 2;
        var unpacked = new float[OutChannels * numFrames * latH * latW];

        for (int f = 0; f < numFrames; f++)
        {
            for (int ph = 0; ph < patchH; ph++)
            {
                for (int pw = 0; pw < patchW; pw++)
                {
                    int tokenIdx = (f * patchH + ph) * patchW + pw;
                    int tokenOff = tokenIdx * InChannels;
                    int chanOffset = 0;

                    for (int c = 0; c < OutChannels; c++)
                    {
                        for (int dy = 0; dy < 2; dy++)
                        {
                            for (int dx = 0; dx < 2; dx++)
                            {
                                int y = ph * 2 + dy;
                                int x = pw * 2 + dx;
                                int dstIdx = ((c * numFrames + f) * latH + y) * latW + x;
                                unpacked[dstIdx] = packed[tokenOff + chanOffset++];
                            }
                        }
                    }
                }
            }
        }
        return unpacked;
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
