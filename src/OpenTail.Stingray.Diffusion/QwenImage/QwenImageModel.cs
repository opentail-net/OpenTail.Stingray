using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.QwenImage;

/// <summary>
/// Native C# Qwen Image 60-layer MM-DiT diffusion transformer model.
/// Supports standard Text-to-Image and Qwen Image Edit (reference visual conditioning).
/// Reference: stable-diffusion.cpp:src/model/diffusion/qwen_image.hpp:QwenImageModel
/// </summary>
public sealed class QwenImageModel : IDisposable
{
    private readonly IWeightLoader _weights;
    private readonly string _prefix;
    private readonly IComputeBackend? _backend;
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    private readonly Dictionary<string, CoreTensor>? _gpuWeights;
    private readonly int _numLayers;
    private bool _disposed;

    public const int HiddenDim = 3072;
    public const int NumHeads = 24;
    public const int HeadDim = 128;
    public const int PatchSize = 2;
    public const int InChannels = 64;   // 16 * 2 * 2
    public const int OutChannels = 16;
    public const int ContextDim = 3584; // Qwen2.5-VL text dimension

    public int NumLayers => _numLayers;

    public QwenImageModel(IWeightLoader weights, string prefix = "", int numLayers = 60, IComputeBackend? backend = null)
    {
        _weights = weights;
        _prefix = prefix;
        _backend = backend;
        _numLayers = DetectNumLayers(weights, prefix, numLayers);
        if (backend is not null)
            _gpuWeights = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
    }

    private static int DetectNumLayers(IWeightLoader weights, string prefix, int fallback)
    {
        for (int i = 80; i >= 0; i--)
        {
            string key = $"{prefix}transformer_blocks.{i}.attn.to_q.weight";
            if (weights.Contains(key) || weights.Contains("model.diffusion_model." + key))
                return i + 1;
        }
        return fallback;
    }

    private string Resolve(string name)
    {
        string direct = _prefix + name;
        if (_weights.Contains(direct)) return direct;
        if (_weights.Contains("model.diffusion_model." + direct)) return "model.diffusion_model." + direct;
        if (_weights.Contains("transformer." + direct)) return "transformer." + direct;
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
    /// Evaluates the Qwen Image transformer forward pass.
    /// Supports optional reference image latents for Qwen Image Edit.
    /// </summary>
    public float[] Forward(
        float[] latent,
        float timestep,
        float[] textContext,
        int latH,
        int latW,
        float[]? refLatent = null)
    {
        int patchH = latH / PatchSize;
        int patchW = latW / PatchSize;
        int numTargetTokens = patchH * patchW;
        int numTxtTokens = textContext.Length / ContextDim;

        // 1. Pack 16-channel latents into 64-channel patches
        var packedTarget = PackLatents(latent, latH, latW);
        float[] packedInput;
        int[]? modulateIndex = null;
        int numImgTokens = numTargetTokens;

        if (refLatent is not null)
        {
            // Qwen Image Edit: concatenate reference visual tokens with target canvas tokens
            var packedRef = PackLatents(refLatent, latH, latW);
            numImgTokens = numTargetTokens + numTargetTokens;
            packedInput = new float[numImgTokens * InChannels];
            Array.Copy(packedTarget, 0, packedInput, 0, packedTarget.Length);
            Array.Copy(packedRef, 0, packedInput, packedTarget.Length, packedRef.Length);

            modulateIndex = new int[numImgTokens];
            for (int i = 0; i < numTargetTokens; i++) modulateIndex[i] = 0; // target: 0
            for (int i = numTargetTokens; i < numImgTokens; i++) modulateIndex[i] = 1; // ref: 1
        }
        else
        {
            packedInput = packedTarget;
        }

        // 2. Input projections
        var imgTokens = Linear("img_in", packedInput, InChannels, HiddenDim);
        var txtTokens = Linear("txt_in", textContext, ContextDim, HiddenDim);

        // 3. Timestep embedding (sinusoidal 256 -> linear 3072 -> silu -> linear 3072)
        var tEmb = ComputeTimestepEmbedding(timestep);

        // 4. 3D-RoPE positional encoding
        var (cos, sin) = QwenImageRoPE.Compute3DRoPE(numTxtTokens, patchH, patchW, HeadDim);

        // 5. 60 MM-DiT Transformer blocks
        for (int b = 0; b < _numLayers; b++)
        {
            string p = $"transformer_blocks.{b}";
            (imgTokens, txtTokens) = TransformerBlock(p, imgTokens, txtTokens, tEmb, cos, sin, numImgTokens, numTxtTokens, modulateIndex);
        }

        // 6. Final layer norm and projection (AdaLN + Linear 3072 -> 64)
        var finalNorm = Linear("final_layer.adaLN_modulation.1", DiffusionOpsSilu(tEmb), HiddenDim, HiddenDim * 2);
        var finalGamma = finalNorm.AsSpan(0, HiddenDim);
        var finalBeta = finalNorm.AsSpan(HiddenDim, HiddenDim);

        // Extract target canvas tokens (first numTargetTokens)
        var targetTokens = imgTokens.AsSpan(0, numTargetTokens * HiddenDim).ToArray();
        var normImg = (float[])targetTokens.Clone();
        DiffusionOps.LayerNorm(normImg, finalGamma, finalBeta, HiddenDim);

        var outPacked = Linear("final_layer.linear", normImg, HiddenDim, InChannels);

        // 7. Unpack patches [numTargetTokens, 64] -> [16, latH, latW]
        return UnpackLatents(outPacked, latH, latW);
    }

    private (float[] imgOut, float[] txtOut) TransformerBlock(
        string prefix,
        float[] img,
        float[] txt,
        float[] tEmb,
        float[] cos,
        float[] sin,
        int numImg,
        int numTxt,
        int[]? modulateIndex)
    {
        // 1. Modulations
        var imgMod = Linear($"{prefix}.img_mod.1", DiffusionOpsSilu(tEmb), HiddenDim, HiddenDim * 6);
        var txtMod = Linear($"{prefix}.txt_mod.1", DiffusionOpsSilu(tEmb), HiddenDim, HiddenDim * 6);

        var imgS1 = imgMod.AsSpan(0 * HiddenDim, HiddenDim);
        var imgSc1 = imgMod.AsSpan(1 * HiddenDim, HiddenDim);
        var imgG1 = imgMod.AsSpan(2 * HiddenDim, HiddenDim);
        var imgS2 = imgMod.AsSpan(3 * HiddenDim, HiddenDim);
        var imgSc2 = imgMod.AsSpan(4 * HiddenDim, HiddenDim);
        var imgG2 = imgMod.AsSpan(5 * HiddenDim, HiddenDim);

        var txtS1 = txtMod.AsSpan(0 * HiddenDim, HiddenDim);
        var txtSc1 = txtMod.AsSpan(1 * HiddenDim, HiddenDim);
        var txtG1 = txtMod.AsSpan(2 * HiddenDim, HiddenDim);
        var txtS2 = txtMod.AsSpan(3 * HiddenDim, HiddenDim);
        var txtSc2 = txtMod.AsSpan(4 * HiddenDim, HiddenDim);
        var txtG2 = txtMod.AsSpan(5 * HiddenDim, HiddenDim);

        // 2. Modulated norm1
        var normedImg1 = Modulate(img, numImg, imgS1, imgSc1, modulateIndex);
        var normedTxt1 = Modulate(txt, numTxt, txtS1, txtSc1, null);

        // 3. Joint Attention
        var (imgAttn, txtAttn) = JointAttention($"{prefix}.attn", normedImg1, normedTxt1, cos, sin, numImg, numTxt);

        // 4. Apply gate1 & residual
        ApplyGatedResidual(img, imgAttn, numImg, imgG1, modulateIndex);
        ApplyGatedResidual(txt, txtAttn, numTxt, txtG1, null);

        // 5. Modulated norm2 + MLP
        var normedImg2 = Modulate(img, numImg, imgS2, imgSc2, modulateIndex);
        var normedTxt2 = Modulate(txt, numTxt, txtS2, txtSc2, null);

        var imgMlp = FeedForward($"{prefix}.img_mlp", normedImg2, numImg);
        var txtMlp = FeedForward($"{prefix}.txt_mlp", normedTxt2, numTxt);

        ApplyGatedResidual(img, imgMlp, numImg, imgG2, modulateIndex);
        ApplyGatedResidual(txt, txtMlp, numTxt, txtG2, null);

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

        var imgQ = Linear($"{prefix}.to_q", img, HiddenDim, HiddenDim);
        var imgK = Linear($"{prefix}.to_k", img, HiddenDim, HiddenDim);
        var imgV = Linear($"{prefix}.to_v", img, HiddenDim, HiddenDim);

        var txtQ = Linear($"{prefix}.add_q_proj", txt, HiddenDim, HiddenDim);
        var txtK = Linear($"{prefix}.add_k_proj", txt, HiddenDim, HiddenDim);
        var txtV = Linear($"{prefix}.add_v_proj", txt, HiddenDim, HiddenDim);

        RmsNormHeads(imgQ, numImg, NumHeads, HeadDim, GetWeight($"{prefix}.norm_q.weight"));
        RmsNormHeads(imgK, numImg, NumHeads, HeadDim, GetWeight($"{prefix}.norm_k.weight"));
        RmsNormHeads(txtQ, numTxt, NumHeads, HeadDim, GetWeight($"{prefix}.norm_added_q.weight"));
        RmsNormHeads(txtK, numTxt, NumHeads, HeadDim, GetWeight($"{prefix}.norm_added_k.weight"));

        var q = ConcatSequences(txtQ, imgQ, numTxt, numImg);
        var k = ConcatSequences(txtK, imgK, numTxt, numImg);
        var v = ConcatSequences(txtV, imgV, numTxt, numImg);

        // Apply 3D-RoPE (repeated if multi-token edit sequence)
        int ropeSeq = Math.Min(totalSeq, cos.Length / HeadDim);
        QwenImageRoPE.ApplyRoPE(q, cos, sin, ropeSeq, NumHeads, HeadDim);
        QwenImageRoPE.ApplyRoPE(k, cos, sin, ropeSeq, NumHeads, HeadDim);

        var attnOut = MultiHeadAttention(q, k, v, totalSeq, NumHeads, HeadDim);

        var txtAttnSlice = attnOut.AsSpan(0, numTxt * HiddenDim).ToArray();
        var imgAttnSlice = attnOut.AsSpan(numTxt * HiddenDim, numImg * HiddenDim).ToArray();

        var finalImg = Linear($"{prefix}.to_out.0", imgAttnSlice, HiddenDim, HiddenDim);
        var finalTxt = Linear($"{prefix}.to_add_out", txtAttnSlice, HiddenDim, HiddenDim);

        return (finalImg, finalTxt);
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
                    qk[off + d] = qk[off + d] * invRms * gamma[d];
            }
        }
    }

    private static float[] ConcatSequences(float[] a, float[] b, int lenA, int lenB)
    {
        var result = new float[(lenA + lenB) * HiddenDim];
        Array.Copy(a, 0, result, 0, lenA * HiddenDim);
        Array.Copy(b, 0, result, lenA * HiddenDim, lenB * HiddenDim);
        return result;
    }

    private float[] FeedForward(string prefix, float[] x, int seqLen)
    {
        int intermediateDim = HiddenDim * 4;
        var up = Linear($"{prefix}.net.0.proj", x, HiddenDim, intermediateDim * 2);
        var result = new float[seqLen * intermediateDim];

        for (int i = 0; i < seqLen; i++)
        {
            int srcOff = i * (intermediateDim * 2);
            int dstOff = i * intermediateDim;
            for (int d = 0; d < intermediateDim; d++)
            {
                float val = up[srcOff + d];
                float gate = up[srcOff + intermediateDim + d];
                result[dstOff + d] = val * DiffusionOps.Gelu(gate);
            }
        }

        return Linear($"{prefix}.net.2", result, intermediateDim, HiddenDim);
    }

    private static float[] Modulate(float[] x, int seqLen, ReadOnlySpan<float> shift, ReadOnlySpan<float> scale, int[]? index)
    {
        var outF = new float[x.Length];
        for (int i = 0; i < seqLen; i++)
        {
            int off = i * HiddenDim;
            float factor = (index is not null && index[i] == 1) ? 0.0f : 1.0f;
            for (int d = 0; d < HiddenDim; d++)
            {
                float val = x[off + d];
                outF[off + d] = val * (1.0f + scale[d] * factor) + shift[d] * factor;
            }
        }
        return outF;
    }

    private static void ApplyGatedResidual(float[] x, float[] branch, int seqLen, ReadOnlySpan<float> gate, int[]? index)
    {
        for (int i = 0; i < seqLen; i++)
        {
            int off = i * HiddenDim;
            float factor = (index is not null && index[i] == 1) ? 1.0f : gate[0];
            for (int d = 0; d < HiddenDim; d++)
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

        var t0 = Linear("time_text_embed.timestep_embedder.linear_1", emb, 256, HiddenDim);
        DiffusionOps.SiluInPlace(t0);
        return Linear("time_text_embed.timestep_embedder.linear_2", t0, HiddenDim, HiddenDim);
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
        return DiffusionOps.Linear(x, w, b, rows, inDim, outDim);
    }

    public static float[] PackLatents(float[] latents, int latH, int latW)
    {
        int patchH = latH / PatchSize;
        int patchW = latW / PatchSize;
        int numTokens = patchH * patchW;
        var packed = new float[numTokens * InChannels];

        for (int ph = 0; ph < patchH; ph++)
        {
            for (int pw = 0; pw < patchW; pw++)
            {
                int tokenIdx = ph * patchW + pw;
                int tokenOff = tokenIdx * InChannels;
                int chanOffset = 0;

                for (int c = 0; c < OutChannels; c++)
                {
                    for (int dy = 0; dy < PatchSize; dy++)
                    {
                        for (int dx = 0; dx < PatchSize; dx++)
                            packed[tokenOff + chanOffset++] = latents[(c * latH + (ph * PatchSize + dy)) * latW + (pw * PatchSize + dx)];
                    }
                }
            }
        }
        return packed;
    }

    public static float[] UnpackLatents(float[] packed, int latH, int latW)
    {
        int patchH = latH / PatchSize;
        int patchW = latW / PatchSize;
        var unpacked = new float[OutChannels * latH * latW];

        for (int ph = 0; ph < patchH; ph++)
        {
            for (int pw = 0; pw < patchW; pw++)
            {
                int tokenIdx = ph * patchW + pw;
                int tokenOff = tokenIdx * InChannels;
                int chanOffset = 0;

                for (int c = 0; c < OutChannels; c++)
                {
                    for (int dy = 0; dy < PatchSize; dy++)
                    {
                        for (int dx = 0; dx < PatchSize; dx++)
                            unpacked[(c * latH + (ph * PatchSize + dy)) * latW + (pw * PatchSize + dx)] = packed[tokenOff + chanOffset++];
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
