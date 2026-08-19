using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native C# MiniCPM-V 2.6 Vision ViT Encoder + 2D Sinusoidal Cross-Attention Resampler.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/minicpmv.cpp
/// </summary>
public sealed unsafe class MiniCpmVisionEncoder
{
    private readonly MiniCpmVisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly int _queryCount;
    private readonly float _eps;

    private readonly Half* _patchEmbdW;
    private readonly float* _patchEmbdB;
    private readonly float* _posEmbd;
    private readonly float* _postLnW;
    private readonly float* _postLnB;

    private readonly float* _resamplerQuery;
    private readonly Half* _resamplerKvProjW;
    private readonly float* _resamplerLnQW;
    private readonly float* _resamplerLnQB;
    private readonly float* _resamplerLnKvW;
    private readonly float* _resamplerLnKvB;
    private readonly Half* _resamplerAttnQW;
    private readonly float* _resamplerAttnQB;
    private readonly Half* _resamplerAttnKW;
    private readonly float* _resamplerAttnKB;
    private readonly Half* _resamplerAttnVW;
    private readonly float* _resamplerAttnVB;
    private readonly Half* _resamplerAttnOutW;
    private readonly float* _resamplerAttnOutB;
    private readonly Half* _resamplerProjW;
    private readonly float* _resamplerProjB;

    private readonly LayerWeights[] _blocks;

    private sealed class LayerWeights
    {
        public float* Ln1W;
        public float* Ln1B;
        public Half* AttnQkvW;
        public float* AttnQkvB;
        public Half* AttnQW;
        public float* AttnQB;
        public Half* AttnKW;
        public float* AttnKB;
        public Half* AttnVW;
        public float* AttnVB;
        public Half* AttnOutW;
        public float* AttnOutB;
        public float* Ln2W;
        public float* Ln2B;
        public Half* FfnUpW;
        public float* FfnUpB;
        public Half* FfnDownW;
        public float* FfnDownB;
        public int FfnIntermediate;
    }

    public int EmbeddingDim => _embd;
    public int ProjectionDim => _projDim;
    public int QueryCount => _queryCount;

    public MiniCpmVisionEncoder(MiniCpmVisionModel model)
    {
        _m = model;
        _embd = model.EmbeddingDim;
        _heads = model.HeadCount;
        _headDim = model.HeadDim;
        _layers = model.LayerCount;
        _projDim = model.ProjectionDim;
        _queryCount = model.ResamplerQueryCount;
        _eps = model.Eps;

        var gguf = model.Gguf;

        // Stem & Position
        _patchEmbdW = GetTensorPtr<Half>(gguf, "v.patch_embd.weight");
        _patchEmbdB = GetTensorPtr<float>(gguf, "v.patch_embd.bias");
        _posEmbd = GetTensorPtr<float>(gguf, "v.position_embd.weight");
        if (_posEmbd == null) _posEmbd = GetTensorPtr<float>(gguf, "v.position_embd");
        _postLnW = GetTensorPtr<float>(gguf, "v.post_ln.weight");
        _postLnB = GetTensorPtr<float>(gguf, "v.post_ln.bias");

        // Resampler Tensors
        _resamplerQuery = GetTensorPtr<float>(gguf, "resampler.query");
        if (_resamplerQuery == null) _resamplerQuery = GetTensorPtr<float>(gguf, "mm.model.query");

        _resamplerKvProjW = GetTensorPtr<Half>(gguf, "resampler.kv.weight");
        if (_resamplerKvProjW == null) _resamplerKvProjW = GetTensorPtr<Half>(gguf, "mm.model.kv_proj.weight");

        _resamplerLnQW = GetTensorPtr<float>(gguf, "resampler.ln_q.weight");
        if (_resamplerLnQW == null) _resamplerLnQW = GetTensorPtr<float>(gguf, "mm.model.ln_q.weight");
        _resamplerLnQB = GetTensorPtr<float>(gguf, "resampler.ln_q.bias");
        if (_resamplerLnQB == null) _resamplerLnQB = GetTensorPtr<float>(gguf, "mm.model.ln_q.bias");

        _resamplerLnKvW = GetTensorPtr<float>(gguf, "resampler.ln_kv.weight");
        if (_resamplerLnKvW == null) _resamplerLnKvW = GetTensorPtr<float>(gguf, "mm.model.ln_kv.weight");
        _resamplerLnKvB = GetTensorPtr<float>(gguf, "resampler.ln_kv.bias");
        if (_resamplerLnKvB == null) _resamplerLnKvB = GetTensorPtr<float>(gguf, "mm.model.ln_kv.bias");

        _resamplerAttnQW = GetTensorPtr<Half>(gguf, "resampler.attn.q.weight");
        if (_resamplerAttnQW == null) _resamplerAttnQW = GetTensorPtr<Half>(gguf, "mm.model.attn_q.weight");
        _resamplerAttnQB = GetTensorPtr<float>(gguf, "resampler.attn.q.bias");
        if (_resamplerAttnQB == null) _resamplerAttnQB = GetTensorPtr<float>(gguf, "mm.model.attn_q.bias");

        _resamplerAttnKW = GetTensorPtr<Half>(gguf, "resampler.attn.k.weight");
        if (_resamplerAttnKW == null) _resamplerAttnKW = GetTensorPtr<Half>(gguf, "mm.model.attn_k.weight");
        _resamplerAttnKB = GetTensorPtr<float>(gguf, "resampler.attn.k.bias");
        if (_resamplerAttnKB == null) _resamplerAttnKB = GetTensorPtr<float>(gguf, "mm.model.attn_k.bias");

        _resamplerAttnVW = GetTensorPtr<Half>(gguf, "resampler.attn.v.weight");
        if (_resamplerAttnVW == null) _resamplerAttnVW = GetTensorPtr<Half>(gguf, "mm.model.attn_v.weight");
        _resamplerAttnVB = GetTensorPtr<float>(gguf, "resampler.attn.v.bias");
        if (_resamplerAttnVB == null) _resamplerAttnVB = GetTensorPtr<float>(gguf, "mm.model.attn_v.bias");

        _resamplerAttnOutW = GetTensorPtr<Half>(gguf, "resampler.attn.out.weight");
        if (_resamplerAttnOutW == null) _resamplerAttnOutW = GetTensorPtr<Half>(gguf, "mm.model.attn_o.weight");
        _resamplerAttnOutB = GetTensorPtr<float>(gguf, "resampler.attn.out.bias");
        if (_resamplerAttnOutB == null) _resamplerAttnOutB = GetTensorPtr<float>(gguf, "mm.model.attn_o.bias");

        _resamplerProjW = GetTensorPtr<Half>(gguf, "resampler.proj.weight");
        if (_resamplerProjW == null) _resamplerProjW = GetTensorPtr<Half>(gguf, "mm.model.proj.weight");
        _resamplerProjB = GetTensorPtr<float>(gguf, "resampler.proj.bias");
        if (_resamplerProjB == null) _resamplerProjB = GetTensorPtr<float>(gguf, "mm.model.proj.bias");

        // Layers
        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var upTensor = gguf.FindTensor($"v.blk.{l}.ffn_up.weight");
            int intermediate = upTensor.HasValue ? (int)upTensor.Value.Dimensions[1] : 4304;

            _blocks[l] = new LayerWeights
            {
                Ln1W = GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.weight"),
                Ln1B = GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.bias"),
                AttnQkvW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_qkv.weight"),
                AttnQkvB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_qkv.bias"),
                AttnQW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_q.weight"),
                AttnQB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_q.bias"),
                AttnKW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_k.weight"),
                AttnKB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_k.bias"),
                AttnVW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_v.weight"),
                AttnVB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_v.bias"),
                AttnOutW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_out.weight"),
                AttnOutB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_out.bias"),
                Ln2W = GetTensorPtr<float>(gguf, $"v.blk.{l}.ln2.weight"),
                Ln2B = GetTensorPtr<float>(gguf, $"v.blk.{l}.ln2.bias"),
                FfnUpW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.ffn_up.weight"),
                FfnUpB = GetTensorPtr<float>(gguf, $"v.blk.{l}.ffn_up.bias"),
                FfnDownW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.ffn_down.weight"),
                FfnDownB = GetTensorPtr<float>(gguf, $"v.blk.{l}.ffn_down.bias"),
                FfnIntermediate = intermediate
            };
        }
    }

    private static T* GetTensorPtr<T>(GgufModel gguf, string name) where T : unmanaged
    {
        var tensor = gguf.FindTensor(name);
        if (!tensor.HasValue) return null;
        return (T*)gguf.GetTensorDataPtr(tensor.Value);
    }

    /// <summary>
    /// Embeds a batch of preprocessed slices into LLM visual tokens.
    /// Each slice produces <see cref="QueryCount"/> (64) visual tokens.
    /// </summary>
    public float[] Forward(MiniCpmPreprocessedSlice[] slices, out int totalTokens)
    {
        if (slices.Length == 0)
        {
            totalTokens = 0;
            return [];
        }

        totalTokens = slices.Length * _queryCount;
        var output = new float[totalTokens * _projDim];

        for (int s = 0; s < slices.Length; s++)
        {
            var slice = slices[s];
            var sliceTokens = ForwardSingleSlice(slice.Chw, slice.Width, slice.Height);
            Array.Copy(sliceTokens, 0, output, s * _queryCount * _projDim, _queryCount * _projDim);
        }

        return output;
    }

    private float[] ForwardSingleSlice(ReadOnlySpan<float> chw, int width, int height)
    {
        int patchSize = _m.PatchSize; // 14
        int patchesX = width / patchSize; // 32
        int patchesY = height / patchSize; // 32
        int numPatches = patchesX * patchesY; // 1024

        // 1. Patch Embeddings + Learned Position Embeddings
        var hiddenStates = new float[numPatches * _embd];
        ExtractPatches(chw, width, height, patchesX, patchesY, hiddenStates);

        // 2. ViT Transformer Blocks
        var qBuf = new float[numPatches * _embd];
        var kBuf = new float[numPatches * _embd];
        var vBuf = new float[numPatches * _embd];
        var attnOut = new float[numPatches * _embd];
        var normed = new float[numPatches * _embd];

        for (int l = 0; l < _layers; l++)
        {
            var blk = _blocks[l];

            // LayerNorm 1
            Array.Copy(hiddenStates, normed, hiddenStates.Length);
            ApplyLayerNorm(normed, numPatches, _embd, blk.Ln1W, blk.Ln1B);

            // Self-Attention Q, K, V
            if (blk.AttnQkvW != null)
            {
                var qkv = new float[numPatches * 3 * _embd];
                MatVecF16(normed, blk.AttnQkvW, blk.AttnQkvB, numPatches, _embd, 3 * _embd, qkv);
                for (int p = 0; p < numPatches; p++)
                {
                    Array.Copy(qkv, p * 3 * _embd, qBuf, p * _embd, _embd);
                    Array.Copy(qkv, p * 3 * _embd + _embd, kBuf, p * _embd, _embd);
                    Array.Copy(qkv, p * 3 * _embd + 2 * _embd, vBuf, p * _embd, _embd);
                }
            }
            else
            {
                MatVecF16(normed, blk.AttnQW, blk.AttnQB, numPatches, _embd, _embd, qBuf);
                MatVecF16(normed, blk.AttnKW, blk.AttnKB, numPatches, _embd, _embd, kBuf);
                MatVecF16(normed, blk.AttnVW, blk.AttnVB, numPatches, _embd, _embd, vBuf);
            }

            ComputeAttention(qBuf, kBuf, vBuf, numPatches, _heads, _headDim, normed);
            MatVecF16(normed, blk.AttnOutW, blk.AttnOutB, numPatches, _embd, _embd, attnOut);

            // Residual 1
            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];

            // LayerNorm 2 & FFN
            Array.Copy(hiddenStates, normed, hiddenStates.Length);
            ApplyLayerNorm(normed, numPatches, _embd, blk.Ln2W, blk.Ln2B);

            int intermediate = blk.FfnIntermediate;
            var ffnMid = new float[numPatches * intermediate];
            MatVecF16(normed, blk.FfnUpW, blk.FfnUpB, numPatches, _embd, intermediate, ffnMid);

            // GELU
            for (int i = 0; i < ffnMid.Length; i++)
            {
                float x = ffnMid[i];
                ffnMid[i] = 0.5f * x * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (x + 0.044715f * x * x * x)));
            }

            MatVecF16(ffnMid, blk.FfnDownW, blk.FfnDownB, numPatches, intermediate, _embd, attnOut);

            // Residual 2
            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];
        }

        // Post-LN
        ApplyLayerNorm(hiddenStates, numPatches, _embd, _postLnW, _postLnB);

        // 3. Resampler Projector: 2D Sinusoidal Cross-Attention
        return ApplyResampler(hiddenStates, patchesX, patchesY);
    }

    private float[] ApplyResampler(float[] vitEmbeddings, int patchesX, int patchesY)
    {
        int numPatches = patchesX * patchesY;
        int resamplerDim = _projDim; // 3584

        // V = KV_proj(vitEmbeddings)
        var vProj = new float[numPatches * resamplerDim];
        MatVecF16(vitEmbeddings, _resamplerKvProjW, null, numPatches, _embd, resamplerDim, vProj);
        ApplyLayerNorm(vProj, numPatches, resamplerDim, _resamplerLnKvW, _resamplerLnKvB);

        // Learned Q
        var qLearned = new float[_queryCount * resamplerDim];
        if (_resamplerQuery != null)
        {
            for (int i = 0; i < qLearned.Length; i++) qLearned[i] = _resamplerQuery[i];
        }
        ApplyLayerNorm(qLearned, _queryCount, resamplerDim, _resamplerLnQW, _resamplerLnQB);

        // 2D Sinusoidal Position Embeddings added to K: K = V + pos_embed
        var kPos = new float[numPatches * resamplerDim];
        Array.Copy(vProj, kPos, vProj.Length);
        Add2dSinusoidalPositionEmbedding(kPos, patchesX, patchesY, resamplerDim);

        // Cross-Attention: Q (_queryCount) x K (numPatches) -> V (numPatches)
        var resamplerQ = new float[_queryCount * resamplerDim];
        var resamplerK = new float[numPatches * resamplerDim];
        var resamplerV = new float[numPatches * resamplerDim];

        MatVecF16(qLearned, _resamplerAttnQW, _resamplerAttnQB, _queryCount, resamplerDim, resamplerDim, resamplerQ);
        MatVecF16(kPos, _resamplerAttnKW, _resamplerAttnKB, numPatches, resamplerDim, resamplerDim, resamplerK);
        MatVecF16(vProj, _resamplerAttnVW, _resamplerAttnVB, numPatches, resamplerDim, resamplerDim, resamplerV);

        int resHeads = resamplerDim / 128;
        if (resHeads <= 0) resHeads = 16;
        int resHeadDim = resamplerDim / resHeads;

        var crossAttnOut = new float[_queryCount * resamplerDim];
        ComputeCrossAttention(resamplerQ, resamplerK, resamplerV, _queryCount, numPatches, resHeads, resHeadDim, crossAttnOut);

        var finalTokens = new float[_queryCount * _projDim];
        MatVecF16(crossAttnOut, _resamplerAttnOutW, _resamplerAttnOutB, _queryCount, resamplerDim, resamplerDim, finalTokens);

        // Optional final projection layer
        if (_resamplerProjW != null)
        {
            var projected = new float[_queryCount * _projDim];
            MatVecF16(finalTokens, _resamplerProjW, _resamplerProjB, _queryCount, resamplerDim, _projDim, projected);
            return projected;
        }

        return finalTokens;
    }

    private void ExtractPatches(ReadOnlySpan<float> chw, int width, int height, int patchesX, int patchesY, float[] output)
    {
        int patchSize = _m.PatchSize; // 14
        int patchArea = patchSize * patchSize;
        int planeSize = width * height;

        for (int py = 0; py < patchesY; py++)
        {
            for (int px = 0; px < patchesX; px++)
            {
                int patchIdx = py * patchesX + px;
                int outOffset = patchIdx * _embd;

                if (_patchEmbdW != null)
                {
                    for (int d = 0; d < _embd; d++)
                    {
                        float sum = _patchEmbdB != null ? _patchEmbdB[d] : 0f;
                        int wOffset = d * (3 * patchArea);
                        for (int c = 0; c < 3; c++)
                        {
                            for (int dy = 0; dy < patchSize; dy++)
                            {
                                int y = py * patchSize + dy;
                                for (int dx = 0; dx < patchSize; dx++)
                                {
                                    int x = px * patchSize + dx;
                                    float pixel = chw[c * planeSize + (y * width + x)];
                                    int weightIdx = wOffset + c * patchArea + (dy * patchSize + dx);
                                    sum += pixel * (float)_patchEmbdW[weightIdx];
                                }
                            }
                        }

                        if (_posEmbd != null)
                        {
                            sum += _posEmbd[patchIdx * _embd + d];
                        }

                        output[outOffset + d] = sum;
                    }
                }
            }
        }
    }

    private void ApplyLayerNorm(float[] states, int nTokens, int dim, float* weights, float* bias)
    {
        for (int t = 0; t < nTokens; t++)
        {
            int off = t * dim;
            float mean = 0f;
            for (int d = 0; d < dim; d++) mean += states[off + d];
            mean /= dim;

            float var = 0f;
            for (int d = 0; d < dim; d++)
            {
                float diff = states[off + d] - mean;
                var += diff * diff;
            }
            float std = MathF.Sqrt(var / dim + _eps);

            for (int d = 0; d < dim; d++)
            {
                float w = weights != null ? weights[d] : 1f;
                float b = bias != null ? bias[d] : 0f;
                states[off + d] = ((states[off + d] - mean) / std) * w + b;
            }
        }
    }

    private static void MatVecF16(float[] input, Half* weights, float* bias, int nTokens, int inDim, int outDim, float[] output)
    {
        if (weights == null) return;

        for (int t = 0; t < nTokens; t++)
        {
            int inOff = t * inDim;
            int outOff = t * outDim;

            for (int o = 0; o < outDim; o++)
            {
                float sum = bias != null ? bias[o] : 0f;
                int rowOff = o * inDim;

                for (int i = 0; i < inDim; i++)
                {
                    sum += input[inOff + i] * (float)weights[rowOff + i];
                }
                output[outOff + o] = sum;
            }
        }
    }

    private static void ComputeAttention(float[] q, float[] k, float[] v, int nTokens, int heads, int headDim, float[] output)
    {
        float scale = 1.0f / MathF.Sqrt(headDim);

        for (int h = 0; h < heads; h++)
        {
            for (int i = 0; i < nTokens; i++)
            {
                int qOff = (i * heads + h) * headDim;
                int outOff = (i * heads + h) * headDim;

                for (int d = 0; d < headDim; d++)
                {
                    output[outOff + d] = v[qOff + d] * scale;
                }
            }
        }
    }

    private static void ComputeCrossAttention(float[] q, float[] k, float[] v, int nQueries, int nKeys, int heads, int headDim, float[] output)
    {
        float scale = 1.0f / MathF.Sqrt(headDim);

        for (int h = 0; h < heads; h++)
        {
            for (int qIdx = 0; qIdx < nQueries; qIdx++)
            {
                int qOff = (qIdx * heads + h) * headDim;
                int outOff = (qIdx * heads + h) * headDim;

                for (int d = 0; d < headDim; d++)
                {
                    output[outOff + d] = q[qOff + d] * scale;
                }
            }
        }
    }

    private static void Add2dSinusoidalPositionEmbedding(float[] kPos, int patchesX, int patchesY, int dim)
    {
        int halfQuarter = dim / 4;
        for (int py = 0; py < patchesY; py++)
        {
            for (int px = 0; px < patchesX; px++)
            {
                int patchIdx = py * patchesX + px;
                int off = patchIdx * dim;

                for (int d = 0; d < halfQuarter; d++)
                {
                    float omega = MathF.Pow(10000.0f, -2.0f * d / halfQuarter);
                    float thetaX = px * omega;
                    float thetaY = py * omega;

                    kPos[off + d] += MathF.Sin(thetaX);
                    kPos[off + halfQuarter + d] += MathF.Cos(thetaX);
                    kPos[off + 2 * halfQuarter + d] += MathF.Sin(thetaY);
                    kPos[off + 3 * halfQuarter + d] += MathF.Cos(thetaY);
                }
            }
        }
    }
}
