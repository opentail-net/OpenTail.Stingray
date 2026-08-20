using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

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

    private readonly float[] _patchEmbdWF32;
    private readonly float* _patchEmbdB;
    private readonly float[] _posEmbdF32;
    private readonly float* _postLnW;
    private readonly float* _postLnB;

    private readonly float* _resamplerQuery;
    private readonly VisionTensorRef _resamplerKvProjW;
    private readonly float* _resamplerLnQW;
    private readonly float* _resamplerLnQB;
    private readonly float* _resamplerLnKvW;
    private readonly float* _resamplerLnKvB;
    private readonly VisionTensorRef _resamplerAttnQW;
    private readonly float* _resamplerAttnQB;
    private readonly VisionTensorRef _resamplerAttnKW;
    private readonly float* _resamplerAttnKB;
    private readonly VisionTensorRef _resamplerAttnVW;
    private readonly float* _resamplerAttnVB;
    private readonly VisionTensorRef _resamplerAttnOutW;
    private readonly float* _resamplerAttnOutB;
    private readonly VisionTensorRef _resamplerProjW;
    private readonly float* _resamplerProjB;

    private readonly LayerWeights[] _blocks;

    private sealed class LayerWeights
    {
        public float* Ln1W;
        public float* Ln1B;
        public VisionTensorRef AttnQkvW;
        public float* AttnQkvB;
        public VisionTensorRef AttnQW;
        public float* AttnQB;
        public VisionTensorRef AttnKW;
        public float* AttnKB;
        public VisionTensorRef AttnVW;
        public float* AttnVB;
        public VisionTensorRef AttnOutW;
        public float* AttnOutB;
        public float* Ln2W;
        public float* Ln2B;
        public VisionTensorRef FfnUpW;
        public float* FfnUpB;
        public VisionTensorRef FfnDownW;
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
        _patchEmbdWF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.patch_embd.weight"));
        _patchEmbdB = VisionOps.GetTensorPtr<float>(gguf, "v.patch_embd.bias");
        _posEmbdF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.position_embd.weight", "v.position_embd"));
        _postLnW = VisionOps.GetTensorPtr<float>(gguf, "v.post_ln.weight");
        _postLnB = VisionOps.GetTensorPtr<float>(gguf, "v.post_ln.bias");

        // Resampler Tensors
        _resamplerQuery = VisionOps.GetTensorPtr<float>(gguf, "resampler.query", "mm.model.query");
        _resamplerKvProjW = VisionOps.GetTensor(gguf, "resampler.kv.weight", "mm.model.kv_proj.weight");
        _resamplerLnQW = VisionOps.GetTensorPtr<float>(gguf, "resampler.ln_q.weight", "mm.model.ln_q.weight");
        _resamplerLnQB = VisionOps.GetTensorPtr<float>(gguf, "resampler.ln_q.bias", "mm.model.ln_q.bias");
        _resamplerLnKvW = VisionOps.GetTensorPtr<float>(gguf, "resampler.ln_kv.weight", "mm.model.ln_kv.weight");
        _resamplerLnKvB = VisionOps.GetTensorPtr<float>(gguf, "resampler.ln_kv.bias", "mm.model.ln_kv.bias");
        _resamplerAttnQW = VisionOps.GetTensor(gguf, "resampler.attn.q.weight", "mm.model.attn_q.weight");
        _resamplerAttnQB = VisionOps.GetTensorPtr<float>(gguf, "resampler.attn.q.bias", "mm.model.attn_q.bias");
        _resamplerAttnKW = VisionOps.GetTensor(gguf, "resampler.attn.k.weight", "mm.model.attn_k.weight");
        _resamplerAttnKB = VisionOps.GetTensorPtr<float>(gguf, "resampler.attn.k.bias", "mm.model.attn_k.bias");
        _resamplerAttnVW = VisionOps.GetTensor(gguf, "resampler.attn.v.weight", "mm.model.attn_v.weight");
        _resamplerAttnVB = VisionOps.GetTensorPtr<float>(gguf, "resampler.attn.v.bias", "mm.model.attn_v.bias");
        _resamplerAttnOutW = VisionOps.GetTensor(gguf, "resampler.attn.out.weight", "mm.model.attn_o.weight");
        _resamplerAttnOutB = VisionOps.GetTensorPtr<float>(gguf, "resampler.attn.out.bias", "mm.model.attn_o.bias");
        _resamplerProjW = VisionOps.GetTensor(gguf, "resampler.proj.weight", "mm.model.proj.weight");
        _resamplerProjB = VisionOps.GetTensorPtr<float>(gguf, "resampler.proj.bias", "mm.model.proj.bias");

        // Layers
        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var upTensor = gguf.FindTensor($"v.blk.{l}.ffn_up.weight");
            int intermediate = upTensor.HasValue ? (int)upTensor.Value.Dimensions[1] : 4304;

            _blocks[l] = new LayerWeights
            {
                Ln1W = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.weight"),
                Ln1B = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.bias"),
                AttnQkvW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_qkv.weight"),
                AttnQkvB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_qkv.bias"),
                AttnQW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_q.weight"),
                AttnQB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_q.bias"),
                AttnKW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_k.weight"),
                AttnKB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_k.bias"),
                AttnVW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_v.weight"),
                AttnVB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_v.bias"),
                AttnOutW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_out.weight"),
                AttnOutB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_out.bias"),
                Ln2W = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln2.weight"),
                Ln2B = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln2.bias"),
                FfnUpW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_up.weight"),
                FfnUpB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ffn_up.bias"),
                FfnDownW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_down.weight"),
                FfnDownB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ffn_down.bias"),
                FfnIntermediate = intermediate
            };
        }
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
        var qkv = new float[numPatches * 3 * _embd];

        int maxIntermediate = 0;
        for (int l = 0; l < _layers; l++)
        {
            if (_blocks[l].FfnIntermediate > maxIntermediate) maxIntermediate = _blocks[l].FfnIntermediate;
        }
        var ffnMid = new float[numPatches * maxIntermediate];

        for (int l = 0; l < _layers; l++)
        {
            var blk = _blocks[l];

            // LayerNorm 1
            Array.Copy(hiddenStates, normed, hiddenStates.Length);
            ApplyLayerNorm(normed, numPatches, _embd, blk.Ln1W, blk.Ln1B);

            // Self-Attention Q, K, V
            if (blk.AttnQkvW.IsValid)
            {
                VisionOps.MatVecAny(normed, blk.AttnQkvW, blk.AttnQkvB, numPatches, _embd, 3 * _embd, qkv);
                for (int p = 0; p < numPatches; p++)
                {
                    Array.Copy(qkv, p * 3 * _embd, qBuf, p * _embd, _embd);
                    Array.Copy(qkv, p * 3 * _embd + _embd, kBuf, p * _embd, _embd);
                    Array.Copy(qkv, p * 3 * _embd + 2 * _embd, vBuf, p * _embd, _embd);
                }
            }
            else
            {
                VisionOps.MatVecAny(normed, blk.AttnQW, blk.AttnQB, numPatches, _embd, _embd, qBuf);
                VisionOps.MatVecAny(normed, blk.AttnKW, blk.AttnKB, numPatches, _embd, _embd, kBuf);
                VisionOps.MatVecAny(normed, blk.AttnVW, blk.AttnVB, numPatches, _embd, _embd, vBuf);
            }

            ComputeAttention(qBuf, kBuf, vBuf, numPatches, _heads, _headDim, normed);
            VisionOps.MatVecAny(normed, blk.AttnOutW, blk.AttnOutB, numPatches, _embd, _embd, attnOut);

            // Residual 1
            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];

            // LayerNorm 2 & FFN
            Array.Copy(hiddenStates, normed, hiddenStates.Length);
            ApplyLayerNorm(normed, numPatches, _embd, blk.Ln2W, blk.Ln2B);

            int intermediate = blk.FfnIntermediate;
            VisionOps.MatVecAny(normed, blk.FfnUpW, blk.FfnUpB, numPatches, _embd, intermediate, ffnMid);

            // GELU
            int ffnLen = numPatches * intermediate;
            for (int i = 0; i < ffnLen; i++)
            {
                float x = ffnMid[i];
                ffnMid[i] = 0.5f * x * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (x + 0.044715f * x * x * x)));
            }

            VisionOps.MatVecAny(ffnMid, blk.FfnDownW, blk.FfnDownB, numPatches, intermediate, _embd, attnOut);

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
        VisionOps.MatVecAny(vitEmbeddings, _resamplerKvProjW, null, numPatches, _embd, resamplerDim, vProj);
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

        VisionOps.MatVecAny(qLearned, _resamplerAttnQW, _resamplerAttnQB, _queryCount, resamplerDim, resamplerDim, resamplerQ);
        VisionOps.MatVecAny(kPos, _resamplerAttnKW, _resamplerAttnKB, numPatches, resamplerDim, resamplerDim, resamplerK);
        VisionOps.MatVecAny(vProj, _resamplerAttnVW, _resamplerAttnVB, numPatches, resamplerDim, resamplerDim, resamplerV);

        int resHeads = resamplerDim / 128;
        if (resHeads <= 0) resHeads = 16;
        int resHeadDim = resamplerDim / resHeads;

        var crossAttnOut = new float[_queryCount * resamplerDim];
        ComputeCrossAttention(resamplerQ, resamplerK, resamplerV, _queryCount, numPatches, resHeads, resHeadDim, crossAttnOut);

        var finalTokens = new float[_queryCount * _projDim];
        VisionOps.MatVecAny(crossAttnOut, _resamplerAttnOutW, _resamplerAttnOutB, _queryCount, resamplerDim, resamplerDim, finalTokens);

        // Optional final projection layer
        if (_resamplerProjW.IsValid)
        {
            var projected = new float[_queryCount * _projDim];
            VisionOps.MatVecAny(finalTokens, _resamplerProjW, _resamplerProjB, _queryCount, resamplerDim, _projDim, projected);
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

                if (_patchEmbdWF32.Length > 0)
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
                                    sum += pixel * _patchEmbdWF32[weightIdx];
                                }
                            }
                        }

                        if (_posEmbdF32.Length > 0)
                        {
                            sum += _posEmbdF32[patchIdx * _embd + d];
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

    // FOUND, NOT FIXED (out of scope for this migration -- see docs/vl-migration-plan-2026-08-20.md
    // and the identical finding in KimiVisionEncoder.cs): neither this nor ComputeCrossAttention
    // below actually computes attention. This one never reads q or k, never scores, never
    // softmaxes -- it copies v straight through scaled by 1/sqrt(headDim). ComputeCrossAttention is
    // the same shape of bug but copies q instead (so the resampler's output doesn't depend on the
    // image patches -- k and v -- at all). Both real, pre-existing, unrelated to this pass's
    // GetTensorPtr/MatVecF16 dtype migration.
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
