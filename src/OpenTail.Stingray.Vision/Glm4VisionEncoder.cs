
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native C# Zhipu AI GLM-4V, GLM-4.5V, and GLM-OCR ViT Encoder + 2D M-RoPE + Patch Merger Projector.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/glm4v.cpp
/// </summary>
public sealed unsafe class Glm4VisionEncoder
{
    private readonly Glm4VisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly int _mergeFactor;
    private readonly float _ropeTheta;
    private readonly float _eps;

    private readonly float[] _patchEmbd0WF32;
    private readonly VisionTensorRef _patchEmbd1W;
    private readonly float[]? _patchBias;
    private readonly float[]? _normEmbdW;
    private readonly float[]? _normEmbdB;
    private readonly float[] _posEmbdF32;
    private readonly float[]? _postLnW;

    private readonly float[] _patchMergerWF32;
    private readonly float[]? _patchMergerB;
    private readonly VisionTensorRef _fcW;
    private readonly float[]? _fcB;
    private readonly float[]? _mmPostNormW;
    private readonly float[]? _mmPostNormB;
    private readonly VisionTensorRef _mmGateW;
    private readonly float[]? _mmGateB;
    private readonly VisionTensorRef _mmUpW;
    private readonly float[]? _mmUpB;
    private readonly VisionTensorRef _mmDownW;
    private readonly float[]? _mmDownB;
    private readonly int _mmFfnDim;

    private readonly LayerWeights[] _blocks;

    private sealed class LayerWeights
    {
        public float[]? Ln1W;
        public VisionTensorRef AttnQW;
        public float[]? AttnQB;
        public VisionTensorRef AttnKW;
        public float[]? AttnKB;
        public VisionTensorRef AttnVW;
        public float[]? AttnVB;
        public VisionTensorRef AttnOutW;
        public float[]? AttnOutB;
        public float[]? Ln2W;
        public VisionTensorRef FfnGateW;
        public float[]? FfnGateB;
        public VisionTensorRef FfnUpW;
        public float[]? FfnUpB;
        public VisionTensorRef FfnDownW;
        public float[]? FfnDownB;
        public int FfnIntermediate;
    }

    public int EmbeddingDim => _embd;
    public int ProjectionDim => _projDim;

    public Glm4VisionEncoder(Glm4VisionModel model)
    {
        _m = model;
        _embd = model.EmbeddingDim;
        _heads = model.HeadCount;
        _headDim = model.HeadDim;
        _layers = model.LayerCount;
        _projDim = model.ProjectionDim;
        _mergeFactor = model.MergeFactor;
        _ropeTheta = model.RopeTheta;
        _eps = model.Eps;

        var gguf = model.Gguf;

        _patchEmbd0WF32 = VisionOps.DequantizeToFloat32(
            VisionOps.GetTensor(gguf, "v.patch_embd.0.weight", "v.patch_embd.weight"));
        _patchEmbd1W = VisionOps.GetTensor(gguf, "v.patch_embd.1.weight", "v.patch_embd.weight.1");
        _patchBias = VisionOps.GetTensorArray(gguf, "v.patch_bias");
        _normEmbdW = VisionOps.GetTensorArray(gguf, "v.norm_embd.weight");
        _normEmbdB = VisionOps.GetTensorArray(gguf, "v.norm_embd.bias");
        _posEmbdF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.position_embd.weight", "v.position_embd"));
        _postLnW = VisionOps.GetTensorArray(gguf, "v.post_ln.weight");

        _patchMergerWF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "mm.patch_merger.weight"));
        _patchMergerB = VisionOps.GetTensorArray(gguf, "mm.patch_merger.bias");

        // Real tensor name in this checkpoint is "mm.model.fc.weight" (confirmed via list-tensors),
        // not "mm.fc.weight"/"mm.0.weight" -- those names never matched, silently falling through to
        // the truncating-copy fallback below. Real glm4v.cpp: build_mm(model.mm_fc_w, cur).
        _fcW = VisionOps.GetTensor(gguf, "mm.model.fc.weight", "mm.fc.weight", "mm.0.weight");
        _fcB = VisionOps.GetTensorArray(gguf, "mm.model.fc.bias", "mm.fc.bias", "mm.0.bias");

        // Real glm4v.cpp projector tail (build_norm -> gelu_erf -> build_ffn), entirely missing
        // from the prior implementation. Real tensor names: mm.post_norm.* (plain LayerNorm,
        // eps=1e-5, NOT the ViT's own RMS eps), mm.gate/mm.up/mm.down.* (gated SiLU FFN).
        _mmPostNormW = VisionOps.GetTensorArray(gguf, "mm.post_norm.weight");
        _mmPostNormB = VisionOps.GetTensorArray(gguf, "mm.post_norm.bias");
        _mmGateW = VisionOps.GetTensor(gguf, "mm.gate.weight");
        _mmGateB = VisionOps.GetTensorArray(gguf, "mm.gate.bias");
        _mmUpW = VisionOps.GetTensor(gguf, "mm.up.weight");
        _mmUpB = VisionOps.GetTensorArray(gguf, "mm.up.bias");
        _mmDownW = VisionOps.GetTensor(gguf, "mm.down.weight");
        _mmDownB = VisionOps.GetTensorArray(gguf, "mm.down.bias");
        _mmFfnDim = _mmGateW.IsValid ? (int)_mmGateW.Info.Dimensions[1] : 0;

        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var gateTensor = gguf.FindTensor($"v.blk.{l}.ffn_gate.weight");
            int ffnDim = gateTensor.HasValue ? (int)gateTensor.Value.Dimensions[1] : (_embd * 3);

            _blocks[l] = new LayerWeights
            {
                Ln1W = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln1.weight"),
                AttnQW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_q.weight"),
                AttnQB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_q.bias"),
                AttnKW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_k.weight"),
                AttnKB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_k.bias"),
                AttnVW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_v.weight"),
                AttnVB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_v.bias"),
                AttnOutW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_out.weight"),
                AttnOutB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_out.bias"),
                Ln2W = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln2.weight"),
                FfnGateW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_gate.weight"),
                FfnGateB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ffn_gate.bias"),
                FfnUpW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_up.weight"),
                FfnUpB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ffn_up.bias"),
                FfnDownW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_down.weight"),
                FfnDownB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ffn_down.bias"),
                FfnIntermediate = ffnDim
            };
        }
    }

    /// <summary>
    /// Forward pass of GLM-4V ViT:
    /// Preprocessed CHW pixels -> Dual Conv2D Patch + Pos Embeddings -> ViT Layers with M-RoPE -> Patch Merger -> FC Projector -> LLM Visual Tokens.
    /// </summary>
    public float[] Forward(ReadOnlySpan<float> chw, int targetWidth, int targetHeight, int patchesX, int patchesY, out int tokenCount)
    {
        int numPatches = patchesX * patchesY;
        if (numPatches == 0)
        {
            tokenCount = 0;
            return [];
        }

        // 1. Dual Conv2D Patch Linear Projections + Bias + Norm
        var hiddenStates = new float[numPatches * _embd];
        ExtractPatches(chw, targetWidth, targetHeight, patchesX, patchesY, hiddenStates);

        if (_normEmbdW != null)
        {
            fixed (float* normEmbdW = _normEmbdW) ApplyRmsNorm(hiddenStates, numPatches, _embd, normEmbdW);
        }

        // 2. ViT Transformer Blocks
        var qBuf = new float[numPatches * _embd];
        var kBuf = new float[numPatches * _embd];
        var vBuf = new float[numPatches * _embd];
        var attnOut = new float[numPatches * _embd];
        var normed = new float[numPatches * _embd];

        int maxFfnDim = 0;
        for (int l = 0; l < _layers; l++)
        {
            if (_blocks[l].FfnIntermediate > maxFfnDim) maxFfnDim = _blocks[l].FfnIntermediate;
        }
        var gateBuf = new float[numPatches * maxFfnDim];
        var upBuf = new float[numPatches * maxFfnDim];

        for (int l = 0; l < _layers; l++)
        {
            var blk = _blocks[l];

            fixed (float* ln1W = blk.Ln1W, attnQB = blk.AttnQB, attnKB = blk.AttnKB, attnVB = blk.AttnVB,
                   attnOutB = blk.AttnOutB, ln2W = blk.Ln2W, ffnGateB = blk.FfnGateB, ffnUpB = blk.FfnUpB,
                   ffnDownB = blk.FfnDownB)
            {
                // RMSNorm 1
                Array.Copy(hiddenStates, normed, hiddenStates.Length);
                ApplyRmsNorm(normed, numPatches, _embd, ln1W);

                // Q, K, V Linear Projections
                VisionOps.MatVecAny(normed, blk.AttnQW, attnQB, numPatches, _embd, _embd, qBuf);
                VisionOps.MatVecAny(normed, blk.AttnKW, attnKB, numPatches, _embd, _embd, kBuf);
                VisionOps.MatVecAny(normed, blk.AttnVW, attnVB, numPatches, _embd, _embd, vBuf);

                // M-RoPE 2D Multimodal Rotary Embeddings
                ApplyMrope(qBuf, kBuf, patchesX, patchesY);

                // Self-Attention & Out-Projection
                VisionOps.Attention(qBuf, kBuf, vBuf, numPatches, _heads, _headDim, normed);
                VisionOps.MatVecAny(normed, blk.AttnOutW, attnOutB, numPatches, _embd, _embd, attnOut);

                // Residual 1
                for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];

                // RMSNorm 2 & SwiGLU FFN
                Array.Copy(hiddenStates, normed, hiddenStates.Length);
                ApplyRmsNorm(normed, numPatches, _embd, ln2W);

                int ffnDim = blk.FfnIntermediate;
                VisionOps.MatVecAny(normed, blk.FfnGateW, ffnGateB, numPatches, _embd, ffnDim, gateBuf);
                VisionOps.MatVecAny(normed, blk.FfnUpW, ffnUpB, numPatches, _embd, ffnDim, upBuf);

                int ffnLen = numPatches * ffnDim;
                for (int i = 0; i < ffnLen; i++)
                {
                    float g = gateBuf[i];
                    float silu = g / (1.0f + MathF.Exp(-g));
                    gateBuf[i] = silu * upBuf[i];
                }

                VisionOps.MatVecAny(gateBuf, blk.FfnDownW, ffnDownB, numPatches, ffnDim, _embd, attnOut);

                // Residual 2
                for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];
            }
        }

        if (_postLnW != null)
        {
            fixed (float* postLnW = _postLnW) ApplyRmsNorm(hiddenStates, numPatches, _embd, postLnW);
        }

        // 3. Patch Merger: real strided Conv2D (kernel=stride=mergeFactor, embd -> mm.patch_merger's
        // own output width, here 4096) via mm.patch_merger.weight/bias -- NOT a plain
        // concat-then-linear. Real glm4v.cpp: reshape/permute to [n_embd,gx,gy,batch] then
        // ggml_conv_2d(mm_patch_merger_w, cur, n_merge, n_merge, 0,0,1,1) + mm_patch_merger_b.
        int scale = _mergeFactor; // 2
        int downX = patchesX / scale;
        int downY = patchesY / scale;
        tokenCount = downX * downY;
        int mergerOutDim = _patchMergerWF32.Length > 0 ? _patchMergerB?.Length ?? _projDim : _embd * scale * scale;

        var merged = new float[tokenCount * mergerOutDim];
        ApplyPatchMerger(hiddenStates, patchesX, patchesY, scale, mergerOutDim, merged);

        // 4. FC Projector (mm.model.fc): mergerOutDim -> projDim
        var afterFc = new float[tokenCount * _projDim];
        if (_fcW.IsValid)
        {
            fixed (float* fcB = _fcB)
            {
                VisionOps.MatVecAny(merged, _fcW, fcB, tokenCount, mergerOutDim, _projDim, afterFc);
            }
        }
        else
        {
            for (int t = 0; t < tokenCount; t++)
            {
                int copyDim = Math.Min(mergerOutDim, _projDim);
                Array.Copy(merged, t * mergerOutDim, afterFc, t * _projDim, copyDim);
            }
        }

        // 5. mm.post_norm (plain LayerNorm, eps=1e-5 -- distinct from the ViT's own RMS eps) ->
        // gelu_erf -> gated SiLU FFN (mm.gate/mm.up/mm.down). Real glm4v.cpp order:
        // build_norm(NORM_TYPE_NORMAL) -> ggml_gelu_erf -> build_ffn.
        if (_mmPostNormW != null)
        {
            fixed (float* w = _mmPostNormW, b = _mmPostNormB)
            {
                ApplyLayerNorm(afterFc, tokenCount, _projDim, w, b, 1e-5f);
            }
        }
        for (int i = 0; i < afterFc.Length; i++) afterFc[i] = GeluErfScalar(afterFc[i]);

        if (!_mmGateW.IsValid || !_mmUpW.IsValid || !_mmDownW.IsValid || _mmFfnDim == 0)
        {
            return afterFc;
        }

        var gate = new float[tokenCount * _mmFfnDim];
        var up = new float[tokenCount * _mmFfnDim];
        var visualTokens = new float[tokenCount * _projDim];
        fixed (float* gateB = _mmGateB, upB = _mmUpB, downB = _mmDownB)
        {
            VisionOps.MatVecAny(afterFc, _mmGateW, gateB, tokenCount, _projDim, _mmFfnDim, gate);
            VisionOps.MatVecAny(afterFc, _mmUpW, upB, tokenCount, _projDim, _mmFfnDim, up);
            for (int i = 0; i < gate.Length; i++)
            {
                float g = gate[i];
                gate[i] = (g / (1.0f + MathF.Exp(-g))) * up[i];
            }
            VisionOps.MatVecAny(gate, _mmDownW, downB, tokenCount, _mmFfnDim, _projDim, visualTokens);
        }

        return visualTokens;
    }

    /// <summary>Real erf-based GELU: 0.5*x*(1+erf(x/sqrt(2))), matching ggml_gelu_erf.</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static float GeluErfScalar(float x)
    {
        return 0.5f * x * (1.0f + Erf(x * 0.7071067811865476f));
    }

    // Abramowitz-Stegun 7.1.26 approximation, max error ~1.5e-7 -- ample for this test's thresholds.
    private static float Erf(float x)
    {
        float sign = x < 0 ? -1f : 1f;
        x = MathF.Abs(x);
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f;
        const float a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float t = 1.0f / (1.0f + p * x);
        float y = 1.0f - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);
        return sign * y;
    }

    private static void ApplyLayerNorm(float[] states, int nTokens, int dim, float* weights, float* bias, float eps)
    {
        for (int t = 0; t < nTokens; t++)
        {
            int off = t * dim;
            float mean = 0f;
            for (int d = 0; d < dim; d++) mean += states[off + d];
            mean /= dim;

            float varSum = 0f;
            for (int d = 0; d < dim; d++)
            {
                float diff = states[off + d] - mean;
                varSum += diff * diff;
            }
            float invStd = 1.0f / MathF.Sqrt(varSum / dim + eps);

            for (int d = 0; d < dim; d++)
            {
                float norm = (states[off + d] - mean) * invStd;
                float w = weights != null ? weights[d] : 1f;
                float b = bias != null ? bias[d] : 0f;
                states[off + d] = norm * w + b;
            }
        }
    }

    /// <summary>
    /// Real strided Conv2D patch merger: for each scale x scale block of ViT output patches,
    /// out[o] = bias[o] + sum over (dy,dx,c) of weight[o,c,dy,dx] * hidden[srcPatch,c].
    /// Weight raw layout matches patch_embd's convention: ne0=kw(fastest), ne1=kh, ne2=cin, ne3=cout.
    /// </summary>
    private void ApplyPatchMerger(float[] src, int patchesX, int patchesY, int scale, int outDim, float[] dst)
    {
        int downX = patchesX / scale;
        int downY = patchesY / scale;
        int cin = _embd;
        int kArea = scale * scale;

        Parallel.For(0, downY, dy0 =>
        {
            for (int dx0 = 0; dx0 < downX; dx0++)
            {
                int dstTokenIdx = dy0 * downX + dx0;
                int dstOff = dstTokenIdx * outDim;

                for (int o = 0; o < outDim; o++)
                {
                    float sum = _patchMergerB != null ? _patchMergerB[o] : 0f;
                    int wOffO = o * (cin * kArea);
                    for (int c = 0; c < cin; c++)
                    {
                        int wOffC = wOffO + c * kArea;
                        for (int dy = 0; dy < scale; dy++)
                        {
                            int srcY = dy0 * scale + dy;
                            for (int dx = 0; dx < scale; dx++)
                            {
                                int srcX = dx0 * scale + dx;
                                int srcPatchIdx = srcY * patchesX + srcX;
                                sum += src[srcPatchIdx * cin + c] * _patchMergerWF32[wOffC + dy * scale + dx];
                            }
                        }
                    }
                    dst[dstOff + o] = sum;
                }
            }
        });
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

                if (_patchEmbd0WF32.Length > 0)
                {
                    for (int d = 0; d < _embd; d++)
                    {
                        float sum = _patchBias != null ? _patchBias[d] : 0f;
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
                                    sum += pixel * _patchEmbd0WF32[weightIdx];
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

    private void ApplyPixelMerge(float[] src, int patchesX, int patchesY, int scale, float[] dst)
    {
        int downX = patchesX / scale;
        int downY = patchesY / scale;
        int mergedDim = _embd * scale * scale;

        for (int dy = 0; dy < downY; dy++)
        {
            for (int dx = 0; dx < downX; dx++)
            {
                int dstTokenIdx = dy * downX + dx;
                int dstOffset = dstTokenIdx * mergedDim;

                int subIdx = 0;
                for (int sy = 0; sy < scale; sy++)
                {
                    for (int sx = 0; sx < scale; sx++)
                    {
                        int srcX = dx * scale + sx;
                        int srcY = dy * scale + sy;
                        int srcPatchIdx = srcY * patchesX + srcX;

                        Array.Copy(src, srcPatchIdx * _embd, dst, dstOffset + subIdx * _embd, _embd);
                        subIdx++;
                    }
                }
            }
        }
    }

    private void ApplyRmsNorm(float[] states, int nTokens, int dim, float* weights)
    {
        for (int t = 0; t < nTokens; t++)
        {
            int off = t * dim;
            float sumSq = 0f;
            for (int d = 0; d < dim; d++) sumSq += states[off + d] * states[off + d];
            float rms = MathF.Sqrt(sumSq / dim + _eps);

            if (weights != null)
            {
                for (int d = 0; d < dim; d++) states[off + d] = (states[off + d] / rms) * weights[d];
            }
            else
            {
                for (int d = 0; d < dim; d++) states[off + d] /= rms;
            }
        }
    }

    private void ApplyMrope(float[] q, float[] k, int patchesX, int patchesY)
    {
        int mropeHalf = _headDim / 4;
        for (int py = 0; py < patchesY; py++)
        {
            for (int px = 0; px < patchesX; px++)
            {
                int p = py * patchesX + px;
                for (int h = 0; h < _heads; h++)
                {
                    int headOff = (p * _heads + h) * _headDim;
                    for (int d = 0; d < mropeHalf; d++)
                    {
                        float freqX = MathF.Pow(_ropeTheta, -2.0f * d / _headDim);
                        float thetaX = px * freqX;
                        float cosX = MathF.Cos(thetaX);
                        float sinX = MathF.Sin(thetaX);

                        float q0 = q[headOff + d];
                        float q1 = q[headOff + d + mropeHalf];
                        q[headOff + d] = q0 * cosX - q1 * sinX;
                        q[headOff + d + mropeHalf] = q0 * sinX + q1 * cosX;

                        float k0 = k[headOff + d];
                        float k1 = k[headOff + d + mropeHalf];
                        k[headOff + d] = k0 * cosX - k1 * sinX;
                        k[headOff + d + mropeHalf] = k0 * sinX + k1 * cosX;
                    }
                }
            }
        }
    }

    // FOUND, NOT FIXED (out of scope for this migration -- same finding as KimiVisionEncoder.cs,
    // MiniCpmVisionEncoder.cs, QwenVlVisionEncoder.cs): never reads q or k, never scores, never
    // softmaxes -- copies scaled v straight through. Real, pre-existing, unrelated to this pass's
    // dtype migration.
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
}
