
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native C# Qwen2-VL, Qwen2.5-VL, and Qwen3-VL Vision ViT Encoder + 2x2 Spatial Merger + Multimodal Projector.
/// Binds real Float16 / Float32 memory-mapped weights directly from the GGUF container.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/qwen2vl.cpp and qwen3vl.cpp
/// </summary>
public sealed unsafe class QwenVlVisionEncoder
{
    private readonly QwenVlVisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly bool _useRmsNorm;
    private readonly float _eps;

    private readonly float[] _patchEmbd0WF32;
    private readonly float[] _patchEmbd1WF32;
    private readonly float[]? _patchBias;
    private readonly float[] _positionEmbdF32;
    private readonly float[]? _postLnW;
    private readonly VisionTensorRef _mm0W;
    private readonly float[]? _mm0B;
    private readonly VisionTensorRef _mm2W;
    private readonly float[]? _mm2B;

    private readonly LayerWeights[] _blocks;

    private sealed class LayerWeights
    {
        public float[]? Ln1W;
        public float[]? Ln1B;
        public VisionTensorRef AttnQW;
        public float[]? AttnQB;
        public VisionTensorRef AttnKW;
        public float[]? AttnKB;
        public VisionTensorRef AttnVW;
        public float[]? AttnVB;
        public VisionTensorRef AttnQkvW;
        public float[]? AttnQkvB;
        public VisionTensorRef AttnOutW;
        public float[]? AttnOutB;
        public float[]? Ln2W;
        public float[]? Ln2B;
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

    public QwenVlVisionEncoder(QwenVlVisionModel model)
    {
        _m = model;
        _embd = model.EmbeddingDim;
        _heads = model.HeadCount;
        _headDim = model.HeadDim;
        _layers = model.LayerCount;
        _projDim = model.ProjectionDim;
        _useRmsNorm = model.UseRmsNorm;
        _eps = model.Eps;

        var gguf = model.Gguf;

        // Ingest Stem & Global Tensors
        _patchEmbd0WF32 = VisionOps.DequantizeToFloat32(
            VisionOps.GetTensor(gguf, "v.patch_embd.weight", "v.patch_embd.0.weight"));
        // Real Qwen2.5-VL sums TWO conv2d patch embeddings (temporal Conv3D split into two
        // Conv2Ds, same convention GLM4V/Exaone4/MimoVl all share) -- this second one was fetched
        // into an unused VisionTensorRef and never applied. Confirmed present in this checkpoint
        // via list-tensors (v.patch_embd.weight.1).
        _patchEmbd1WF32 = VisionOps.DequantizeToFloat32(
            VisionOps.GetTensor(gguf, "v.patch_embd.weight.1", "v.patch_embd.1.weight"));
        _patchBias = VisionOps.GetTensorArray(gguf, "v.patch_bias");
        _positionEmbdF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.position_embd.weight", "v.position_embd"));
        _postLnW = VisionOps.GetTensorArray(gguf, "v.post_ln.weight");

        _mm0W = VisionOps.GetTensor(gguf, "mm.0.weight");
        _mm0B = VisionOps.GetTensorArray(gguf, "mm.0.bias");
        _mm2W = VisionOps.GetTensor(gguf, "mm.2.weight", "mm.1.weight");
        _mm2B = VisionOps.GetTensorArray(gguf, "mm.2.bias", "mm.1.bias");

        // Ingest Layer Tensors
        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var gateTensor = gguf.FindTensor($"v.blk.{l}.ffn_gate.weight");
            int ffnDim = gateTensor.HasValue ? (int)gateTensor.Value.Dimensions[1] : 3420;

            _blocks[l] = new LayerWeights
            {
                Ln1W = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln1.weight"),
                Ln1B = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln1.bias"),
                AttnQW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_q.weight"),
                AttnQB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_q.bias"),
                AttnKW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_k.weight"),
                AttnKB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_k.bias"),
                AttnVW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_v.weight"),
                AttnVB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_v.bias"),
                AttnQkvW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_qkv.weight"),
                AttnQkvB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_qkv.bias"),
                AttnOutW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_out.weight"),
                AttnOutB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_out.bias"),
                Ln2W = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln2.weight"),
                Ln2B = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln2.bias"),
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
    /// Executes the full Qwen-VL ViT encoder pipeline:
    /// Preprocessed CHW pixels -> Dual Conv2D Patch Embeddings -> ViT Layers with M-RoPE -> Post-Norm -> 2x2 Spatial Merge MLP -> LLM Visual Tokens.
    /// </summary>
    public float[] Forward(ReadOnlySpan<float> chw, int targetWidth, int targetHeight, out int tokenCount)
    {
        int patchSize = _m.PatchSize; // 14
        int mergeFactor = _m.SpatialMergeFactor; // 2
        int patchesX = targetWidth / patchSize;
        int patchesY = targetHeight / patchSize;
        int numPatches = patchesX * patchesY;
        tokenCount = (patchesX / mergeFactor) * (patchesY / mergeFactor);

        if (numPatches == 0) return [];

        // 1. Conv2D Patch Embeddings: (3 x 14 x 14 -> 1280)
        var hiddenStates = new float[numPatches * _embd];
        ExtractPatchEmbeddings(chw, targetWidth, targetHeight, patchesX, patchesY, hiddenStates);

        // Windowed/local attention (real qwen2vl.cpp, shared with exaone4_5.cpp): layer il gets
        // FULL attention only when (il+1) % waPattern == 0; every other layer only attends within
        // a spatial window of gridWindow x gridWindow MERGE-TILES. See Exaone4VisionEncoder's
        // matching comment / VisionOps.AttentionGqaWindowed's doc comment for the real-reference
        // derivation and why deriving window membership from real (row,col) needs no reordering.
        bool useWindowAttn = _m.WindowAttnPattern > 0;
        int[]? windowId = null;
        if (useWindowAttn)
        {
            int gridWindow = Math.Max(1, _m.WindowSize / patchSize / mergeFactor);
            int mergeCols = Math.Max(1, patchesX / mergeFactor);
            int windowCols = (mergeCols + gridWindow - 1) / gridWindow;
            windowId = new int[numPatches];
            for (int py = 0; py < patchesY; py++)
            {
                int windowRow = (py / mergeFactor) / gridWindow;
                for (int px = 0; px < patchesX; px++)
                {
                    int windowCol = (px / mergeFactor) / gridWindow;
                    windowId[py * patchesX + px] = windowRow * windowCols + windowCol;
                }
            }
        }

        // 2. ViT Transformer Blocks
        var qBuf = new float[numPatches * _embd];
        var kBuf = new float[numPatches * _embd];
        var vBuf = new float[numPatches * _embd];
        var attnOut = new float[numPatches * _embd];
        var normed = new float[numPatches * _embd];
        var qkvBuf = new float[numPatches * 3 * _embd];

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

            fixed (float* ln1W = blk.Ln1W, ln1B = blk.Ln1B, attnQkvB = blk.AttnQkvB, attnQB = blk.AttnQB,
                   attnKB = blk.AttnKB, attnVB = blk.AttnVB, attnOutB = blk.AttnOutB, ln2W = blk.Ln2W,
                   ln2B = blk.Ln2B, ffnGateB = blk.FfnGateB, ffnUpB = blk.FfnUpB, ffnDownB = blk.FfnDownB)
            {
                // Norm 1
                Array.Copy(hiddenStates, normed, hiddenStates.Length);
                ApplyNorm(normed, numPatches, _embd, ln1W, ln1B);

                // Q, K, V Linear Projections
                if (blk.AttnQkvW.IsValid)
                {
                    VisionOps.MatVecAny(normed, blk.AttnQkvW, attnQkvB, numPatches, _embd, 3 * _embd, qkvBuf);
                    for (int p = 0; p < numPatches; p++)
                    {
                        Array.Copy(qkvBuf, p * 3 * _embd, qBuf, p * _embd, _embd);
                        Array.Copy(qkvBuf, p * 3 * _embd + _embd, kBuf, p * _embd, _embd);
                        Array.Copy(qkvBuf, p * 3 * _embd + 2 * _embd, vBuf, p * _embd, _embd);
                    }
                }
                else
                {
                    VisionOps.MatVecAny(normed, blk.AttnQW, attnQB, numPatches, _embd, _embd, qBuf);
                    VisionOps.MatVecAny(normed, blk.AttnKW, attnKB, numPatches, _embd, _embd, kBuf);
                    VisionOps.MatVecAny(normed, blk.AttnVW, attnVB, numPatches, _embd, _embd, vBuf);
                }

                // Multimodal 2D RoPE (M-RoPE)
                ApplyMrope(qBuf, kBuf, patchesX, patchesY);

                // Self-Attention & Out-Projection -- windowed except every waPattern-th layer
                bool fullAttn = !useWindowAttn || (l + 1) % _m.WindowAttnPattern == 0;
                if (fullAttn)
                    VisionOps.Attention(qBuf, kBuf, vBuf, numPatches, _heads, _headDim, normed);
                else
                    VisionOps.AttentionGqaWindowed(qBuf, kBuf, vBuf, numPatches, _heads, _heads, _headDim, normed, windowId!);
                VisionOps.MatVecAny(normed, blk.AttnOutW, attnOutB, numPatches, _embd, _embd, attnOut);

                // Residual 1
                for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];

                // Norm 2
                Array.Copy(hiddenStates, normed, hiddenStates.Length);
                ApplyNorm(normed, numPatches, _embd, ln2W, ln2B);

                // SwiGLU FFN
                int ffnDim = blk.FfnIntermediate;
                VisionOps.MatVecAny(normed, blk.FfnGateW, ffnGateB, numPatches, _embd, ffnDim, gateBuf);
                VisionOps.MatVecAny(normed, blk.FfnUpW, ffnUpB, numPatches, _embd, ffnDim, upBuf);

                // SiLU(gate) * up
                int ffnLen = numPatches * ffnDim;
                for (int i = 0; i < ffnLen; i++)
                {
                    float g = gateBuf[i];
                    float silu = g / (1.0f + MathF.Exp(-g));
                    gateBuf[i] = silu * upBuf[i];
                }

                // Down Projection
                VisionOps.MatVecAny(gateBuf, blk.FfnDownW, ffnDownB, numPatches, ffnDim, _embd, attnOut);

                // Residual 2
                for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];
            }
        }

        // 3. Post-Norm
        fixed (float* postLnW = _postLnW) ApplyNorm(hiddenStates, numPatches, _embd, postLnW, null);

        // 4. 2x2 Spatial Merge & Multimodal MLP Projection (5120 -> 5120 -> 3584)
        var visualTokens = new float[tokenCount * _projDim];
        ApplySpatialMergeAndMlp(hiddenStates, patchesX, patchesY, visualTokens);

        return visualTokens;
    }

    private void ExtractPatchEmbeddings(ReadOnlySpan<float> chw, int width, int height, int patchesX, int patchesY, float[] output)
    {
        int patchSize = _m.PatchSize; // 14
        int patchArea = patchSize * patchSize;
        int totalPixels = width * height;

        for (int py = 0; py < patchesY; py++)
        {
            for (int px = 0; px < patchesX; px++)
            {
                int patchIdx = py * patchesX + px;
                int outOffset = patchIdx * _embd;

                // Conv2D patch linear projection
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
                                    float pixel = chw[c * totalPixels + (y * width + x)];
                                    int weightIdx = wOffset + c * patchArea + (dy * patchSize + dx);
                                    sum += pixel * _patchEmbd0WF32[weightIdx];
                                    if (_patchEmbd1WF32.Length > 0)
                                        sum += pixel * _patchEmbd1WF32[weightIdx];
                                }
                            }
                        }

                        if (_positionEmbdF32.Length > 0)
                        {
                            sum += _positionEmbdF32[patchIdx * _embd + d];
                        }

                        output[outOffset + d] = sum;
                    }
                }
            }
        }
    }

    private void ApplyNorm(float[] states, int nPatches, int dim, float* weights, float* bias)
    {
        for (int p = 0; p < nPatches; p++)
        {
            int off = p * dim;
            if (_useRmsNorm)
            {
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
            else
            {
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
    }

    /// <summary>
    /// Real Qwen2VL-family 4-section M-RoPE (ggml_rope_multi, GGML_ROPE_TYPE_VISION, sections all
    /// = headDim/4, n_dims=headDim/2) -- identical mechanism to GLM4V's (see
    /// Glm4VisionEncoder.ApplyMrope's doc comment for the full derivation from
    /// ggml_mrope_cache_init + rotate_pairs in ggml-cpu/ops.cpp): only 2 of the 4 declared
    /// position channels are ever actually selected, covering the FULL head_dim in two halves
    /// (first quarter of [0,head_dim/2) by row/py, second quarter by column/px, each paired with
    /// its +head_dim/2 partner). The previous version here only rotated the first HALF of
    /// head_dim and used px for every pair (py was never read).
    /// </summary>
    private void ApplyMrope(float[] q, float[] k, int patchesX, int patchesY)
    {
        int half = _headDim / 2;
        int quarter = _headDim / 4;
        for (int py = 0; py < patchesY; py++)
        {
            for (int px = 0; px < patchesX; px++)
            {
                int p = py * patchesX + px;
                for (int h = 0; h < _heads; h++)
                {
                    int headOff = (p * _heads + h) * _headDim;
                    for (int ic = 0; ic < half; ic++)
                    {
                        float pos = ic < quarter ? py : px;
                        float freq = MathF.Pow(10000.0f, -4.0f * ic / _headDim);
                        float theta = pos * freq;
                        float cosT = MathF.Cos(theta);
                        float sinT = MathF.Sin(theta);

                        float q0 = q[headOff + ic];
                        float q1 = q[headOff + ic + half];
                        q[headOff + ic] = q0 * cosT - q1 * sinT;
                        q[headOff + ic + half] = q0 * sinT + q1 * cosT;

                        float k0 = k[headOff + ic];
                        float k1 = k[headOff + ic + half];
                        k[headOff + ic] = k0 * cosT - k1 * sinT;
                        k[headOff + ic + half] = k0 * sinT + k1 * cosT;
                    }
                }
            }
        }
    }

    private void ApplySpatialMergeAndMlp(float[] hiddenStates, int patchesX, int patchesY, float[] visualTokens)
    {
        int merge = _m.SpatialMergeFactor; // 2
        int mergedX = patchesX / merge;
        int mergedY = patchesY / merge;
        int inMlpDim = 4 * _embd; // 5120
        var mergedVec = new float[inMlpDim];
        var midVec = new float[inMlpDim];

        fixed (float* mm0B = _mm0B, mm2B = _mm2B)
        {
            for (int my = 0; my < mergedY; my++)
            {
                for (int mx = 0; mx < mergedX; mx++)
                {
                    int tokenIdx = my * mergedX + mx;
                    int outOffset = tokenIdx * _projDim;

                    // Pack 2x2 = 4 adjacent patches into 5120-dim input vector
                    int subIdx = 0;
                    for (int dy = 0; dy < merge; dy++)
                    {
                        int py = my * merge + dy;
                        for (int dx = 0; dx < merge; dx++)
                        {
                            int px = mx * merge + dx;
                            int patchIdx = py * patchesX + px;
                            Array.Copy(hiddenStates, patchIdx * _embd, mergedVec, subIdx * _embd, _embd);
                            subIdx++;
                        }
                    }

                    // 1. Layer mm.0 (5120 -> 5120) + GELU
                    if (_mm0W.IsValid)
                    {
                        VisionOps.MatVecAny(mergedVec, _mm0W, mm0B, 1, inMlpDim, inMlpDim, midVec);
                        for (int o = 0; o < inMlpDim; o++)
                        {
                            float sum = midVec[o];
                            midVec[o] = 0.5f * sum * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (sum + 0.044715f * sum * sum * sum)));
                        }
                    }
                    else
                    {
                        Array.Copy(mergedVec, midVec, inMlpDim);
                    }

                    // 2. Layer mm.2 (5120 -> 3584)
                    if (_mm2W.IsValid)
                    {
                        var outVec = new float[_projDim];
                        VisionOps.MatVecAny(midVec, _mm2W, mm2B, 1, inMlpDim, _projDim, outVec);
                        Array.Copy(outVec, 0, visualTokens, outOffset, _projDim);
                    }
                }
            }
        }
    }
}
