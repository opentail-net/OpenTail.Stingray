
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
    private readonly float[]? _postLnB;
    private readonly DeepstackWeights[] _deepstack;

    private sealed class DeepstackWeights
    {
        public int Layer;
        public float[]? NormW, NormB, Fc1B, Fc2B;
        public VisionTensorRef Fc1W, Fc2W;
    }
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
        _patchBias = VisionOps.GetTensorArray(gguf, "v.patch_bias", "v.patch_embd.bias");
        _positionEmbdF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.position_embd.weight", "v.position_embd"));
        _postLnW = VisionOps.GetTensorArray(gguf, "v.post_ln.weight");
        _postLnB = VisionOps.GetTensorArray(gguf, "v.post_ln.bias");
        _deepstack = model.DeepstackLayers.Select(l => new DeepstackWeights
        {
            Layer = l,
            NormW = VisionOps.GetTensorArray(gguf, $"v.deepstack.{l}.norm.weight"),
            NormB = VisionOps.GetTensorArray(gguf, $"v.deepstack.{l}.norm.bias"),
            Fc1W = VisionOps.GetTensor(gguf, $"v.deepstack.{l}.fc1.weight"),
            Fc1B = VisionOps.GetTensorArray(gguf, $"v.deepstack.{l}.fc1.bias"),
            Fc2W = VisionOps.GetTensor(gguf, $"v.deepstack.{l}.fc2.weight"),
            Fc2B = VisionOps.GetTensorArray(gguf, $"v.deepstack.{l}.fc2.bias"),
        }).ToArray();

        _mm0W = VisionOps.GetTensor(gguf, "mm.0.weight");
        _mm0B = VisionOps.GetTensorArray(gguf, "mm.0.bias");
        _mm2W = VisionOps.GetTensor(gguf, "mm.2.weight", "mm.1.weight");
        _mm2B = VisionOps.GetTensorArray(gguf, "mm.2.bias", "mm.1.bias");

        // Ingest Layer Tensors
        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var gateTensor = gguf.FindTensor($"v.blk.{l}.ffn_gate.weight") ?? gguf.FindTensor($"v.blk.{l}.ffn_up.weight");
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
        var deepstackOut = new float[_deepstack.Length][];

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

                int ffnDim = blk.FfnIntermediate;
                int ffnLen = numPatches * ffnDim;
                if (blk.FfnGateW.IsValid)
                {
                    // SwiGLU FFN: SiLU(gate) * up
                    VisionOps.MatVecAny(normed, blk.FfnGateW, ffnGateB, numPatches, _embd, ffnDim, gateBuf);
                    VisionOps.MatVecAny(normed, blk.FfnUpW, ffnUpB, numPatches, _embd, ffnDim, upBuf);
                    for (int i = 0; i < ffnLen; i++)
                    {
                        float g = gateBuf[i];
                        float silu = g / (1.0f + MathF.Exp(-g));
                        gateBuf[i] = silu * upBuf[i];
                    }
                }
                else
                {
                    // Qwen3-VL: GELU(up), no gate (clip.use_gelu, FFN_GELU)
                    VisionOps.MatVecAny(normed, blk.FfnUpW, ffnUpB, numPatches, _embd, ffnDim, gateBuf);
                    for (int i = 0; i < ffnLen; i++) gateBuf[i] = Gelu(gateBuf[i]);
                }

                // Down Projection
                VisionOps.MatVecAny(gateBuf, blk.FfnDownW, ffnDownB, numPatches, ffnDim, _embd, attnOut);

                // Residual 2
                for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];
            }

            for (int d = 0; d < _deepstack.Length; d++)
                if (_deepstack[d].Layer == l)
                    deepstackOut[d] = Deepstack(_deepstack[d], hiddenStates, patchesX, patchesY);
        }

        // 3. Post-Norm
        fixed (float* postLnW = _postLnW, postLnB = _postLnB) ApplyNorm(hiddenStates, numPatches, _embd, postLnW, postLnB);

        // 4. 2x2 Spatial Merge & Multimodal MLP Projection (5120 -> 5120 -> 3584)
        var visualTokens = new float[tokenCount * _projDim];
        ApplySpatialMergeAndMlp(hiddenStates, patchesX, patchesY, visualTokens);
        if (_deepstack.Length == 0) return visualTokens;

        // Qwen3-VL: each token is [projection | deepstack 0 | deepstack 1 | ...] (ggml_concat on the feature dim).
        int width = _projDim * (1 + _deepstack.Length);
        var wide = new float[tokenCount * width];
        for (int t = 0; t < tokenCount; t++)
        {
            Array.Copy(visualTokens, t * _projDim, wide, t * width, _projDim);
            for (int d = 0; d < _deepstack.Length; d++)
                Array.Copy(deepstackOut[d], t * _projDim, wide, t * width + (1 + d) * _projDim, _projDim);
        }
        return wide;
    }

    /// <summary>ggml_gelu (tanh approximation), as the projector below uses.</summary>
    private static float Gelu(float x) => 0.5f * x * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (x + 0.044715f * x * x * x)));

    /// <summary>
    /// One Qwen3-VL deepstack branch on a layer's output: 2x2 merge to 4*embd per token, LayerNorm, fc1, GELU, fc2
    /// (llama.cpp qwen3vl.cpp, layer.has_deepstack()). Tokens come out in merge-tile raster order, like the main
    /// projector.
    /// </summary>
    private float[] Deepstack(DeepstackWeights w, float[] hidden, int patchesX, int patchesY)
    {
        int merge = _m.SpatialMergeFactor, mergedX = patchesX / merge, mergedY = patchesY / merge;
        int n = mergedX * mergedY, dim = merge * merge * _embd;
        var merged = new float[n * dim];
        for (int my = 0; my < mergedY; my++)
            for (int mx = 0; mx < mergedX; mx++)
            {
                int sub = 0, off = (my * mergedX + mx) * dim;
                for (int dy = 0; dy < merge; dy++)
                    for (int dx = 0; dx < merge; dx++)
                        Array.Copy(hidden, ((my * merge + dy) * patchesX + mx * merge + dx) * _embd, merged, off + _embd * sub++, _embd);
            }
        var mid = new float[n * dim];
        var output = new float[n * _projDim];
        fixed (float* nw = w.NormW, nb = w.NormB, b1 = w.Fc1B, b2 = w.Fc2B)
        {
            LayerNorm(merged, n, dim, nw, nb);
            VisionOps.MatVecAny(merged, w.Fc1W, b1, n, dim, dim, mid);
            for (int i = 0; i < mid.Length; i++) mid[i] = Gelu(mid[i]);
            VisionOps.MatVecAny(mid, w.Fc2W, b2, n, dim, _projDim, output);
        }
        return output;
    }

    private void LayerNorm(float[] states, int rows, int dim, float* weights, float* bias)
    {
        for (int p = 0; p < rows; p++)
        {
            int off = p * dim;
            float mean = 0f;
            for (int d = 0; d < dim; d++) mean += states[off + d];
            mean /= dim;
            float var = 0f;
            for (int d = 0; d < dim; d++) { float diff = states[off + d] - mean; var += diff * diff; }
            float inv = 1f / MathF.Sqrt(var / dim + _eps);
            for (int d = 0; d < dim; d++)
                states[off + d] = (states[off + d] - mean) * inv * (weights != null ? weights[d] : 1f) + (bias != null ? bias[d] : 0f);
        }
    }

    /// <summary>
    /// Qwen3-VL learned position grid (sqrt(n) x sqrt(n)) resized to the patch grid with ggml's bilinear
    /// align-corners interpolation (clip.cpp resize_position_embeddings). Row-major (py, px), like the patches.
    /// </summary>
    private float[] ResizedPositionEmbedding(int patchesX, int patchesY)
    {
        int side = (int)MathF.Round(MathF.Sqrt(_positionEmbdF32.Length / _embd));
        if (side == patchesX && side == patchesY) return _positionEmbdF32;
        var output = new float[patchesX * patchesY * _embd];
        float sfx = patchesX > 1 && side > 1 ? (float)(patchesX - 1) / (side - 1) : (float)patchesX / side;
        float sfy = patchesY > 1 && side > 1 ? (float)(patchesY - 1) / (side - 1) : (float)patchesY / side;
        for (int py = 0; py < patchesY; py++)
        {
            float y = py / sfy;
            int y0 = Math.Clamp((int)MathF.Floor(y), 0, side - 1), y1 = Math.Clamp(y0 + 1, 0, side - 1);
            float dy = Math.Clamp(y - y0, 0f, 1f);
            for (int px = 0; px < patchesX; px++)
            {
                float x = px / sfx;
                int x0 = Math.Clamp((int)MathF.Floor(x), 0, side - 1), x1 = Math.Clamp(x0 + 1, 0, side - 1);
                float dx = Math.Clamp(x - x0, 0f, 1f);
                int o = (py * patchesX + px) * _embd;
                int a = (y0 * side + x0) * _embd, b = (y0 * side + x1) * _embd, c = (y1 * side + x0) * _embd, d = (y1 * side + x1) * _embd;
                for (int k = 0; k < _embd; k++)
                    output[o + k] = _positionEmbdF32[a + k] * (1 - dx) * (1 - dy) + _positionEmbdF32[b + k] * dx * (1 - dy)
                                  + _positionEmbdF32[c + k] * (1 - dx) * dy + _positionEmbdF32[d + k] * dx * dy;
            }
        }
        return output;
    }

    private void ExtractPatchEmbeddings(ReadOnlySpan<float> chw, int width, int height, int patchesX, int patchesY, float[] output)
    {
        int patchSize = _m.PatchSize; // 14
        int patchArea = patchSize * patchSize;
        int totalPixels = width * height;
        var positionEmbd = _m.IsQwen3Vl ? ResizedPositionEmbedding(patchesX, patchesY) : _positionEmbdF32;

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

                        if (positionEmbd.Length > 0)
                        {
                            sum += positionEmbd[patchIdx * _embd + d];
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
    // llama.cpp qwen2vl.cpp: ggml_rope_multi(..., d_head/2, {d/4 x4}, GGML_ROPE_TYPE_VISION): rows on the
    // first quarter-pairs, columns on the second, and each section restarts the frequency ladder
    // (indep_sects). The earlier port continued one ladder across both sections -- the same bug
    // dots.ocr had; found 2026-09-27 with QwenVl rainbow448 parity against llama-mtmd-debug.
    private void ApplyMrope(float[] q, float[] k, int patchesX, int patchesY) =>
        VisionOps.ApplyMRoPE(q, k, patchesX, patchesY, _heads, _heads, _headDim, theta: 10000.0f, independentSections: true);

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
