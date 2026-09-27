
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native C# Moonshot AI Kimi K2.5 and Kimi-VL ViT Encoder + 2D Interleaved RoPE + Patch Merger Projector.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/kimik25.cpp
/// </summary>
public sealed unsafe class KimiVisionEncoder
{
    private readonly KimiVisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly int _mergeFactor;
    private readonly float _ropeTheta;
    private readonly float _eps;

    private readonly float[] _patchEmbdWF32;
    private readonly float[]? _patchEmbdB;
    private readonly float[] _posEmbdF32;
    private readonly float[]? _postLnW;
    private readonly float[]? _postLnB;

    private readonly float[]? _mmInputNormW;
    private readonly float[]? _mmInputNormB;
    private readonly VisionTensorRef _mm1W;
    private readonly float[]? _mm1B;
    private readonly int _mm1OutDim;
    private readonly VisionTensorRef _mm2W;
    private readonly float[]? _mm2B;

    private readonly LayerWeights[] _blocks;

    private sealed class LayerWeights
    {
        public float[]? Ln1W;
        public float[]? Ln1B;
        public VisionTensorRef AttnQkvW;
        public float[]? AttnQkvB;
        public VisionTensorRef AttnQW;
        public float[]? AttnQB;
        public VisionTensorRef AttnKW;
        public float[]? AttnKB;
        public VisionTensorRef AttnVW;
        public float[]? AttnVB;
        public VisionTensorRef AttnOutW;
        public float[]? AttnOutB;
        public float[]? Ln2W;
        public float[]? Ln2B;
        public VisionTensorRef FfnUpW;
        public float[]? FfnUpB;
        public VisionTensorRef FfnDownW;
        public float[]? FfnDownB;
        public int FfnIntermediate;
    }

    public int EmbeddingDim => _embd;
    public int ProjectionDim => _projDim;

    public KimiVisionEncoder(KimiVisionModel model)
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

        _patchEmbdWF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.patch_embd.weight"));
        _patchEmbdB = VisionOps.GetTensorArray(gguf, "v.patch_embd.bias");
        _posEmbdF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.position_embd.weight", "v.position_embd"));
        _postLnW = VisionOps.GetTensorArray(gguf, "v.post_ln.weight");
        _postLnB = VisionOps.GetTensorArray(gguf, "v.post_ln.bias");

        _mmInputNormW = VisionOps.GetTensorArray(gguf, "mm.input_norm.weight");
        _mmInputNormB = VisionOps.GetTensorArray(gguf, "mm.input_norm.bias");
        _mm1W = VisionOps.GetTensor(gguf, "mm.1.weight");
        _mm1B = VisionOps.GetTensorArray(gguf, "mm.1.bias");
        _mm2W = VisionOps.GetTensor(gguf, "mm.2.weight");
        _mm2B = VisionOps.GetTensorArray(gguf, "mm.2.bias");
        // Real bug: mm.1's output width was hardcoded to _projDim (2048), but its real GGUF shape
        // is mergedDim x mergedDim (4608x4608, confirmed via list-tensors) -- mm.1 projects
        // mergedDim->mergedDim, NOT mergedDim->projDim; only mm.2 (4608->2048) narrows to projDim.
        // Telling MatVecAny the wrong (smaller) outDim/inDim across the mm.1->mm.2 boundary made
        // the kernel use the wrong row stride when reading mm.2's real 4608-wide weight rows,
        // reading garbage bytes from adjacent rows and producing exploding (~1e15) output values.
        _mm1OutDim = _mm1W.IsValid ? (int)_mm1W.Info.Dimensions[1] : _projDim;

        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var upTensor = gguf.FindTensor($"v.blk.{l}.ffn_up.weight");
            int intermediate = upTensor.HasValue ? (int)upTensor.Value.Dimensions[1] : (_embd * 4);

            _blocks[l] = new LayerWeights
            {
                Ln1W = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln1.weight"),
                Ln1B = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln1.bias"),
                AttnQkvW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_qkv.weight"),
                AttnQkvB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_qkv.bias"),
                AttnQW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_q.weight"),
                AttnQB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_q.bias"),
                AttnKW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_k.weight"),
                AttnKB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_k.bias"),
                AttnVW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_v.weight"),
                AttnVB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_v.bias"),
                AttnOutW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_out.weight"),
                AttnOutB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_out.bias"),
                Ln2W = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln2.weight"),
                Ln2B = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln2.bias"),
                FfnUpW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_up.weight"),
                FfnUpB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ffn_up.bias"),
                FfnDownW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_down.weight"),
                FfnDownB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ffn_down.bias"),
                FfnIntermediate = intermediate
            };
        }
    }

    /// <summary>
    /// Forward pass of Kimi-VL ViT:
    /// Preprocessed CHW pixels -> Conv2D Patch + Pos Embeddings -> ViT Layers with Interleaved 2D RoPE -> Patch Merger -> 2-layer GELU MLP -> Visual Tokens.
    /// </summary>
    public float[] Forward(ReadOnlySpan<float> chw, int targetWidth, int targetHeight, int patchesX, int patchesY, out int tokenCount)
    {
        int numPatches = patchesX * patchesY;
        if (numPatches == 0)
        {
            tokenCount = 0;
            return [];
        }

        // 1. Patch Embeddings + Learned Position Embeddings
        var hiddenStates = new float[numPatches * _embd];
        ExtractPatches(chw, targetWidth, targetHeight, patchesX, patchesY, hiddenStates);

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

            fixed (float* ln1W = blk.Ln1W, ln1B = blk.Ln1B, attnQkvB = blk.AttnQkvB, attnQB = blk.AttnQB,
                   attnKB = blk.AttnKB, attnVB = blk.AttnVB, attnOutB = blk.AttnOutB, ln2W = blk.Ln2W,
                   ln2B = blk.Ln2B, ffnUpB = blk.FfnUpB, ffnDownB = blk.FfnDownB)
            {
                // LayerNorm 1
                Array.Copy(hiddenStates, normed, hiddenStates.Length);
                ApplyLayerNorm(normed, numPatches, _embd, ln1W, ln1B);

                // Self-Attention Q, K, V
                if (blk.AttnQkvW.IsValid)
                {
                    VisionOps.MatVecAny(normed, blk.AttnQkvW, attnQkvB, numPatches, _embd, 3 * _embd, qkv);
                    for (int p = 0; p < numPatches; p++)
                    {
                        Array.Copy(qkv, p * 3 * _embd, qBuf, p * _embd, _embd);
                        Array.Copy(qkv, p * 3 * _embd + _embd, kBuf, p * _embd, _embd);
                        Array.Copy(qkv, p * 3 * _embd + 2 * _embd, vBuf, p * _embd, _embd);
                    }
                }
                else
                {
                    VisionOps.MatVecAny(normed, blk.AttnQW, attnQB, numPatches, _embd, _embd, qBuf);
                    VisionOps.MatVecAny(normed, blk.AttnKW, attnKB, numPatches, _embd, _embd, kBuf);
                    VisionOps.MatVecAny(normed, blk.AttnVW, attnVB, numPatches, _embd, _embd, vBuf);
                }

                // 2D Interleaved RoPE
                Apply2dInterleavedRope(qBuf, kBuf, patchesX, patchesY);

                // Self-Attention & Out-Projection
                VisionOps.Attention(qBuf, kBuf, vBuf, numPatches, _heads, _headDim, normed);
                VisionOps.MatVecAny(normed, blk.AttnOutW, attnOutB, numPatches, _embd, _embd, attnOut);

                // Residual 1
                for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];

                // LayerNorm 2 & FFN
                Array.Copy(hiddenStates, normed, hiddenStates.Length);
                ApplyLayerNorm(normed, numPatches, _embd, ln2W, ln2B);

                int intermediate = blk.FfnIntermediate;
                VisionOps.MatVecAny(normed, blk.FfnUpW, ffnUpB, numPatches, _embd, intermediate, ffnMid);

                // GELU
                int ffnLen = numPatches * intermediate;
                for (int i = 0; i < ffnLen; i++)
                {
                    float x = ffnMid[i];
                    ffnMid[i] = 0.5f * x * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (x + 0.044715f * x * x * x)));
                }

                VisionOps.MatVecAny(ffnMid, blk.FfnDownW, ffnDownB, numPatches, intermediate, _embd, attnOut);

                // Residual 2
                for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];
            }
        }

        if (_postLnW != null)
        {
            fixed (float* postLnW = _postLnW, postLnB = _postLnB) ApplyLayerNorm(hiddenStates, numPatches, _embd, postLnW, postLnB);
        }

        // 3. Patch Merger (2x2 spatial downsample -> 4 * embd)
        int scale = _mergeFactor; // 2
        int downX = patchesX / scale;
        int downY = patchesY / scale;
        tokenCount = downX * downY;
        int mergedDim = _embd * scale * scale;

        // Real bug: mm.input_norm.weight/bias are sized for _embd (1152), not mergedDim (4608,
        // confirmed via list-tensors) -- this norm applies to each patch's embd-sized vector
        // BEFORE the 2x2 pixel-shuffle merge, not after. Applying it post-merge with dim=mergedDim
        // read far past the end of the real 1152-element weight/bias arrays into adjacent garbage
        // heap memory, exploding every value to ~1e16-1e19 magnitude.
        if (_mmInputNormW != null)
        {
            fixed (float* mmInputNormW = _mmInputNormW, mmInputNormB = _mmInputNormB)
            {
                ApplyLayerNorm(hiddenStates, numPatches, _embd, mmInputNormW, mmInputNormB);
            }
        }

        var merged = new float[tokenCount * mergedDim];
        ApplyPixelMerge(hiddenStates, patchesX, patchesY, scale, merged);

        var visualTokens = new float[tokenCount * _projDim];
        if (_mm1W.IsValid && _mm2W.IsValid)
        {
            fixed (float* mm1B = _mm1B, mm2B = _mm2B)
            {
                var midBuf = new float[tokenCount * _mm1OutDim];
                VisionOps.MatVecAny(merged, _mm1W, mm1B, tokenCount, mergedDim, _mm1OutDim, midBuf);

                // GELU
                for (int i = 0; i < midBuf.Length; i++)
                {
                    float x = midBuf[i];
                    midBuf[i] = 0.5f * x * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (x + 0.044715f * x * x * x)));
                }

                VisionOps.MatVecAny(midBuf, _mm2W, mm2B, tokenCount, _mm1OutDim, _projDim, visualTokens);
            }
        }
        else
        {
            for (int t = 0; t < tokenCount; t++)
            {
                int copyDim = Math.Min(mergedDim, _projDim);
                Array.Copy(merged, t * mergedDim, visualTokens, t * _projDim, copyDim);
            }
        }

        return visualTokens;
    }

    private void ExtractPatches(ReadOnlySpan<float> chw, int width, int height, int patchesX, int patchesY, float[] output)
    {
        int patchSize = _m.PatchSize; // 14
        int patchArea = patchSize * patchSize;
        int planeSize = width * height;
        float[] posEmbd = ResizedPositionEmbeddings(patchesX, patchesY);

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

                        if (posEmbd.Length > 0)
                        {
                            sum += posEmbd[patchIdx * _embd + d];
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

    /// <summary>
    /// 2D RoPE as llama.cpp's <c>clip_graph::build_rope_2d(cur, pos_w, pos_h, theta, interleave_freq: false)</c>
    /// (kimivl.cpp): the first half of each head is rotated with the patch COLUMN, the second half with
    /// the patch ROW. Each half is an ordinary (mode 0, adjacent-pair) RoPE over <c>headDim/2</c> dims with
    /// its own frequency ladder <c>theta^(-2i/(headDim/2))</c>. Found 2026-09-27 against llama-mtmd-debug:
    /// the earlier port paired dim d with d+headDim/2 and interleaved the x/y frequencies.
    /// </summary>
    private void Apply2dInterleavedRope(float[] q, float[] k, int patchesX, int patchesY)
    {
        int half = _headDim / 2;
        int pairs = half / 2;
        var freqs = new float[pairs];
        for (int i = 0; i < pairs; i++) freqs[i] = MathF.Pow(_ropeTheta, -2.0f * i / half);

        Parallel.For(0, patchesY, py =>
        {
            for (int px = 0; px < patchesX; px++)
            {
                int p = py * patchesX + px;
                for (int h = 0; h < _heads; h++)
                {
                    int headOff = (p * _heads + h) * _headDim;
                    for (int part = 0; part < 2; part++)
                    {
                        float pos = part == 0 ? px : py;
                        int baseOff = headOff + part * half;
                        for (int i = 0; i < pairs; i++)
                        {
                            float theta = pos * freqs[i];
                            float c = MathF.Cos(theta), sn = MathF.Sin(theta);
                            int o = baseOff + 2 * i;
                            float q0 = q[o], q1 = q[o + 1];
                            q[o] = q0 * c - q1 * sn; q[o + 1] = q0 * sn + q1 * c;
                            float k0 = k[o], k1 = k[o + 1];
                            k[o] = k0 * c - k1 * sn; k[o + 1] = k0 * sn + k1 * c;
                        }
                    }
                }
            }
        });
    }

    /// <summary>
    /// The learned position table (<c>n_per_side x n_per_side</c>) resized to the patch grid exactly as
    /// llama.cpp's <c>resize_position_embeddings()</c> does it: <c>ggml_interpolate</c> with
    /// <c>GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS</c> (PyTorch bilinear, align_corners=False,
    /// antialias=True: triangle filter, support widened when downscaling). Found 2026-09-27: the earlier
    /// port indexed the raw 64x64 table with the patch index, which is only right for a 64x64 grid.
    /// </summary>
    private float[] ResizedPositionEmbeddings(int width, int height)
    {
        if (_posEmbdF32.Length == 0) return [];
        int side = (int)Math.Round(Math.Sqrt(_posEmbdF32.Length / _embd));
        if (width == side && height == side) return _posEmbdF32;

        var dst = new float[width * height * _embd];
        float sf0 = (float)width / side, sf1 = (float)height / side;
        float support0 = Math.Max(1f, 1f / sf0), invScale0 = 1f / support0;
        float support1 = Math.Max(1f, 1f / sf1), invScale1 = 1f / support1;
        const float off = 0.5f;
        static float Tri(float x) => Math.Max(1f - MathF.Abs(x), 0f);
        var src = _posEmbdF32;
        int embd = _embd;

        Parallel.For(0, height, i1 =>
        {
            float y = (i1 + off) / sf1;
            int yMin = Math.Max((int)(y - support1 + off), 0);
            int yMax = Math.Min((int)(y + support1 + off), side);
            var acc = new float[embd];
            for (int i0 = 0; i0 < width; i0++)
            {
                float x = (i0 + off) / sf0;
                int xMin = Math.Max((int)(x - support0 + off), 0);
                int xMax = Math.Min((int)(x + support0 + off), side);
                Array.Clear(acc);
                float total = 0f;
                for (int sy = yMin; sy < yMax; sy++)
                {
                    float wy = Tri((sy - y + off) * invScale1);
                    for (int sx = xMin; sx < xMax; sx++)
                    {
                        float w = Tri((sx - x + off) * invScale0) * wy;
                        if (w <= 0f) continue;
                        int so = (sy * side + sx) * embd;
                        for (int d = 0; d < embd; d++) acc[d] += src[so + d] * w;
                        total += w;
                    }
                }
                int dOff = (i1 * width + i0) * embd;
                float inv = total > 0f ? 1f / total : 1f;
                for (int d = 0; d < embd; d++) dst[dOff + d] = acc[d] * inv;
            }
        });
        return dst;
    }

}
