
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native C# Dots-OCR and PaddleOCR-VL Vision ViT Encoder + 2D M-RoPE + Patch Merger + GELU MLP Projector.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/dotsocr.cpp
/// </summary>
public sealed unsafe class DotsOcrVisionEncoder
{
    private readonly DotsOcrVisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly int _mergeFactor;
    private readonly float _eps;

    private readonly float[] _patchEmbdWF32;
    private readonly float[]? _patchEmbdB;
    private readonly float[] _posEmbdF32;
    private readonly float[]? _preLnW;
    private readonly float[]? _postLnW;

    private readonly float[]? _inputNormW;
    private readonly float[]? _inputNormB;
    private readonly VisionTensorRef _mlp0W;
    private readonly float[]? _mlp0B;
    private readonly VisionTensorRef _mlp2W;
    private readonly float[]? _mlp2B;

    private readonly LayerWeights[] _blocks;

    private sealed class LayerWeights
    {
        public float[]? Ln1W;
        public float[]? Ln1B;
        // dots.ocr ships a fused attn_qkv (Q rows, then K, then V) and a gated SwiGLU FFN
        // (ffn_gate + ffn_up, clip.use_silu); the separate q/k/v + GELU-MLP fields stay for
        // checkpoints that have those instead.
        public VisionTensorRef AttnQkvW;
        public VisionTensorRef FfnGateW;
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

    public DotsOcrVisionEncoder(DotsOcrVisionModel model)
    {
        _m = model;
        _embd = model.EmbeddingDim;
        _heads = model.HeadCount;
        _headDim = model.HeadDim;
        _layers = model.LayerCount;
        _projDim = model.ProjectionDim;
        _mergeFactor = model.MergeFactor;
        _eps = model.Eps;

        var gguf = model.Gguf;

        _patchEmbdWF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.patch_embd.weight"));
        _patchEmbdB = VisionOps.GetTensorArray(gguf, "v.patch_embd.bias");
        _posEmbdF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.position_embd.weight", "v.position_embd"));
        _preLnW = VisionOps.GetTensorArray(gguf, "v.pre_ln.weight");
        // dots.ocr's post-trunk RMS norm is stored as mm.post_norm (clip.cpp: post_ln_w = TN_MM_POST_NORM).
        _postLnW = VisionOps.GetTensorArray(gguf, "v.post_ln.weight", "mm.post_norm.weight");

        _inputNormW = VisionOps.GetTensorArray(gguf, "mm.input_norm.weight", "mm.0.weight");
        _inputNormB = VisionOps.GetTensorArray(gguf, "mm.input_norm.bias", "mm.0.bias");
        _mlp0W = VisionOps.GetTensor(gguf, "mm.1.weight", "mm.0.weight");
        _mlp0B = VisionOps.GetTensorArray(gguf, "mm.1.bias", "mm.0.bias");
        _mlp2W = VisionOps.GetTensor(gguf, "mm.2.weight", "mm.3.weight");
        _mlp2B = VisionOps.GetTensorArray(gguf, "mm.2.bias", "mm.3.bias");

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
                FfnGateW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_gate.weight"),
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

    public float[] Forward(ReadOnlySpan<float> chw, int targetWidth, int targetHeight, int patchesX, int patchesY, out int tokenCount)
    {
        int numPatches = patchesX * patchesY;
        var hiddenStates = new float[numPatches * _embd];

        fixed (float* chwPtr = chw)
        {
            ExtractPatches(chwPtr, targetWidth, targetHeight, patchesX, patchesY, hiddenStates);
        }

        if (_preLnW != null)
        {
            fixed (float* preLnW = _preLnW) VisionOps.RmsNorm(hiddenStates, numPatches, _embd, preLnW, _eps);
        }

        var qBuf = new float[numPatches * _embd];
        var kBuf = new float[numPatches * _embd];
        var vBuf = new float[numPatches * _embd];
        var attnOut = new float[numPatches * _embd];
        var normed = new float[numPatches * _embd];

        int maxIntermediate = 0;
        for (int l = 0; l < _layers; l++)
        {
            if (_blocks[l].FfnIntermediate > maxIntermediate) maxIntermediate = _blocks[l].FfnIntermediate;
        }
        var ffnMid = new float[numPatches * maxIntermediate];
        var ffnGate = new float[numPatches * maxIntermediate];
        var qkv = new float[numPatches * 3 * _embd];

        for (int l = 0; l < _layers; l++)
        {
            var blk = _blocks[l];

            fixed (float* ln1W = blk.Ln1W, attnQB = blk.AttnQB, attnKB = blk.AttnKB, attnVB = blk.AttnVB,
                   attnOutB = blk.AttnOutB, ln2W = blk.Ln2W, ffnUpB = blk.FfnUpB, ffnDownB = blk.FfnDownB)
            {
                Array.Copy(hiddenStates, normed, hiddenStates.Length);
                VisionOps.RmsNorm(normed, numPatches, _embd, ln1W, _eps);

                if (blk.AttnQkvW.IsValid)
                {
                    VisionOps.MatVecAny(normed, blk.AttnQkvW, null, numPatches, _embd, 3 * _embd, qkv);
                    for (int t = 0; t < numPatches; t++)
                    {
                        Array.Copy(qkv, t * 3 * _embd, qBuf, t * _embd, _embd);
                        Array.Copy(qkv, t * 3 * _embd + _embd, kBuf, t * _embd, _embd);
                        Array.Copy(qkv, t * 3 * _embd + 2 * _embd, vBuf, t * _embd, _embd);
                    }
                }
                else
                {
                    VisionOps.MatVecAny(normed, blk.AttnQW, attnQB, numPatches, _embd, _embd, qBuf);
                    VisionOps.MatVecAny(normed, blk.AttnKW, attnKB, numPatches, _embd, _embd, kBuf);
                    VisionOps.MatVecAny(normed, blk.AttnVW, attnVB, numPatches, _embd, _embd, vBuf);
                }

                // Vision M-RoPE, dotsocr.cpp: ggml_rope_multi over d_head/2 with sections d_head/4 and
                // positions [h, w, h, w] per patch in raster order.
                VisionOps.ApplyMRoPE(qBuf, kBuf, patchesX, patchesY, _heads, _heads, _headDim, independentSections: true);

                VisionOps.Attention(qBuf, kBuf, vBuf, numPatches, _heads, _headDim, normed);
                VisionOps.MatVecAny(normed, blk.AttnOutW, attnOutB, numPatches, _embd, _embd, attnOut);

                for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];

                Array.Copy(hiddenStates, normed, hiddenStates.Length);
                VisionOps.RmsNorm(normed, numPatches, _embd, ln2W, _eps);

                int intermediate = blk.FfnIntermediate;
                VisionOps.MatVecAny(normed, blk.FfnUpW, ffnUpB, numPatches, _embd, intermediate, ffnMid);
                if (blk.FfnGateW.IsValid)
                {
                    // SwiGLU: silu(gate) * up
                    VisionOps.MatVecAny(normed, blk.FfnGateW, null, numPatches, _embd, intermediate, ffnGate);
                    int n = numPatches * intermediate;
                    for (int i = 0; i < n; i++)
                    {
                        float g = ffnGate[i];
                        ffnMid[i] *= g / (1f + MathF.Exp(-g));
                    }
                }
                else
                {
                    VisionOps.Gelu(ffnMid.AsSpan(0, numPatches * intermediate));
                }
                VisionOps.MatVecAny(ffnMid, blk.FfnDownW, ffnDownB, numPatches, intermediate, _embd, attnOut);

                for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];
            }
        }

        if (_postLnW != null)
        {
            fixed (float* postLnW = _postLnW) VisionOps.RmsNorm(hiddenStates, numPatches, _embd, postLnW, _eps);
        }

        // LayerNorm input norm
        if (_inputNormW != null)
        {
            fixed (float* inputNormW = _inputNormW, inputNormB = _inputNormB)
            {
                VisionOps.LayerNorm(hiddenStates, numPatches, _embd, inputNormW, inputNormB, 1e-6f);   // dotsocr.cpp: eps 1e-6
            }
        }

        // Patch merge 2x2
        int downH = patchesY / 2;
        int downW = patchesX / 2;
        tokenCount = downH * downW;
        int mergedDim = _embd * 4;

        var merged = new float[tokenCount * mergedDim];
        VisionOps.PixelShuffle2x2(hiddenStates, patchesY, patchesX, _embd, merged);

        // 2-layer GELU MLP Projector
        var visualTokens = new float[tokenCount * _projDim];
        if (_mlp0W.IsValid && _mlp2W.IsValid)
        {
            fixed (float* mlp0B = _mlp0B, mlp2B = _mlp2B)
            {
                var midBuf = new float[tokenCount * mergedDim];
                VisionOps.MatVecAny(merged, _mlp0W, mlp0B, tokenCount, mergedDim, mergedDim, midBuf);
                // nn.GELU() default: exact erf GELU (dotsocr.cpp FFN_GELU_ERF), not the tanh form.
                for (int i = 0; i < midBuf.Length; i++)
                    midBuf[i] = 0.5f * midBuf[i] * (1f + Erf(midBuf[i] * 0.70710678f));
                VisionOps.MatVecAny(midBuf, _mlp2W, mlp2B, tokenCount, mergedDim, _projDim, visualTokens);
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

    // Abramowitz-Stegun 7.1.26 is only ~1e-7 absolute; use the double-precision form for parity.
    private static float Erf(float x)
    {
        double t = 1.0 / (1.0 + 0.5 * Math.Abs(x));
        double y = 1.0 - t * Math.Exp(-x * (double)x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
            + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
        return (float)(x >= 0 ? y : -y);
    }

    private void ExtractPatches(float* chw, int width, int height, int patchesX, int patchesY, float[] output)
    {
        int patchSize = _m.PatchSize;
        int patchArea = patchSize * patchSize;
        int planeSize = width * height;

        Parallel.For(0, patchesY, py =>
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

                        if (_posEmbdF32.Length > 0 && patchIdx < 729)
                        {
                            sum += _posEmbdF32[patchIdx * _embd + d];
                        }

                        output[outOffset + d] = sum;
                    }
                }
            }
        });
    }
}
