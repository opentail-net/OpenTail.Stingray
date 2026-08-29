
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native C# DeepSeek-OCR and DeepSeek-OCR2 Vision ViT Encoder + Dual Feature Fusion + FC Projector.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/deepseekocr.cpp
/// </summary>
public sealed unsafe class DeepSeekOcrVisionEncoder
{
    private readonly DeepSeekOcrVisionModel _m;
    private readonly int _embd;
    private readonly int _samEmbd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly bool _isV2;
    private readonly float _eps;

    // See InternVlVisionEncoder.cs / docs/done/vl-untested-code-findings-2026-08-20.md for why
    // patch-embed is dequantized once to F32 instead of requested as a fixed CLR type: it's read
    // per-pixel in ExtractPatchesWithCls's inline conv loop, not through a batched MatVec.
    private readonly float[] _patchEmbdWF32;
    private readonly float[]? _patchEmbdB;
    private readonly float[] _clsEmbdF32;
    private readonly float[] _posEmbdF32;
    private readonly float[]? _postLnW;
    private readonly float[]? _postLnB;

    private readonly VisionTensorRef _fcW;
    private readonly float[]? _fcB;
    private readonly float[]? _viewSep;
    private readonly float[]? _imgNl;

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

    public DeepSeekOcrVisionEncoder(DeepSeekOcrVisionModel model)
    {
        _m = model;
        _embd = model.EmbeddingDim;
        _samEmbd = model.SamEmbeddingDim;
        _heads = model.HeadCount;
        _headDim = model.HeadDim;
        _layers = model.LayerCount;
        _projDim = model.ProjectionDim;
        _isV2 = model.IsV2;
        _eps = model.Eps;

        var gguf = model.Gguf;

        _patchEmbdWF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.patch_embd.weight"));
        _patchEmbdB = VisionOps.GetTensorArray(gguf, "v.patch_embd.bias");
        _clsEmbdF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.class_embd", "v.cls_embd"));
        _posEmbdF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.position_embd.weight", "v.position_embd"));
        _postLnW = VisionOps.GetTensorArray(gguf, "v.post_ln.weight");
        _postLnB = VisionOps.GetTensorArray(gguf, "v.post_ln.bias");

        _fcW = VisionOps.GetTensor(gguf, "mm.model.fc.weight", "mm.fc.weight", "mm.0.weight");
        _fcB = VisionOps.GetTensorArray(gguf, "mm.model.fc.bias", "mm.fc.bias", "mm.0.bias");
        _viewSep = VisionOps.GetTensorArray(gguf, "model.view_seperator");
        _imgNl = VisionOps.GetTensorArray(gguf, "model.image_newline");

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
                FfnGateW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_gate.weight"),
                FfnGateB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ffn_gate.bias"),
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
        int totalTokensIn = numPatches + 1;

        var hiddenStates = new float[totalTokensIn * _embd];
        fixed (float* chwPtr = chw)
        {
            ExtractPatchesWithCls(chwPtr, targetWidth, targetHeight, patchesX, patchesY, hiddenStates);
        }

        var qBuf = new float[totalTokensIn * _embd];
        var kBuf = new float[totalTokensIn * _embd];
        var vBuf = new float[totalTokensIn * _embd];
        var attnOut = new float[totalTokensIn * _embd];
        var normed = new float[totalTokensIn * _embd];
        var qkv = new float[totalTokensIn * 3 * _embd];

        int maxIntermediate = 0;
        for (int l = 0; l < _layers; l++)
        {
            if (_blocks[l].FfnIntermediate > maxIntermediate) maxIntermediate = _blocks[l].FfnIntermediate;
        }
        var ffnMid = new float[totalTokensIn * maxIntermediate];
        var gateBuf = new float[totalTokensIn * maxIntermediate];

        for (int l = 0; l < _layers; l++)
        {
            var blk = _blocks[l];

            fixed (float* ln1W = blk.Ln1W, ln1B = blk.Ln1B, attnQkvB = blk.AttnQkvB, attnQB = blk.AttnQB,
                   attnKB = blk.AttnKB, attnVB = blk.AttnVB, attnOutB = blk.AttnOutB, ln2W = blk.Ln2W,
                   ln2B = blk.Ln2B, ffnGateB = blk.FfnGateB, ffnUpB = blk.FfnUpB, ffnDownB = blk.FfnDownB)
            {
                Array.Copy(hiddenStates, normed, hiddenStates.Length);
                VisionOps.LayerNorm(normed, totalTokensIn, _embd, ln1W, ln1B, _eps);

                if (blk.AttnQkvW.IsValid)
                {
                    VisionOps.MatVecAny(normed, blk.AttnQkvW, attnQkvB, totalTokensIn, _embd, 3 * _embd, qkv);
                    for (int p = 0; p < totalTokensIn; p++)
                    {
                        Array.Copy(qkv, p * 3 * _embd, qBuf, p * _embd, _embd);
                        Array.Copy(qkv, p * 3 * _embd + _embd, kBuf, p * _embd, _embd);
                        Array.Copy(qkv, p * 3 * _embd + 2 * _embd, vBuf, p * _embd, _embd);
                    }
                }
                else
                {
                    VisionOps.MatVecAny(normed, blk.AttnQW, attnQB, totalTokensIn, _embd, _embd, qBuf);
                    VisionOps.MatVecAny(normed, blk.AttnKW, attnKB, totalTokensIn, _embd, _embd, kBuf);
                    VisionOps.MatVecAny(normed, blk.AttnVW, attnVB, totalTokensIn, _embd, _embd, vBuf);
                }

                VisionOps.Attention(qBuf, kBuf, vBuf, totalTokensIn, _heads, _headDim, normed);
                VisionOps.MatVecAny(normed, blk.AttnOutW, attnOutB, totalTokensIn, _embd, _embd, attnOut);

                for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];

                Array.Copy(hiddenStates, normed, hiddenStates.Length);
                VisionOps.LayerNorm(normed, totalTokensIn, _embd, ln2W, ln2B, _eps);

                int intermediate = blk.FfnIntermediate;
                int ffnLen = totalTokensIn * intermediate;

                if (blk.FfnGateW.IsValid)
                {
                    VisionOps.MatVecAny(normed, blk.FfnGateW, ffnGateB, totalTokensIn, _embd, intermediate, gateBuf);
                    VisionOps.MatVecAny(normed, blk.FfnUpW, ffnUpB, totalTokensIn, _embd, intermediate, ffnMid);
                    for (int i = 0; i < ffnLen; i++)
                    {
                        float g = gateBuf[i];
                        float silu = g / (1.0f + MathF.Exp(-g));
                        ffnMid[i] = silu * ffnMid[i];
                    }
                }
                else
                {
                    VisionOps.MatVecAny(normed, blk.FfnUpW, ffnUpB, totalTokensIn, _embd, intermediate, ffnMid);
                    VisionOps.QuickGelu(ffnMid.AsSpan(0, ffnLen));
                }

                VisionOps.MatVecAny(ffnMid, blk.FfnDownW, ffnDownB, totalTokensIn, intermediate, _embd, attnOut);

                for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];
            }
        }

        if (_postLnW != null)
        {
            fixed (float* postLnW = _postLnW, postLnB = _postLnB)
            {
                VisionOps.LayerNorm(hiddenStates, totalTokensIn, _embd, postLnW, postLnB, _eps);
            }
        }

        // Strip CLS token
        tokenCount = numPatches;
        var patchEmbeddings = new float[numPatches * _embd];
        Array.Copy(hiddenStates, _embd, patchEmbeddings, 0, numPatches * _embd);

        // Projector
        var visualTokens = new float[tokenCount * _projDim];
        if (_fcW.IsValid)
        {
            fixed (float* fcB = _fcB)
            {
                VisionOps.MatVecAny(patchEmbeddings, _fcW, fcB, tokenCount, _embd, _projDim, visualTokens);
            }
        }
        else
        {
            for (int t = 0; t < tokenCount; t++)
            {
                int copyDim = Math.Min(_embd, _projDim);
                Array.Copy(patchEmbeddings, t * _embd, visualTokens, t * _projDim, copyDim);
            }
        }

        return visualTokens;
    }

    private void ExtractPatchesWithCls(float* chw, int width, int height, int patchesX, int patchesY, float[] output)
    {
        int patchSize = _m.PatchSize;
        int patchArea = patchSize * patchSize;
        int planeSize = width * height;

        if (_clsEmbdF32.Length > 0)
        {
            for (int d = 0; d < _embd; d++) output[d] = _clsEmbdF32[d];
        }
        if (_posEmbdF32.Length > 0)
        {
            for (int d = 0; d < _embd; d++) output[d] += _posEmbdF32[d];
        }

        Parallel.For(0, patchesY, py =>
        {
            for (int px = 0; px < patchesX; px++)
            {
                int patchIdx = py * patchesX + px;
                int tokenIdx = patchIdx + 1;
                int outOffset = tokenIdx * _embd;

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
                            sum += _posEmbdF32[tokenIdx * _embd + d];
                        }

                        output[outOffset + d] = sum;
                    }
                }
            }
        });
    }
}
