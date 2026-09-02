
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native C# LLaVA-1.5 / LLaVA-NeXT / LLaVA-OneVision Vision ViT Encoder + 2-layer GELU MLP Projector.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/llava.cpp
/// </summary>
public sealed unsafe class LlavaVisionEncoder
{
    private readonly LlavaVisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly float _eps;

    private readonly float[] _patchEmbdWF32;
    private readonly float[]? _patchEmbdB;
    private readonly float[]? _clsEmbd;
    private readonly float[] _posEmbdF32;
    private readonly float[]? _preLnW;
    private readonly float[]? _preLnB;
    private readonly float[]? _postLnW;
    private readonly float[]? _postLnB;

    private readonly VisionTensorRef _mlp0W;
    private readonly float[]? _mlp0B;
    private readonly VisionTensorRef _mlp2W;
    private readonly float[]? _mlp2B;

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
        // Named by FUNCTION (first = embd->intermediate, second = intermediate->embd), not by
        // the GGUF tensor name -- some real checkpoints (e.g. llava-v1.5-7b's own mmproj) name
        // "ffn_up"/"ffn_down" backwards relative to their actual direction. See the constructor
        // for how these are assigned per-checkpoint from the tensors' own real shapes.
        public VisionTensorRef FfnFirstW;
        public float[]? FfnFirstB;
        public VisionTensorRef FfnSecondW;
        public float[]? FfnSecondB;
        public int FfnIntermediate;
    }

    public int EmbeddingDim => _embd;
    public int ProjectionDim => _projDim;

    public LlavaVisionEncoder(LlavaVisionModel model)
    {
        _m = model;
        _embd = model.EmbeddingDim;
        _heads = model.HeadCount;
        _headDim = model.HeadDim;
        _layers = model.LayerCount;
        _projDim = model.ProjectionDim;
        _eps = model.Eps;

        var gguf = model.Gguf;

        _patchEmbdWF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.patch_embd.weight"));
        _patchEmbdB = VisionOps.GetTensorArray(gguf, "v.patch_embd.bias");
        _clsEmbd = VisionOps.GetTensorArray(gguf, "v.class_embd", "v.cls_embd");
        _posEmbdF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.position_embd.weight", "v.position_embd"));
        _preLnW = VisionOps.GetTensorArray(gguf, "v.pre_ln.weight");
        _preLnB = VisionOps.GetTensorArray(gguf, "v.pre_ln.bias");
        _postLnW = VisionOps.GetTensorArray(gguf, "v.post_ln.weight");
        _postLnB = VisionOps.GetTensorArray(gguf, "v.post_ln.bias");

        _mlp0W = VisionOps.GetTensor(gguf, "mm.0.weight");
        _mlp0B = VisionOps.GetTensorArray(gguf, "mm.0.bias");
        _mlp2W = VisionOps.GetTensor(gguf, "mm.2.weight");
        _mlp2B = VisionOps.GetTensorArray(gguf, "mm.2.bias");

        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var upTensor = gguf.FindTensor($"v.blk.{l}.ffn_up.weight");
            var downTensor = gguf.FindTensor($"v.blk.{l}.ffn_down.weight");

            var upW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_up.weight");
            var upB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ffn_up.bias");
            var downW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_down.weight");
            var downB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ffn_down.bias");

            // Real GGUF ne is [in,out] (ne0 = fastest = the tensor's own real input width,
            // matching VisionOps.MatVecAny's row-major [outDim,inDim] contract). Determine which
            // of ffn_up/ffn_down is genuinely the FIRST (embd->intermediate) linear by checking
            // which one's own input width (Dimensions[0]) equals embd, rather than assuming the
            // name "ffn_up" means "first" -- confirmed against a real checkpoint
            // (llava-v1.5-7b's own mmproj) where the names are backwards: "ffn_up" is actually
            // intermediate->embd (Dimensions=[4096,1024], in=4096) and "ffn_down" is actually
            // embd->intermediate (Dimensions=[1024,4096], in=1024). Found via a real golden
            // numeric mismatch against scripts/llava_ref.py (2026-09-01) -- the previous
            // hardcoded "ffn_up is always first" assumption silently read the wrong quarter of
            // each FFN weight tensor for this checkpoint, undetected by the differentiation-only
            // real-weight test (which doesn't check numeric correctness, only "not degenerate").
            bool upIsFirst = upTensor.HasValue && (int)upTensor.Value.Dimensions[0] == _embd;
            VisionTensorRef firstW, secondW;
            float[]? firstB, secondB;
            int intermediate;
            if (upIsFirst)
            {
                firstW = upW; firstB = upB; secondW = downW; secondB = downB;
                intermediate = (int)upTensor!.Value.Dimensions[1];
            }
            else if (downTensor.HasValue && (int)downTensor.Value.Dimensions[0] == _embd)
            {
                firstW = downW; firstB = downB; secondW = upW; secondB = upB;
                intermediate = (int)downTensor.Value.Dimensions[1];
            }
            else
            {
                // Neither tensor's input width matches embd (missing tensors, or a genuinely
                // unexpected shape) -- fall back to the old naming assumption rather than throw.
                firstW = upW; firstB = upB; secondW = downW; secondB = downB;
                intermediate = upTensor.HasValue ? (int)upTensor.Value.Dimensions[1] : (_embd * 4);
            }

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
                FfnFirstW = firstW,
                FfnFirstB = firstB,
                FfnSecondW = secondW,
                FfnSecondB = secondB,
                FfnIntermediate = intermediate
            };
        }
    }

    public float[] Forward(ReadOnlySpan<float> chw, int targetWidth, int targetHeight, int patchesX, int patchesY, out int tokenCount)
    {
        int numPatches = patchesX * patchesY;
        // Real reference (clip.cpp): `n_pos = num_patches + (model.class_embedding ? 1 : 0)` --
        // the CLS slot is conditional on whether a real v.class_embd tensor exists at all. SigLIP-
        // style checkpoints (e.g. granite-vision-3.2-2b's mmproj, routed here via projector_type
        // "mlp") have NO class_embedding tensor and a position_embd table sized for EXACTLY
        // numPatches positions (confirmed via list-tensors: 729 = 27x27, no +1 row) -- unconditionally
        // reserving a CLS slot here indexed every patch's position embedding one row too far,
        // running off the end of _posEmbdF32 for any checkpoint without a real CLS token.
        bool hasCls = _clsEmbd != null;
        int totalTokensIn = numPatches + (hasCls ? 1 : 0);

        var hiddenStates = new float[totalTokensIn * _embd];
        fixed (float* chwPtr = chw)
        {
            ExtractPatchesWithCls(chwPtr, targetWidth, targetHeight, patchesX, patchesY, hasCls, hiddenStates);
        }

        if (_preLnW != null)
        {
            fixed (float* preLnW = _preLnW, preLnB = _preLnB)
            {
                VisionOps.LayerNorm(hiddenStates, totalTokensIn, _embd, preLnW, preLnB, _eps);
            }
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

        for (int l = 0; l < _layers; l++)
        {
            var blk = _blocks[l];

            fixed (float* ln1W = blk.Ln1W, ln1B = blk.Ln1B, attnQkvB = blk.AttnQkvB, attnQB = blk.AttnQB,
                   attnKB = blk.AttnKB, attnVB = blk.AttnVB, attnOutB = blk.AttnOutB, ln2W = blk.Ln2W,
                   ln2B = blk.Ln2B, ffnFirstB = blk.FfnFirstB, ffnSecondB = blk.FfnSecondB)
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
                VisionOps.MatVecAny(normed, blk.FfnFirstW, ffnFirstB, totalTokensIn, _embd, intermediate, ffnMid);
                VisionOps.QuickGelu(ffnMid.AsSpan(0, totalTokensIn * intermediate));
                VisionOps.MatVecAny(ffnMid, blk.FfnSecondW, ffnSecondB, totalTokensIn, intermediate, _embd, attnOut);

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

        // Strip CLS token (only if this checkpoint actually has one -- see hasCls above).
        tokenCount = numPatches;
        var patchEmbeddings = new float[numPatches * _embd];
        Array.Copy(hiddenStates, hasCls ? _embd : 0, patchEmbeddings, 0, numPatches * _embd);

        // 2-layer GELU MLP Projector
        var visualTokens = new float[tokenCount * _projDim];
        if (_mlp0W.IsValid && _mlp2W.IsValid)
        {
            fixed (float* mlp0B = _mlp0B, mlp2B = _mlp2B)
            {
                var midBuf = new float[tokenCount * _projDim];
                VisionOps.MatVecAny(patchEmbeddings, _mlp0W, mlp0B, tokenCount, _embd, _projDim, midBuf);
                VisionOps.Gelu(midBuf);
                VisionOps.MatVecAny(midBuf, _mlp2W, mlp2B, tokenCount, _projDim, _projDim, visualTokens);
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

    private void ExtractPatchesWithCls(float* chw, int width, int height, int patchesX, int patchesY, bool hasCls, float[] output)
    {
        int patchSize = _m.PatchSize;
        int patchArea = patchSize * patchSize;
        int planeSize = width * height;
        int patchOffset = hasCls ? 1 : 0; // real clip.cpp: `patch_offset = model.class_embedding ? 1 : 0`

        if (hasCls)
        {
            for (int d = 0; d < _embd; d++) output[d] = _clsEmbd![d];
            if (_posEmbdF32.Length > 0)
            {
                for (int d = 0; d < _embd; d++) output[d] += _posEmbdF32[d];
            }
        }

        Parallel.For(0, patchesY, py =>
        {
            for (int px = 0; px < patchesX; px++)
            {
                int patchIdx = py * patchesX + px;
                int tokenIdx = patchIdx + patchOffset;
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
