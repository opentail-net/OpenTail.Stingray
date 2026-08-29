
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native C# Mistral Pixtral 12B Vision ViT Encoder + 2D Continuous RoPE + SwiGLU + GELU MLP Projector.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/pixtral.cpp
/// </summary>
public sealed unsafe class PixtralVisionEncoder
{
    private readonly PixtralVisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;

    private readonly float[] _patchEmbdWF32;
    private readonly float[]? _patchEmbdB;
    private readonly float[]? _preLnW;
    private readonly float[]? _postLnW;

    private readonly VisionTensorRef _mlp0W;
    private readonly float[]? _mlp0B;
    private readonly VisionTensorRef _mlp2W;
    private readonly float[]? _mlp2B;

    private readonly LayerWeights[] _blocks;

    private sealed class LayerWeights
    {
        public float[]? Ln1W;
        public VisionTensorRef AttnQW;
        public VisionTensorRef AttnKW;
        public VisionTensorRef AttnVW;
        public VisionTensorRef AttnOutW;
        public float[]? Ln2W;
        public VisionTensorRef FfnGateW;
        public VisionTensorRef FfnUpW;
        public VisionTensorRef FfnDownW;
        public float[]? FfnDownB;
        public int FfnIntermediate;
    }

    public int EmbeddingDim => _embd;
    public int ProjectionDim => _projDim;

    public PixtralVisionEncoder(PixtralVisionModel model)
    {
        _m = model;
        _embd = model.EmbeddingDim;
        _heads = model.HeadCount;
        _headDim = model.HeadDim;
        _layers = model.LayerCount;
        _projDim = model.ProjectionDim;

        var gguf = model.Gguf;

        _patchEmbdWF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.patch_embd.weight"));
        _patchEmbdB = VisionOps.GetTensorArray(gguf, "v.patch_embd.bias");
        _preLnW = VisionOps.GetTensorArray(gguf, "v.pre_ln.weight");
        _postLnW = VisionOps.GetTensorArray(gguf, "v.post_ln.weight");

        _mlp0W = VisionOps.GetTensor(gguf, "mm.0.weight");
        _mlp0B = VisionOps.GetTensorArray(gguf, "mm.0.bias");
        _mlp2W = VisionOps.GetTensor(gguf, "mm.2.weight");
        _mlp2B = VisionOps.GetTensorArray(gguf, "mm.2.bias");

        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var upTensor = gguf.FindTensor($"v.blk.{l}.ffn_up.weight");
            int intermediate = upTensor.HasValue ? (int)upTensor.Value.Dimensions[1] : (_embd * 4);

            _blocks[l] = new LayerWeights
            {
                Ln1W = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln1.weight"),
                AttnQW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_q.weight"),
                AttnKW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_k.weight"),
                AttnVW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_v.weight"),
                AttnOutW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_out.weight"),
                Ln2W = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln2.weight"),
                FfnGateW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_gate.weight"),
                FfnUpW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_up.weight"),
                FfnDownW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_down.weight"),
                FfnDownB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ffn_down.bias"),
                FfnIntermediate = intermediate
            };
        }
    }

    public float[] Forward(ReadOnlySpan<float> chw, int targetWidth, int targetHeight, int patchesX, int patchesY, out int tokenCount)
    {
        tokenCount = patchesX * patchesY;
        var hiddenStates = new float[tokenCount * _embd];

        fixed (float* chwPtr = chw)
        {
            ExtractPatches(chwPtr, targetWidth, targetHeight, patchesX, patchesY, hiddenStates);
        }

        if (_preLnW != null)
        {
            fixed (float* preLnW = _preLnW) VisionOps.RmsNorm(hiddenStates, tokenCount, _embd, preLnW);
        }

        var qBuf = new float[tokenCount * _embd];
        var kBuf = new float[tokenCount * _embd];
        var vBuf = new float[tokenCount * _embd];
        var attnOut = new float[tokenCount * _embd];
        var normed = new float[tokenCount * _embd];

        int maxFfnDim = 0;
        for (int l = 0; l < _layers; l++)
        {
            if (_blocks[l].FfnIntermediate > maxFfnDim) maxFfnDim = _blocks[l].FfnIntermediate;
        }
        var gateBuf = new float[tokenCount * maxFfnDim];
        var upBuf = new float[tokenCount * maxFfnDim];

        for (int l = 0; l < _layers; l++)
        {
            var blk = _blocks[l];

            fixed (float* ln1W = blk.Ln1W, ln2W = blk.Ln2W, ffnDownB = blk.FfnDownB)
            {
                Array.Copy(hiddenStates, normed, hiddenStates.Length);
                VisionOps.RmsNorm(normed, tokenCount, _embd, ln1W);

                VisionOps.MatVecAny(normed, blk.AttnQW, null, tokenCount, _embd, _embd, qBuf);
                VisionOps.MatVecAny(normed, blk.AttnKW, null, tokenCount, _embd, _embd, kBuf);
                VisionOps.MatVecAny(normed, blk.AttnVW, null, tokenCount, _embd, _embd, vBuf);

                // 2D Continuous RoPE
                VisionOps.Continuous2DRoPE(qBuf, kBuf, patchesX, patchesY, _heads, _headDim);

                VisionOps.Attention(qBuf, kBuf, vBuf, tokenCount, _heads, _headDim, normed);
                VisionOps.MatVecAny(normed, blk.AttnOutW, null, tokenCount, _embd, _embd, attnOut);

                for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];

                Array.Copy(hiddenStates, normed, hiddenStates.Length);
                VisionOps.RmsNorm(normed, tokenCount, _embd, ln2W);

                int ffnDim = blk.FfnIntermediate;
                VisionOps.MatVecAny(normed, blk.FfnGateW, null, tokenCount, _embd, ffnDim, gateBuf);
                VisionOps.MatVecAny(normed, blk.FfnUpW, null, tokenCount, _embd, ffnDim, upBuf);

                // SwiGLU: silu(gate) * up
                int ffnLen = tokenCount * ffnDim;
                for (int i = 0; i < ffnLen; i++)
                {
                    float g = gateBuf[i];
                    float silu = g / (1.0f + MathF.Exp(-g));
                    gateBuf[i] = silu * upBuf[i];
                }

                VisionOps.MatVecAny(gateBuf, blk.FfnDownW, ffnDownB, tokenCount, ffnDim, _embd, attnOut);

                for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];
            }
        }

        if (_postLnW != null)
        {
            fixed (float* postLnW = _postLnW) VisionOps.RmsNorm(hiddenStates, tokenCount, _embd, postLnW);
        }

        var visualTokens = new float[tokenCount * _projDim];
        if (_mlp0W.IsValid && _mlp2W.IsValid)
        {
            fixed (float* mlp0B = _mlp0B, mlp2B = _mlp2B)
            {
                var midBuf = new float[tokenCount * _projDim];
                VisionOps.MatVecAny(hiddenStates, _mlp0W, mlp0B, tokenCount, _embd, _projDim, midBuf);
                VisionOps.Gelu(midBuf);
                VisionOps.MatVecAny(midBuf, _mlp2W, mlp2B, tokenCount, _projDim, _projDim, visualTokens);
            }
        }
        else
        {
            for (int t = 0; t < tokenCount; t++)
            {
                int copyDim = Math.Min(_embd, _projDim);
                Array.Copy(hiddenStates, t * _embd, visualTokens, t * _projDim, copyDim);
            }
        }

        return visualTokens;
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
                        output[outOffset + d] = sum;
                    }
                }
            }
        });
    }
}
