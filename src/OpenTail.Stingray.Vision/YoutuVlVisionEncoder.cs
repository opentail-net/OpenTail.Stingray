
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native vision encoder for Tencent YouTu Lab YoutuVL (ViT + Conv3D-as-linear + 2D M-RoPE + VLPatchMerger).
/// </summary>
public sealed unsafe class YoutuVlVisionEncoder
{
    private readonly YoutuVlVisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly int _mergeFactor;
    private readonly float _eps;

    private readonly float[]? _patchEmbdW;
    private readonly float[]? _patchEmbdB;
    private readonly float[]? _preLnW;
    private readonly float[]? _preLnB;
    private readonly float[]? _postLnW;
    private readonly float[]? _postLnB;
    private readonly float[]? _mmInputNormW;
    private readonly float[]? _mm0W;
    private readonly float[]? _mm0B;
    private readonly float[]? _mm1W;
    private readonly float[]? _mm1B;

    private readonly int _mm0OutDim;
    private readonly int _mm1OutDim;

    private readonly LayerWeights[] _blocks;

    private sealed class LayerWeights
    {
        public float[]? Ln1W, Ln1B, Ln2W, Ln2B;
        public float[]? QW, KW, VW, OW;
        public float[]? QB, KB, VB, OB;
        public float[]? FfnUpW, FfnDownW;
        public float[]? FfnUpB, FfnDownB;
        public int FfnIntermediate;
    }

    public int ProjectionDim => _projDim;

    public YoutuVlVisionEncoder(YoutuVlVisionModel model)
    {
        _m          = model;
        _embd       = model.EmbeddingDim;
        _heads      = model.HeadCount;
        _headDim    = model.HeadDim;
        _layers     = model.LayerCount;
        _projDim    = model.ProjectionDim;
        _mergeFactor = model.SpatialMergeFactor;
        _eps        = model.Eps;

        var gguf = model.Gguf;

        _patchEmbdW   = VisionOps.LoadTensorF32(gguf, "v.patch_embd.weight");
        _patchEmbdB   = VisionOps.GetTensorArray(gguf, "v.patch_embd.bias");
        _preLnW       = VisionOps.GetTensorArray(gguf, "v.pre_ln.weight");
        _preLnB       = VisionOps.GetTensorArray(gguf, "v.pre_ln.bias");
        _postLnW      = VisionOps.GetTensorArray(gguf, "v.post_ln.weight");
        _postLnB      = VisionOps.GetTensorArray(gguf, "v.post_ln.bias");
        _mmInputNormW = VisionOps.GetTensorArray(gguf, "mm.input_norm.weight");
        _mm0W         = VisionOps.LoadTensorF32(gguf, "mm.0.weight");
        _mm0B         = VisionOps.GetTensorArray(gguf, "mm.0.bias");
        // Real tensor name in this checkpoint is "mm.2.weight" (confirmed via list-tensors), not
        // "mm.1.weight" -- that name never matched, silently leaving mm1W null and the final
        // MatVec a no-op (missing-weight contract), which is exactly why the whole embedding
        // came back all-zero: the second projector layer never ran at all.
        _mm1W         = VisionOps.LoadTensorF32(gguf, "mm.2.weight", "mm.1.weight");
        _mm1B         = VisionOps.GetTensorArray(gguf, "mm.2.bias", "mm.1.bias");

        var mm0T = gguf.FindTensor("mm.0.weight");
        _mm0OutDim = mm0T.HasValue ? (int)mm0T.Value.Dimensions[1] : _projDim;
        var mm1T = gguf.FindTensor("mm.2.weight") ?? gguf.FindTensor("mm.1.weight");
        _mm1OutDim = mm1T.HasValue ? (int)mm1T.Value.Dimensions[1] : _projDim;

        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var upT = gguf.FindTensor($"v.blk.{l}.ffn_up.weight");
            _blocks[l] = new LayerWeights
            {
                Ln1W     = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln1.weight"),
                Ln1B     = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln1.bias"),
                QW       = VisionOps.LoadTensorF32(gguf, $"v.blk.{l}.attn_q.weight"),
                QB       = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_q.bias"),
                KW       = VisionOps.LoadTensorF32(gguf, $"v.blk.{l}.attn_k.weight"),
                KB       = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_k.bias"),
                VW       = VisionOps.LoadTensorF32(gguf, $"v.blk.{l}.attn_v.weight"),
                VB       = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_v.bias"),
                OW       = VisionOps.LoadTensorF32(gguf, $"v.blk.{l}.attn_out.weight"),
                OB       = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_out.bias"),
                Ln2W     = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln2.weight"),
                Ln2B     = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln2.bias"),
                FfnUpW   = VisionOps.LoadTensorF32(gguf, $"v.blk.{l}.ffn_up.weight"),
                FfnUpB   = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ffn_up.bias"),
                FfnDownW = VisionOps.LoadTensorF32(gguf, $"v.blk.{l}.ffn_down.weight"),
                FfnDownB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ffn_down.bias"),
                FfnIntermediate = upT.HasValue ? (int)upT.Value.Dimensions[1] : _embd * 4,
            };
        }
    }

    public float[] Forward(
        float[] chw, int imgWidth, int imgHeight,
        int patchesX, int patchesY,
        out int tokenCount)
    {
        int ps = _m.PatchSize;
        int nP = patchesX * patchesY;

        var x = new float[nP * _embd];
        Im2ColAndEmbed(chw, imgWidth, imgHeight, ps, patchesX, patchesY, x);

        if (_preLnW != null)
        {
            fixed (float* preLnW = _preLnW, preLnB = _preLnB) VisionOps.LayerNorm(x, nP, _embd, preLnW, preLnB, _eps);
        }

        var normed = new float[nP * _embd];
        var qBuf   = new float[nP * _embd];
        var kBuf   = new float[nP * _embd];
        var vBuf   = new float[nP * _embd];
        var attnOut = new float[nP * _embd];
        var tmp    = new float[nP * _embd];

        int maxIntermediate = 0;
        for (int l = 0; l < _layers; l++)
        {
            if (_blocks[l].FfnIntermediate > maxIntermediate) maxIntermediate = _blocks[l].FfnIntermediate;
        }
        var mid = new float[nP * maxIntermediate];

        for (int l = 0; l < _layers; l++)
        {
            var b = _blocks[l];

            fixed (float* ln1W = b.Ln1W, ln1B = b.Ln1B, qb = b.QB, kb = b.KB, vb = b.VB, ob = b.OB,
                   ln2W = b.Ln2W, ln2B = b.Ln2B, ffnUpB = b.FfnUpB, ffnDownB = b.FfnDownB)
            {
                Array.Copy(x, normed, x.Length);
                VisionOps.LayerNorm(normed, nP, _embd, ln1W, ln1B, _eps);

                VisionOps.MatVec(normed, b.QW, qb, nP, _embd, _embd, qBuf);
                VisionOps.MatVec(normed, b.KW, kb, nP, _embd, _embd, kBuf);
                VisionOps.MatVec(normed, b.VW, vb, nP, _embd, _embd, vBuf);

                // 2D M-RoPE
                VisionOps.ApplyMRoPE(qBuf, kBuf, patchesY, patchesX, _heads, _heads, _headDim, theta: 10000.0f);

                VisionOps.Attention(qBuf, kBuf, vBuf, nP, _heads, _headDim, attnOut);
                VisionOps.MatVec(attnOut, b.OW, ob, nP, _embd, _embd, tmp);
                for (int i = 0; i < x.Length; i++) x[i] += tmp[i];

                Array.Copy(x, normed, x.Length);
                VisionOps.LayerNorm(normed, nP, _embd, ln2W, ln2B, _eps);

                int inter = b.FfnIntermediate;
                VisionOps.MatVec(normed, b.FfnUpW, ffnUpB, nP, _embd, inter, mid);
                VisionOps.Gelu(mid.AsSpan(0, nP * inter));

                VisionOps.MatVec(mid, b.FfnDownW, ffnDownB, nP, inter, _embd, tmp);
                for (int i = 0; i < x.Length; i++) x[i] += tmp[i];
            }
        }

        if (_postLnW != null)
        {
            fixed (float* postLnW = _postLnW, postLnB = _postLnB) VisionOps.LayerNorm(x, nP, _embd, postLnW, postLnB, _eps);
        }

        // VLPatchMerger: RMSNorm -> spatial merge -> mm.0 (GELU) -> mm.1
        if (_mmInputNormW != null)
        {
            fixed (float* mmInputNormW = _mmInputNormW) VisionOps.RmsNorm(x, nP, _embd, mmInputNormW, _eps);
        }

        int m = _mergeFactor;
        int outX = Math.Max(1, patchesX / m);
        int outY = Math.Max(1, patchesY / m);
        tokenCount = outX * outY;
        int mergedDim = _embd * m * m;

        var merged = new float[tokenCount * mergedDim];
        VisionOps.PixelShuffle2x2(x, patchesY, patchesX, _embd, merged);

        var mm0Out = new float[tokenCount * _mm0OutDim];
        fixed (float* mm0B = _mm0B, mm1B = _mm1B)
        {
            VisionOps.MatVec(merged, _mm0W, mm0B, tokenCount, mergedDim, _mm0OutDim, mm0Out);
            VisionOps.Gelu(mm0Out);

            var mm1Out = new float[tokenCount * _mm1OutDim];
            VisionOps.MatVec(mm0Out, _mm1W, mm1B, tokenCount, _mm0OutDim, _mm1OutDim, mm1Out);

            return mm1Out;
        }
    }

    private void Im2ColAndEmbed(float[] chw, int imgW, int imgH, int ps, int px, int py, float[] dst)
    {
        int nP       = px * py;
        int patchDim = 3 * ps * ps;
        var patches  = new float[nP * patchDim];

        Parallel.For(0, py, iy =>
        {
            for (int ix = 0; ix < px; ix++)
            {
                int pIdx   = iy * px + ix;
                int dstOff = pIdx * patchDim;
                for (int c = 0; c < 3; c++)
                {
                    int cOff = c * imgW * imgH;
                    for (int sy = 0; sy < ps; sy++)
                    for (int sx = 0; sx < ps; sx++)
                    {
                        int srcY = iy * ps + sy;
                        int srcX = ix * ps + sx;
                        if (srcY < imgH && srcX < imgW)
                            patches[dstOff + c * ps * ps + sy * ps + sx] = chw[cOff + srcY * imgW + srcX];
                    }
                }
            }
        });

        fixed (float* patchEmbdB = _patchEmbdB)
        {
            VisionOps.MatVec(patches, _patchEmbdW, patchEmbdB, nP, patchDim, _embd, dst);
        }
    }

}
