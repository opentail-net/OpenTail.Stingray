
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native vision encoder for StepFun Step-3 VL (SigLIP ViT + 2-stage spatial downsamplers + linear projection).
/// </summary>
public sealed unsafe class Step3VlVisionEncoder
{
    private readonly Step3VlVisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly float _eps;

    private readonly float[]? _patchEmbdW;
    private readonly float[]? _patchEmbdB;
    private readonly float[]? _posEmbd;
    private readonly float[]? _postLnW;
    private readonly float[]? _postLnB;
    private readonly float[]? _mm0W;
    private readonly float[]? _mm0B;
    private readonly float[]? _mm1W;
    private readonly float[]? _mm1B;
    private readonly float[]? _mmModelProjW;
    private readonly float[]? _mmModelProjB;

    private readonly int _mm0OutDim;
    private readonly int _mm1OutDim;

    private readonly LayerWeights[] _blocks;

    private sealed class LayerWeights
    {
        public float[]? Ln1W, Ln1B, Ln2W, Ln2B;
        public float[]? QW, KW, VW, OW;
        public float[]? QB, KB, VB, OB;
        public float[]? FfnUpW, FfnGateW, FfnDownW;
        public float[]? FfnUpB, FfnGateB, FfnDownB;
        public int FfnIntermediate;
    }

    public int ProjectionDim => _projDim;

    public Step3VlVisionEncoder(Step3VlVisionModel model)
    {
        _m       = model;
        _embd    = model.EmbeddingDim;
        _heads   = model.HeadCount;
        _headDim = model.HeadDim;
        _layers  = model.LayerCount;
        _projDim = model.ProjectionDim;
        _eps     = model.Eps;

        var gguf = model.Gguf;

        _patchEmbdW   = VisionOps.LoadTensorF32(gguf, "v.patch_embd.weight");
        _patchEmbdB   = VisionOps.GetTensorArray(gguf, "v.patch_embd.bias");
        _posEmbd      = VisionOps.LoadTensorF32(gguf, "v.position_embd.weight", "v.position_embd");
        _postLnW      = VisionOps.GetTensorArray(gguf, "v.post_ln.weight");
        _postLnB      = VisionOps.GetTensorArray(gguf, "v.post_ln.bias");
        _mm0W         = VisionOps.LoadTensorF32(gguf, "mm.0.weight");
        _mm0B         = VisionOps.GetTensorArray(gguf, "mm.0.bias");
        _mm1W         = VisionOps.LoadTensorF32(gguf, "mm.1.weight");
        _mm1B         = VisionOps.GetTensorArray(gguf, "mm.1.bias");
        _mmModelProjW = VisionOps.LoadTensorF32(gguf, "mm.model_proj.weight");
        _mmModelProjB = VisionOps.GetTensorArray(gguf, "mm.model_proj.bias");

        var mm0T = gguf.FindTensor("mm.0.weight");
        _mm0OutDim = mm0T.HasValue ? (int)mm0T.Value.Dimensions[1] : _embd * 2;
        var mm1T = gguf.FindTensor("mm.1.weight");
        _mm1OutDim = mm1T.HasValue ? (int)mm1T.Value.Dimensions[1] : _embd * 4;

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
                FfnGateW = VisionOps.LoadTensorF32(gguf, $"v.blk.{l}.ffn_gate.weight"),
                FfnGateB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ffn_gate.bias"),
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

        if (_posEmbd != null)
        {
            int maxP = Math.Min(nP * _embd, _posEmbd.Length);
            for (int i = 0; i < maxP; i++) x[i] += _posEmbd[i];
        }

        var normed = new float[nP * _embd];
        var qBuf   = new float[nP * _embd];
        var kBuf   = new float[nP * _embd];
        var vBuf   = new float[nP * _embd];
        var tmp    = new float[nP * _embd];

        int maxIntermediate = 0;
        for (int l = 0; l < _layers; l++)
        {
            if (_blocks[l].FfnIntermediate > maxIntermediate) maxIntermediate = _blocks[l].FfnIntermediate;
        }
        var mid  = new float[nP * maxIntermediate];
        var gate = new float[nP * maxIntermediate];

        for (int l = 0; l < _layers; l++)
        {
            var b = _blocks[l];

            fixed (float* ln1W = b.Ln1W, ln1B = b.Ln1B, qb = b.QB, kb = b.KB, vb = b.VB, ob = b.OB,
                   ln2W = b.Ln2W, ln2B = b.Ln2B, ffnUpB = b.FfnUpB, ffnGateB = b.FfnGateB,
                   ffnDownB = b.FfnDownB)
            {
                Array.Copy(x, normed, x.Length);
                VisionOps.LayerNorm(normed, nP, _embd, ln1W, ln1B, _eps);

                VisionOps.MatVec(normed, b.QW, qb, nP, _embd, _embd, qBuf);
                VisionOps.MatVec(normed, b.KW, kb, nP, _embd, _embd, kBuf);
                VisionOps.MatVec(normed, b.VW, vb, nP, _embd, _embd, vBuf);
                VisionOps.Attention(qBuf, kBuf, vBuf, nP, _heads, _headDim, normed);
                VisionOps.MatVec(normed, b.OW, ob, nP, _embd, _embd, tmp);
                for (int i = 0; i < x.Length; i++) x[i] += tmp[i];

                Array.Copy(x, normed, x.Length);
                VisionOps.LayerNorm(normed, nP, _embd, ln2W, ln2B, _eps);

                int inter = b.FfnIntermediate;
                int interLen = nP * inter;
                VisionOps.MatVec(normed, b.FfnUpW, ffnUpB, nP, _embd, inter, mid);

                if (b.FfnGateW != null)
                {
                    VisionOps.MatVec(normed, b.FfnGateW, ffnGateB, nP, _embd, inter, gate);
                    VisionOps.Silu(mid.AsSpan(0, interLen));
                    for (int i = 0; i < interLen; i++) mid[i] *= gate[i];
                }
                else
                {
                    VisionOps.Gelu(mid.AsSpan(0, interLen));
                }

                VisionOps.MatVec(mid, b.FfnDownW, ffnDownB, nP, inter, _embd, tmp);
                for (int i = 0; i < x.Length; i++) x[i] += tmp[i];
            }
        }

        if (_postLnW != null)
        {
            fixed (float* postLnW = _postLnW, postLnB = _postLnB) VisionOps.LayerNorm(x, nP, _embd, postLnW, postLnB, _eps);
        }

        // Projector: 2-stage spatial downsampling
        int outX = Math.Max(1, patchesX / 2);
        int outY = Math.Max(1, patchesY / 2);
        tokenCount = outX * outY;
        int mergedDim = _embd * 4;

        var merged = new float[tokenCount * mergedDim];
        VisionOps.PixelShuffle2x2(x, patchesY, patchesX, _embd, merged);

        var mm0Out = new float[tokenCount * _mm0OutDim];
        fixed (float* mm0B = _mm0B, mm1B = _mm1B, mmModelProjB = _mmModelProjB)
        {
            VisionOps.MatVec(merged, _mm0W, mm0B, tokenCount, mergedDim, _mm0OutDim, mm0Out);
            VisionOps.Gelu(mm0Out);

            var mm1Out = new float[tokenCount * _mm1OutDim];
            VisionOps.MatVec(mm0Out, _mm1W, mm1B, tokenCount, _mm0OutDim, _mm1OutDim, mm1Out);

            float[] proj;
            if (_mmModelProjW != null)
            {
                proj = new float[tokenCount * _projDim];
                VisionOps.MatVec(mm1Out, _mmModelProjW, mmModelProjB, tokenCount, _mm1OutDim, _projDim, proj);
            }
            else
            {
                proj = mm1Out;
            }

            return proj;
        }
    }

    private void Im2ColAndEmbed(float[] chw, int imgW, int imgH, int ps, int px, int py, float[] dst)
    {
        int nP      = px * py;
        int patchDim = 3 * ps * ps;
        var patches = new float[nP * patchDim];

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
