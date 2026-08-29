
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native vision encoder for LG AI Research EXAONE 4.5 Vision (ViT + GQA + dual Conv2D patch embed sum + 2D M-RoPE + RMSNorm + 4-patch spatial merge).
/// </summary>
public sealed unsafe class Exaone4VisionEncoder
{
    private readonly Exaone4VisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _kvHeads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly int _waPattern;
    private readonly float _eps;

    private readonly float[]? _patchEmbd0W;
    private readonly float[]? _patchEmbd1W;
    private readonly float[]? _patchEmbdB;
    private readonly float[]? _postLnW;
    private readonly float[]? _postLnB;
    private readonly float[]? _mm0W;
    private readonly float[]? _mm0B;
    private readonly float[]? _mm1W;
    private readonly float[]? _mm1B;

    private readonly int _mm0OutDim;
    private readonly int _mm1OutDim;

    private readonly LayerWeights[] _blocks;

    private sealed class LayerWeights
    {
        public float[]? QkvW;
        public float[]? QkvB;
        public float[]? OW;
        public float[]? OB;
        public float[]? Ln1W, Ln1B, Ln2W, Ln2B;
        public float[]? FfnUpW, FfnGateW, FfnDownW;
        public float[]? FfnUpB, FfnGateB, FfnDownB;
        public int FfnIntermediate;
        public int QkvOutDim;
    }

    public int ProjectionDim => _projDim;

    public Exaone4VisionEncoder(Exaone4VisionModel model)
    {
        _m       = model;
        _embd    = model.EmbeddingDim;
        _heads   = model.HeadCount;
        _kvHeads = model.KvHeadCount;
        _headDim = model.HeadDim;
        _layers  = model.LayerCount;
        _projDim = model.ProjectionDim;
        _waPattern = model.WindowAttnPattern;
        _eps     = model.Eps;

        var gguf = model.Gguf;

        _patchEmbd0W = VisionOps.LoadTensorF32(gguf, "v.patch_embd.0.weight", "v.patch_embd.weight");
        _patchEmbd1W = VisionOps.LoadTensorF32(gguf, "v.patch_embd.1.weight", "v.patch_embd.weight.1");
        _patchEmbdB  = VisionOps.GetTensorArray(gguf, "v.patch_embd.bias");
        _postLnW     = VisionOps.GetTensorArray(gguf, "v.post_ln.weight");
        _postLnB     = VisionOps.GetTensorArray(gguf, "v.post_ln.bias");
        _mm0W        = VisionOps.LoadTensorF32(gguf, "mm.0.weight");
        _mm0B        = VisionOps.GetTensorArray(gguf, "mm.0.bias");
        _mm1W        = VisionOps.LoadTensorF32(gguf, "mm.1.weight", "mm.2.weight");
        _mm1B        = VisionOps.GetTensorArray(gguf, "mm.1.bias", "mm.2.bias");

        var mm0T = gguf.FindTensor("mm.0.weight");
        _mm0OutDim = mm0T.HasValue ? (int)mm0T.Value.Dimensions[1] : _projDim;
        var mm1T = gguf.FindTensor("mm.1.weight") ?? gguf.FindTensor("mm.2.weight");
        _mm1OutDim = mm1T.HasValue ? (int)mm1T.Value.Dimensions[1] : _projDim;

        int qkvOutDim = (_heads + 2 * _kvHeads) * _headDim;

        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var upT = gguf.FindTensor($"v.blk.{l}.ffn_up.weight");
            _blocks[l] = new LayerWeights
            {
                QkvW     = VisionOps.LoadTensorF32(gguf,  $"v.blk.{l}.attn_qkv.weight"),
                QkvB     = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_qkv.bias"),
                OW       = VisionOps.LoadTensorF32(gguf,  $"v.blk.{l}.attn_out.weight"),
                OB       = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_out.bias"),
                Ln1W     = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln1.weight"),
                Ln1B     = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln1.bias"),
                Ln2W     = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln2.weight"),
                Ln2B     = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln2.bias"),
                FfnUpW   = VisionOps.LoadTensorF32(gguf,  $"v.blk.{l}.ffn_up.weight"),
                FfnUpB   = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ffn_up.bias"),
                FfnGateW = VisionOps.LoadTensorF32(gguf,  $"v.blk.{l}.ffn_gate.weight"),
                FfnGateB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ffn_gate.bias"),
                FfnDownW = VisionOps.LoadTensorF32(gguf,  $"v.blk.{l}.ffn_down.weight"),
                FfnDownB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ffn_down.bias"),
                FfnIntermediate = upT.HasValue ? (int)upT.Value.Dimensions[1] : _embd * 4,
                QkvOutDim = qkvOutDim,
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
        DualPatchEmbed(chw, imgWidth, imgHeight, ps, patchesX, patchesY, x);

        var normed   = new float[nP * _embd];
        var qkvBuf   = new float[nP * (_heads + 2 * _kvHeads) * _headDim];
        var qBuf     = new float[nP * _heads * _headDim];
        var kBuf     = new float[nP * _kvHeads * _headDim];
        var vBuf     = new float[nP * _kvHeads * _headDim];
        var attnOut  = new float[nP * _heads * _headDim];
        var tmp      = new float[nP * _embd];

        int maxIntermediate = 0;
        for (int l = 0; l < _layers; l++)
        {
            if (_blocks[l].FfnIntermediate > maxIntermediate) maxIntermediate = _blocks[l].FfnIntermediate;
        }
        var up   = new float[nP * maxIntermediate];
        var gate = new float[nP * maxIntermediate];

        for (int l = 0; l < _layers; l++)
        {
            var b = _blocks[l];

            fixed (float* ln1W = b.Ln1W, qkvB = b.QkvB, ob = b.OB, ln2W = b.Ln2W, ffnUpB = b.FfnUpB,
                   ffnGateB = b.FfnGateB, ffnDownB = b.FfnDownB)
            {
                Array.Copy(x, normed, x.Length);
                VisionOps.RmsNorm(normed, nP, _embd, ln1W, _eps);

                // Project QKV
                VisionOps.MatVec(normed, b.QkvW, qkvB, nP, _embd, b.QkvOutDim, qkvBuf);
                SplitQkv(qkvBuf, nP, _heads, _kvHeads, _headDim, qBuf, kBuf, vBuf);

                // 2D M-RoPE
                VisionOps.ApplyMRoPE(qBuf, kBuf, patchesY, patchesX, _heads, _kvHeads, _headDim, theta: 10000.0f);

                // GQA Attention
                VisionOps.AttentionGqa(qBuf, kBuf, vBuf, nP, _heads, _kvHeads, _headDim, attnOut);

                // Project attention out & residual
                VisionOps.MatVec(attnOut, b.OW, ob, nP, _heads * _headDim, _embd, tmp);
                for (int i = 0; i < x.Length; i++) x[i] += tmp[i];

                // FFN
                Array.Copy(x, normed, x.Length);
                VisionOps.RmsNorm(normed, nP, _embd, ln2W, _eps);

                int inter = b.FfnIntermediate;
                VisionOps.MatVec(normed, b.FfnUpW, ffnUpB, nP, _embd, inter, up);
                VisionOps.MatVec(normed, b.FfnGateW, ffnGateB, nP, _embd, inter, gate);
                int interLen = nP * inter;
                VisionOps.Silu(gate.AsSpan(0, interLen));
                for (int i = 0; i < interLen; i++) up[i] *= gate[i];

                VisionOps.MatVec(up, b.FfnDownW, ffnDownB, nP, inter, _embd, tmp);
                for (int i = 0; i < x.Length; i++) x[i] += tmp[i];
            }
        }

        if (_postLnW != null)
        {
            fixed (float* postLnW = _postLnW) VisionOps.RmsNorm(x, nP, _embd, postLnW, _eps);
        }

        // Projector: 4-patch spatial merge -> mm.0 (GELU) -> mm.1
        int outX = Math.Max(1, patchesX / 2);
        int outY = Math.Max(1, patchesY / 2);
        tokenCount = outX * outY;
        int mergedDim = _embd * 4;

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

    private void DualPatchEmbed(float[] chw, int imgW, int imgH, int ps, int px, int py, float[] dst)
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
            VisionOps.MatVec(patches, _patchEmbd0W, patchEmbdB, nP, patchDim, _embd, dst);
        }

        if (_patchEmbd1W != null)
        {
            var p1 = new float[nP * _embd];
            VisionOps.MatVec(patches, _patchEmbd1W, null, nP, patchDim, _embd, p1);
            for (int i = 0; i < dst.Length; i++) dst[i] += p1[i];
        }
    }

    private static void SplitQkv(
        float[] qkv, int nP, int heads, int kvHeads, int headDim,
        float[] q, float[] k, float[] v)
    {
        int qDim   = heads * headDim;
        int kvDim  = kvHeads * headDim;
        int rowDim = qDim + 2 * kvDim;

        Parallel.For(0, nP, t =>
        {
            int srcOff = t * rowDim;
            Array.Copy(qkv, srcOff, q, t * qDim, qDim);
            Array.Copy(qkv, srcOff + qDim, k, t * kvDim, kvDim);
            Array.Copy(qkv, srcOff + qDim + kvDim, v, t * kvDim, kvDim);
        });
    }

}
