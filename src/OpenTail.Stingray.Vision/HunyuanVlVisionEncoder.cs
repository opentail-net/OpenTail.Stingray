
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native vision encoder for Tencent HunyuanVL (SigLIP ViT + Perceiver spatial downsampler + image wrap tokens).
/// </summary>
public sealed unsafe class HunyuanVlVisionEncoder
{
    private readonly HunyuanVlVisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly int _nMerge;
    private readonly float _eps;

    private readonly float[]? _patchEmbdW;
    private readonly float[]? _patchEmbdB;
    private readonly float[]? _posEmbd;
    private readonly float[]? _preLnW;
    private readonly float[]? _preLnB;
    private readonly float[]? _postLnW;
    private readonly float[]? _postLnB;
    private readonly float[]? _mmPreNormW;
    private readonly float[]? _mm0W;
    private readonly float[]? _mm0B;
    private readonly float[]? _mm2W;
    private readonly float[]? _mm2B;
    private readonly float[]? _mmModelProjW;
    private readonly float[]? _mmModelProjB;
    private readonly float[]? _mmPostNormW;
    private readonly float[]? _imageNewline;
    private readonly float[]? _imageBegin;
    private readonly float[]? _imageEnd;

    // Intermediate FFN dim for projector mm.0 (real strided Conv2D output channels) and mm.2 (1x1 conv, real GGUF name "mm.2.weight")
    private readonly int _mm0OutDim;
    private readonly int _mm2OutDim;

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

    public HunyuanVlVisionEncoder(HunyuanVlVisionModel model)
    {
        _m       = model;
        _embd    = model.EmbeddingDim;
        _heads   = model.HeadCount;
        _headDim = model.HeadDim;
        _layers  = model.LayerCount;
        _projDim = model.ProjectionDim;
        _nMerge  = model.NMerge;
        _eps     = model.Eps;

        var gguf = model.Gguf;

        _patchEmbdW   = VisionOps.LoadTensorF32(gguf, "v.patch_embd.weight");
        _patchEmbdB   = VisionOps.GetTensorArray(gguf, "v.patch_embd.bias");
        _posEmbd      = VisionOps.LoadTensorF32(gguf, "v.position_embd.weight", "v.position_embd");
        _preLnW       = VisionOps.GetTensorArray(gguf, "v.pre_ln.weight");
        _preLnB       = VisionOps.GetTensorArray(gguf, "v.pre_ln.bias");
        _postLnW      = VisionOps.GetTensorArray(gguf, "v.post_ln.weight");
        _postLnB      = VisionOps.GetTensorArray(gguf, "v.post_ln.bias");
        _mmPreNormW   = VisionOps.GetTensorArray(gguf, "mm.pre_norm.weight");
        // mm.0: real strided Conv2D (kernel=stride=n_merge), embd -> mm0OutDim. Raw GGUF layout
        // [kw,kh,cin,cout] (ne0=kw fastest .. ne3=cout slowest) - channel OUTER, spatial INNER
        // per-output-channel, same convention as v.patch_embd.weight / GLM-4.6V's patch merger.
        _mm0W         = VisionOps.LoadTensorF32(gguf, "mm.0.weight");
        _mm0B         = VisionOps.GetTensorArray(gguf, "mm.0.bias");
        // mm.2: real 1x1 Conv2D (mathematically a plain per-position Linear). Real GGUF name is
        // "mm.2.weight", NOT "mm.1.weight" (mm.1.* does not exist in the real checkpoint).
        _mm2W         = VisionOps.LoadTensorF32(gguf, "mm.2.weight");
        _mm2B         = VisionOps.GetTensorArray(gguf, "mm.2.bias");
        // Real GGUF name is "mm.model.fc.weight", NOT "mm.model_proj.weight" (which does not exist).
        _mmModelProjW = VisionOps.LoadTensorF32(gguf, "mm.model.fc.weight");
        _mmModelProjB = VisionOps.GetTensorArray(gguf, "mm.model.fc.bias");
        _mmPostNormW  = VisionOps.GetTensorArray(gguf, "mm.post_norm.weight");
        _imageNewline = VisionOps.GetTensorArray(gguf, "v.image_newline");
        _imageBegin   = VisionOps.GetTensorArray(gguf, "mm.image_begin");
        _imageEnd     = VisionOps.GetTensorArray(gguf, "mm.image_end");

        // Infer projector intermediate dimensions from tensor shapes. mm.0.weight real raw shape
        // is [kw,kh,cin,cout] (ne3=cout is the LAST/slowest dimension), not [in,out].
        var mm0T = gguf.FindTensor("mm.0.weight");
        _mm0OutDim = mm0T.HasValue ? (int)mm0T.Value.Dimensions[^1] : _embd * _nMerge * _nMerge;
        var mm2T = gguf.FindTensor("mm.2.weight");
        _mm2OutDim = mm2T.HasValue ? (int)mm2T.Value.Dimensions[^1] : _projDim;

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

        // 1. Patch embed
        var x = new float[nP * _embd];
        Im2ColAndEmbed(chw, imgWidth, imgHeight, ps, patchesX, patchesY, x);

        // 2. Learned position embeddings: real v.position_embd.weight is a native 128x128 grid
        // that must be bilinearly resized (pixel-center, align_corners=False) to the real
        // (patchesX,patchesY) grid on every forward pass -- NOT a raw truncated add.
        if (_posEmbd != null)
        {
            AddResizedPosEmbd(x, patchesX, patchesY);
        }

        // 3. Pre-LN
        if (_preLnW != null)
        {
            fixed (float* preLnW = _preLnW, preLnB = _preLnB) VisionOps.LayerNorm(x, nP, _embd, preLnW, preLnB, _eps);
        }

        // 4. ViT blocks
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

        // 5. Post-LN
        if (_postLnW != null)
        {
            fixed (float* postLnW = _postLnW, postLnB = _postLnB) VisionOps.LayerNorm(x, nP, _embd, postLnW, postLnB, _eps);
        }

        // 6. Perceiver projector: RMSNorm -> spatial merge -> linear(GELU) -> linear -> linear_proj -> RMSNorm
        if (_mmPreNormW != null)
        {
            fixed (float* mmPreNormW = _mmPreNormW) VisionOps.RmsNorm(x, nP, _embd, mmPreNormW, _eps);
        }

        int m    = _nMerge;
        int outX = Math.Max(1, patchesX / m);
        int outY = Math.Max(1, patchesY / m);
        int mergedTokens = outX * outY;

        // mm.0: real strided Conv2D (kernel=stride=n_merge), NOT pixel-shuffle-then-linear.
        var mm0Out = new float[mergedTokens * _mm0OutDim];
        ApplyStridedConv2DMerge(x, patchesX, patchesY, m, _mm0OutDim, mm0Out);
        VisionOps.Gelu(mm0Out);

        // mm.2: real 1x1 Conv2D == plain per-position Linear (real GGUF name "mm.2.weight").
        var mm2Out = new float[mergedTokens * _mm2OutDim];
        fixed (float* mm2B = _mm2B) VisionOps.MatVec(mm0Out, _mm2W, mm2B, mergedTokens, _mm0OutDim, _mm2OutDim, mm2Out);

        // Insert the learned image_newline embedding after every row of the merged grid before
        // the final projection: real token count is (outX+1)*outY, not outX*outY.
        float[] withNewlines;
        int rowTokens;
        if (_imageNewline != null)
        {
            rowTokens = outX + 1;
            withNewlines = new float[rowTokens * outY * _mm2OutDim];
            for (int row = 0; row < outY; row++)
            {
                Array.Copy(mm2Out, row * outX * _mm2OutDim, withNewlines, row * rowTokens * _mm2OutDim, outX * _mm2OutDim);
                Array.Copy(_imageNewline, 0, withNewlines, (row * rowTokens + outX) * _mm2OutDim, Math.Min(_mm2OutDim, _imageNewline.Length));
            }
        }
        else
        {
            rowTokens = outX;
            withNewlines = mm2Out;
        }
        int gridTokens = rowTokens * outY;

        // Final projection to LLM hidden size (real GGUF name "mm.model.fc.weight").
        float[] proj;
        int projTokens;
        if (_mmModelProjW != null)
        {
            var projected = new float[gridTokens * _projDim];
            fixed (float* mmModelProjB = _mmModelProjB)
                VisionOps.MatVec(withNewlines, _mmModelProjW, mmModelProjB, gridTokens, _mm2OutDim, _projDim, projected);

            // Wrap with mm.image_begin / mm.image_end learned LLM-hidden-size embeddings.
            if (_imageBegin != null && _imageEnd != null)
            {
                projTokens = gridTokens + 2;
                proj = new float[projTokens * _projDim];
                Array.Copy(_imageBegin, 0, proj, 0, Math.Min(_projDim, _imageBegin.Length));
                Array.Copy(projected, 0, proj, _projDim, projected.Length);
                Array.Copy(_imageEnd, 0, proj, (projTokens - 1) * _projDim, Math.Min(_projDim, _imageEnd.Length));
            }
            else
            {
                projTokens = gridTokens;
                proj = projected;
            }
        }
        else
        {
            projTokens = gridTokens;
            proj = withNewlines;
        }

        tokenCount = projTokens;

        // mm.post_norm applies to the WHOLE wrapped sequence (including begin/end markers).
        if (_mmPostNormW != null)
        {
            fixed (float* mmPostNormW = _mmPostNormW) VisionOps.RmsNorm(proj, projTokens, _projDim, mmPostNormW, _eps);
        }

        return proj;
    }

    /// <summary>
    /// Real learned position-embedding bilinear resize: v.position_embd.weight is a native
    /// 128x128 grid that must be resized to the real (patchesX,patchesY) grid on every forward
    /// pass (pixel-center convention, align_corners=False), matching clip.cpp's
    /// PROJECTOR_TYPE_HUNYUANVL branch exactly -- not a raw truncated add.
    /// </summary>
    private void AddResizedPosEmbd(float[] x, int patchesX, int patchesY)
    {
        int nGrid = (int)Math.Round(Math.Sqrt(_posEmbd!.Length / (double)_embd));
        if (nGrid <= 0) return;

        float sx = (patchesX + 0.1f) / nGrid;
        float sy = (patchesY + 0.1f) / nGrid;

        Parallel.For(0, patchesY, y =>
        {
            float fy = (y + 0.5f) / sy - 0.5f;
            int y0 = Math.Clamp((int)MathF.Floor(fy), 0, nGrid - 1);
            int y1 = Math.Clamp(y0 + 1, 0, nGrid - 1);
            float wy1 = Math.Clamp(fy - y0, 0f, 1f);
            float wy0 = 1f - wy1;

            for (int xIdx = 0; xIdx < patchesX; xIdx++)
            {
                float fx = (xIdx + 0.5f) / sx - 0.5f;
                int x0 = Math.Clamp((int)MathF.Floor(fx), 0, nGrid - 1);
                int x1 = Math.Clamp(x0 + 1, 0, nGrid - 1);
                float wx1 = Math.Clamp(fx - x0, 0f, 1f);
                float wx0 = 1f - wx1;

                int dstOff = (y * patchesX + xIdx) * _embd;
                int o00 = (y0 * nGrid + x0) * _embd;
                int o01 = (y0 * nGrid + x1) * _embd;
                int o10 = (y1 * nGrid + x0) * _embd;
                int o11 = (y1 * nGrid + x1) * _embd;

                for (int d = 0; d < _embd; d++)
                {
                    x[dstOff + d] +=
                        wy0 * wx0 * _posEmbd[o00 + d] + wy0 * wx1 * _posEmbd[o01 + d] +
                        wy1 * wx0 * _posEmbd[o10 + d] + wy1 * wx1 * _posEmbd[o11 + d];
                }
            }
        });
    }

    /// <summary>
    /// Real strided Conv2D projector merger (mm.0): for each n_merge x n_merge block of ViT
    /// output patches, out[o] = bias[o] + sum over (c,dy,dx) of weight[o,c,dy,dx] * hidden[srcPatch,c].
    /// Weight raw GGUF layout is [kw,kh,cin,cout] (ne0=kw fastest .. ne3=cout slowest), i.e. for a
    /// fixed output channel the real per-position index order is channel OUTER, spatial (dy,dx)
    /// INNER -- the same convention as v.patch_embd.weight and GLM-4.6V's patch merger.
    /// </summary>
    private void ApplyStridedConv2DMerge(float[] src, int patchesX, int patchesY, int scale, int outDim, float[] dst)
    {
        int downX = patchesX / scale;
        int downY = patchesY / scale;
        int cin = _embd;
        int kArea = scale * scale;

        fixed (float* w = _mm0W)
        {
            var wLocal = w;
            Parallel.For(0, downY, dy0 =>
            {
                for (int dx0 = 0; dx0 < downX; dx0++)
                {
                    int dstTokenIdx = dy0 * downX + dx0;
                    int dstOff = dstTokenIdx * outDim;

                    for (int o = 0; o < outDim; o++)
                    {
                        float sum = _mm0B != null ? _mm0B[o] : 0f;
                        int wOffO = o * (cin * kArea);
                        for (int c = 0; c < cin; c++)
                        {
                            int wOffC = wOffO + c * kArea;
                            for (int dy = 0; dy < scale; dy++)
                            {
                                int srcY = dy0 * scale + dy;
                                for (int dx = 0; dx < scale; dx++)
                                {
                                    int srcX = dx0 * scale + dx;
                                    int srcPatchIdx = srcY * patchesX + srcX;
                                    sum += src[srcPatchIdx * cin + c] * wLocal[wOffC + dy * scale + dx];
                                }
                            }
                        }
                        dst[dstOff + o] = sum;
                    }
                }
            });
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
