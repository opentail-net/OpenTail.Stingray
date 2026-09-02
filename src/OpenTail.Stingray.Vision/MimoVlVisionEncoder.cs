
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native vision encoder for ByteDance MiMo-VL (MiMo-V2.5 ViT).
/// Implements dual Conv2D patch embed, 2D M-RoPE, GQA with per-head attention sinks,
/// RMSNorm, SwiGLU with biases, and 4-patch spatial merge GELU MLP projector.
/// </summary>
public sealed unsafe class MimoVlVisionEncoder
{
    private readonly MimoVlVisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _kvHeads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly int _nMerge;
    private readonly float _eps;

    private readonly float[]? _patchEmbd0W;
    private readonly float[]? _patchEmbd1W;
    private readonly float[]? _postLnW;
    private readonly float[]? _postLnB;
    private readonly float[]? _mm0W;
    private readonly float[]? _mm1W;

    private readonly int _mm0OutDim;
    private readonly int _mm1OutDim;

    private readonly LayerWeights[] _blocks;

    private sealed class LayerWeights
    {
        public float[]? QkvW;
        public float[]? QkvB;
        // Real local checkpoints under this class's name have SEPARATE attn_q/k/v tensors, not a
        // fused attn_qkv -- QkvW/QkvB above were always null for them (no such tensor exists),
        // silently no-op'ing the QKV projection (MatVec's "no-op on missing weight" contract)
        // and leaving every layer's attention running on all-zero Q/K/V the whole time.
        public float[]? QW, QB, KW, KB, VW, VB;
        public float[]? OW;
        public float[]? OB;
        public float[]? Ln1W, Ln1B, Ln2W, Ln2B;
        public float[]? FfnUpW, FfnGateW, FfnDownW;
        public float[]? FfnUpB, FfnGateB, FfnDownB;
        public int FfnIntermediate;
        public float[]? AttnSinks;
    }

    public int ProjectionDim => _projDim;

    public MimoVlVisionEncoder(MimoVlVisionModel model)
    {
        _m       = model;
        _embd    = model.EmbeddingDim;
        _heads   = model.HeadCount;
        _kvHeads = model.KvHeadCount;
        _headDim = model.HeadDim;
        _layers  = model.LayerCount;
        _projDim = model.ProjectionDim;
        _nMerge  = model.NMerge;
        _eps     = model.Eps;

        var gguf = model.Gguf;

        _patchEmbd0W = VisionOps.LoadTensorF32(gguf, "v.patch_embd.0.weight", "v.patch_embd.weight");
        _patchEmbd1W = VisionOps.LoadTensorF32(gguf, "v.patch_embd.1.weight", "v.patch_embd.weight.1");
        _postLnW     = VisionOps.GetTensorArray(gguf, "v.post_ln.weight");
        _postLnB     = VisionOps.GetTensorArray(gguf, "v.post_ln.bias");
        _mm0W        = VisionOps.LoadTensorF32(gguf,  "mm.0.weight");
        _mm1W        = VisionOps.LoadTensorF32(gguf,  "mm.1.weight", "mm.2.weight");

        var mm0T = gguf.FindTensor("mm.0.weight");
        _mm0OutDim = mm0T.HasValue ? (int)mm0T.Value.Dimensions[1] : _projDim;
        var mm1T = gguf.FindTensor("mm.1.weight") ?? gguf.FindTensor("mm.2.weight");
        _mm1OutDim = mm1T.HasValue ? (int)mm1T.Value.Dimensions[1] : _projDim;

        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var upT = gguf.FindTensor($"v.blk.{l}.ffn_up.weight");
            _blocks[l] = new LayerWeights
            {
                QkvW     = VisionOps.LoadTensorF32(gguf,  $"v.blk.{l}.attn_qkv.weight"),
                QkvB     = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_qkv.bias"),
                QW       = VisionOps.LoadTensorF32(gguf,  $"v.blk.{l}.attn_q.weight"),
                QB       = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_q.bias"),
                KW       = VisionOps.LoadTensorF32(gguf,  $"v.blk.{l}.attn_k.weight"),
                KB       = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_k.bias"),
                VW       = VisionOps.LoadTensorF32(gguf,  $"v.blk.{l}.attn_v.weight"),
                VB       = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_v.bias"),
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
                AttnSinks = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_sinks"),
            };
        }
    }

    /// <summary>
    /// Encodes preprocessed image CHW pixels -> projected token embeddings.
    /// </summary>
    public float[] Forward(
        float[] chw, int imgWidth, int imgHeight,
        int patchesX, int patchesY,
        out int tokenCount)
    {
        int patchSize = _m.PatchSize;
        int nPatches  = patchesX * patchesY;

        // 1. Dual Conv2D patch embed extraction + summation
        var x = new float[nPatches * _embd];
        ExtractDualPatchEmbeddings(chw, imgWidth, imgHeight, patchesX, patchesY, patchSize, x);

        // Windowed/local attention -- same simple n_wa_pattern spatial-window mechanism as
        // Exaone4/QwenVl (see those encoders' matching comments): this class's real local
        // checkpoints are qwen2.5vl_merger-projector-typed, not the more elaborate real MIMOVL
        // row/col-banded-sink graph, so that's the mechanism that actually applies here.
        bool useWindowAttn = _m.WindowAttnPattern > 0;
        int[]? windowId = null;
        if (useWindowAttn)
        {
            int gridWindow = Math.Max(1, _m.WindowSize / patchSize / _nMerge);
            int mergeCols = Math.Max(1, patchesX / _nMerge);
            int windowCols = (mergeCols + gridWindow - 1) / gridWindow;
            windowId = new int[nPatches];
            for (int py = 0; py < patchesY; py++)
            {
                int windowRow = (py / _nMerge) / gridWindow;
                for (int px = 0; px < patchesX; px++)
                {
                    int windowCol = (px / _nMerge) / gridWindow;
                    windowId[py * patchesX + px] = windowRow * windowCols + windowCol;
                }
            }
        }

        var normed = new float[nPatches * _embd];
        int qkvOutDim = (_heads + 2 * _kvHeads) * _headDim;
        var qkvBuf = new float[nPatches * qkvOutDim];

        var qBuf = new float[nPatches * _heads * _headDim];
        var kBuf = new float[nPatches * _kvHeads * _headDim];
        var vBuf = new float[nPatches * _kvHeads * _headDim];

        var attnOut = new float[nPatches * _heads * _headDim];
        var tmp = new float[nPatches * _embd];

        int maxIntermediate = 0;
        for (int l = 0; l < _layers; l++)
        {
            if (_blocks[l].FfnIntermediate > maxIntermediate) maxIntermediate = _blocks[l].FfnIntermediate;
        }
        var up = new float[nPatches * maxIntermediate];
        var gate = new float[nPatches * maxIntermediate];

        for (int l = 0; l < _layers; l++)
        {
            var b = _blocks[l];

            fixed (float* ln1W = b.Ln1W, qkvB = b.QkvB, qb = b.QB, kb = b.KB, vb = b.VB, ob = b.OB,
                   ln2W = b.Ln2W, ffnUpB = b.FfnUpB, ffnGateB = b.FfnGateB, ffnDownB = b.FfnDownB,
                   attnSinks = b.AttnSinks)
            {
                // Pre-Attn Norm (RMSNorm)
                Array.Copy(x, normed, x.Length);
                VisionOps.RmsNorm(normed, nPatches, _embd, ln1W, _eps);

                // Compute QKV -- fused if present, else separate Q/K/V (real local checkpoints)
                if (b.QkvW != null)
                {
                    VisionOps.MatVec(normed, b.QkvW, qkvB, nPatches, _embd, qkvOutDim, qkvBuf);
                    SplitQkv(qkvBuf, nPatches, _heads, _kvHeads, _headDim, qBuf, kBuf, vBuf);
                }
                else
                {
                    VisionOps.MatVec(normed, b.QW, qb, nPatches, _embd, _heads * _headDim, qBuf);
                    VisionOps.MatVec(normed, b.KW, kb, nPatches, _embd, _kvHeads * _headDim, kBuf);
                    VisionOps.MatVec(normed, b.VW, vb, nPatches, _embd, _kvHeads * _headDim, vBuf);
                }

                // 2D M-RoPE
                // ApplyMRoPE's real parameter order is (q,k,patchesX,patchesY,...) -- this call
                // had them swapped (harmless only on square test images; wrong on real non-square
                // ones, where row/col position would be transposed).
                VisionOps.ApplyMRoPE(qBuf, kBuf, patchesX, patchesY, _heads, _kvHeads, _headDim, theta: 10000.0f);

                // GQA Attention with optional Attention Sinks -- windowed except every
                // waPattern-th layer (real full_attn = (il+1) % waPattern == 0)
                bool fullAttn = !useWindowAttn || (l + 1) % _m.WindowAttnPattern == 0;
                if (fullAttn)
                    VisionOps.AttentionGqa(qBuf, kBuf, vBuf, nPatches, _heads, _kvHeads, _headDim, attnOut, attnSinks);
                else
                    VisionOps.AttentionGqaWindowed(qBuf, kBuf, vBuf, nPatches, _heads, _kvHeads, _headDim, attnOut, windowId!);

                // Project attention out & residual
                VisionOps.MatVec(attnOut, b.OW, ob, nPatches, _heads * _headDim, _embd, tmp);
                for (int i = 0; i < x.Length; i++) x[i] += tmp[i];

                // Pre-FFN Norm (RMSNorm)
                Array.Copy(x, normed, x.Length);
                VisionOps.RmsNorm(normed, nPatches, _embd, ln2W, _eps);

                // SwiGLU FFN with optional biases
                int inter = b.FfnIntermediate;
                VisionOps.MatVec(normed, b.FfnUpW, ffnUpB, nPatches, _embd, inter, up);
                VisionOps.MatVec(normed, b.FfnGateW, ffnGateB, nPatches, _embd, inter, gate);
                int interLen = nPatches * inter;
                VisionOps.Silu(gate.AsSpan(0, interLen));
                for (int i = 0; i < interLen; i++) up[i] *= gate[i];

                VisionOps.MatVec(up, b.FfnDownW, ffnDownB, nPatches, inter, _embd, tmp);
                for (int i = 0; i < x.Length; i++) x[i] += tmp[i];
            }
        }

        // Post-Norm -- real qwen2vl.cpp/exaone4_5.cpp use RMSNorm uniformly throughout the ViT
        // (norm_t=NORM_TYPE_RMS for the whole graph, including this final norm); using plain
        // LayerNorm (mean-centered) here instead was a real mismatch against the graph these
        // checkpoints actually run.
        if (_postLnW != null)
        {
            fixed (float* postLnW = _postLnW)
            {
                VisionOps.RmsNorm(x, nPatches, _embd, postLnW, _eps);
            }
        }

        // 4-patch spatial merge: 2x2 grid -> 4 * embd
        int outX = Math.Max(1, patchesX / 2);
        int outY = Math.Max(1, patchesY / 2);
        tokenCount = outX * outY;
        int mergedDim = _embd * 4;

        var merged = new float[tokenCount * mergedDim];
        VisionOps.PixelShuffle2x2(x, patchesY, patchesX, _embd, merged);

        // Projector: 2-layer GELU MLP (mm.0 -> GELU -> mm.1)
        var mm0Out = new float[tokenCount * _mm0OutDim];
        VisionOps.MatVec(merged, _mm0W, null, tokenCount, mergedDim, _mm0OutDim, mm0Out);
        VisionOps.Gelu(mm0Out);

        var mm1Out = new float[tokenCount * _mm1OutDim];
        VisionOps.MatVec(mm0Out, _mm1W, null, tokenCount, _mm0OutDim, _mm1OutDim, mm1Out);

        return mm1Out;
    }

    private void ExtractDualPatchEmbeddings(
        float[] chw, int imgW, int imgH, int px, int py, int ps, float[] dst)
    {
        int nPatches = px * py;
        int patchDim = 3 * ps * ps;
        var patches = new float[nPatches * patchDim];

        Parallel.For(0, py, iy =>
        {
            for (int ix = 0; ix < px; ix++)
            {
                int pIdx = iy * px + ix;
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

        // Sum patch embedding 0 and patch embedding 1
        VisionOps.MatVec(patches, _patchEmbd0W, null, nPatches, patchDim, _embd, dst);

        if (_patchEmbd1W != null)
        {
            var p1 = new float[nPatches * _embd];
            VisionOps.MatVec(patches, _patchEmbd1W, null, nPatches, patchDim, _embd, p1);
            for (int i = 0; i < dst.Length; i++) dst[i] += p1[i];
        }
    }

    private static void SplitQkv(
        float[] qkv, int nPatches, int heads, int kvHeads, int headDim,
        float[] q, float[] k, float[] v)
    {
        int qDim = heads * headDim;
        int kvDim = kvHeads * headDim;
        int rowDim = qDim + 2 * kvDim;

        Parallel.For(0, nPatches, t =>
        {
            int srcOff = t * rowDim;
            Array.Copy(qkv, srcOff, q, t * qDim, qDim);
            Array.Copy(qkv, srcOff + qDim, k, t * kvDim, kvDim);
            Array.Copy(qkv, srcOff + qDim + kvDim, v, t * kvDim, kvDim);
        });
    }

}
