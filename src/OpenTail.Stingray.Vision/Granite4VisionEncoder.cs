
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native C# IBM Granite 4 Vision SigLIP Vision Tower + WindowQFormer Spatial Projector.
///
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/granite4-vision.cpp
/// (clip_graph_granite4_vision::build()/build_block()) plus the runtime index-construction
/// math in examples/llama.cpp/llama.cpp/tools/mtmd/clip.cpp (PROJECTOR_TYPE_GRANITE4_VISION
/// case, lines ~5128-5203: make_win_idx/make_unwin_idx/make_spatial_idx). The projector is a
/// stack of K independent "WindowQFormer" blocks (K = clip.vision.feature_layer's length, 8 for
/// the real 3B checkpoint): each block reads a DIFFERENT intermediate SigLIP layer's hidden
/// state (not just the final layer), partitions it into (window_side x window_side) windows,
/// downsamples each window to a (query_side x query_side) query grid (either an average-pool or
/// an indexed 2x2-checkerboard-phase gather depending on that block's spatial_offset), runs a
/// self-attention + cross-attention QFormer sub-layer (learned query embeddings attending first
/// to themselves, then cross-attending to the windowed encoder features) plus an FFN, scatters
/// back to raster order ("unwin"), and projects 1152 (SigLIP dim) -> 2560 (LLM dim) with its own
/// output linear layer. All K blocks' outputs are concatenated along the TOKEN axis (not the
/// feature axis) to form the final soft-token sequence returned to the LLM.
/// </summary>
public sealed unsafe class Granite4VisionEncoder
{
    private readonly Granite4VisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly float _eps;

    private readonly float[] _patchEmbdWF32;
    private readonly float[]? _patchEmbdB;
    private readonly float[] _posEmbdF32;

    private readonly SiglipLayerWeights[] _blocks;
    private readonly QfBlockWeights[] _qfBlocks;

    private const float QformerEps = 1e-12f;
    private const int QfHeadDim = 64; // fixed d_h in the real reference, not derived from head_count metadata.

    private sealed class SiglipLayerWeights
    {
        public float[]? Ln1W;
        public float[]? Ln1B;
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

    private sealed class QfBlockWeights
    {
        public int FeatureLayer;
        public int SpatialOffset;

        public float[]? NormW;
        public float[]? NormB;
        public float[]? PostNormW;
        public float[]? PostNormB;

        public float[] Query = [];       // [16, 1152] flattened rows, query_len x n_embd
        public float[] ImgPos = [];      // [64, 1152] flattened rows, enc_len x n_embd

        public VisionTensorRef SaQW;
        public float[]? SaQB;
        public VisionTensorRef SaKW;
        public float[]? SaKB;
        public VisionTensorRef SaVW;
        public float[]? SaVB;
        public VisionTensorRef SaOW;
        public float[]? SaOB;
        public float[]? SaNormW;
        public float[]? SaNormB;

        public VisionTensorRef CaQW;
        public float[]? CaQB;
        public VisionTensorRef CaKW;
        public float[]? CaKB;
        public VisionTensorRef CaVW;
        public float[]? CaVB;
        public VisionTensorRef CaOW;
        public float[]? CaOB;
        public float[]? CaNormW;
        public float[]? CaNormB;

        public VisionTensorRef FfnUpW;
        public float[]? FfnUpB;
        public VisionTensorRef FfnDownW;
        public float[]? FfnDownB;
        public int FfnIntermediate;
        public float[]? FfnNormW;
        public float[]? FfnNormB;

        public VisionTensorRef LinearW;
        public float[]? LinearB;
    }

    public int EmbeddingDim => _embd;
    public int ProjectionDim => _projDim;

    public Granite4VisionEncoder(Granite4VisionModel model)
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
        _posEmbdF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.position_embd.weight", "v.position_embd"));

        _blocks = new SiglipLayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var upTensor = gguf.FindTensor($"v.blk.{l}.ffn_up.weight");
            int intermediate = upTensor.HasValue ? (int)upTensor.Value.Dimensions[1] : (_embd * 4);

            _blocks[l] = new SiglipLayerWeights
            {
                Ln1W = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln1.weight"),
                Ln1B = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln1.bias"),
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

        int k = model.FeatureLayers.Length;
        _qfBlocks = new QfBlockWeights[k];
        for (int b = 0; b < k; b++)
        {
            string p = $"v.proj_blk.{b}.";
            var upTensor = gguf.FindTensor(p + "ffn_up.weight");
            int intermediate = upTensor.HasValue ? (int)upTensor.Value.Dimensions[1] : (_embd * 4);

            _qfBlocks[b] = new QfBlockWeights
            {
                FeatureLayer = model.FeatureLayers[b],
                SpatialOffset = model.ProjSpatialOffsets[b],

                NormW = VisionOps.GetTensorArray(gguf, p + "norm.weight"),
                NormB = VisionOps.GetTensorArray(gguf, p + "norm.bias"),
                PostNormW = VisionOps.GetTensorArray(gguf, p + "post_norm.weight"),
                PostNormB = VisionOps.GetTensorArray(gguf, p + "post_norm.bias"),

                Query = VisionOps.GetTensorArray(gguf, p + "query") ?? [],
                ImgPos = VisionOps.GetTensorArray(gguf, p + "img_pos") ?? [],

                SaQW = VisionOps.GetTensor(gguf, p + "self_attn_q.weight"),
                SaQB = VisionOps.GetTensorArray(gguf, p + "self_attn_q.bias"),
                SaKW = VisionOps.GetTensor(gguf, p + "self_attn_k.weight"),
                SaKB = VisionOps.GetTensorArray(gguf, p + "self_attn_k.bias"),
                SaVW = VisionOps.GetTensor(gguf, p + "self_attn_v.weight"),
                SaVB = VisionOps.GetTensorArray(gguf, p + "self_attn_v.bias"),
                SaOW = VisionOps.GetTensor(gguf, p + "self_attn_out.weight"),
                SaOB = VisionOps.GetTensorArray(gguf, p + "self_attn_out.bias"),
                SaNormW = VisionOps.GetTensorArray(gguf, p + "self_attn_norm.weight"),
                SaNormB = VisionOps.GetTensorArray(gguf, p + "self_attn_norm.bias"),

                CaQW = VisionOps.GetTensor(gguf, p + "cross_attn_q.weight"),
                CaQB = VisionOps.GetTensorArray(gguf, p + "cross_attn_q.bias"),
                CaKW = VisionOps.GetTensor(gguf, p + "cross_attn_k.weight"),
                CaKB = VisionOps.GetTensorArray(gguf, p + "cross_attn_k.bias"),
                CaVW = VisionOps.GetTensor(gguf, p + "cross_attn_v.weight"),
                CaVB = VisionOps.GetTensorArray(gguf, p + "cross_attn_v.bias"),
                CaOW = VisionOps.GetTensor(gguf, p + "cross_attn_out.weight"),
                CaOB = VisionOps.GetTensorArray(gguf, p + "cross_attn_out.bias"),
                CaNormW = VisionOps.GetTensorArray(gguf, p + "cross_attn_norm.weight"),
                CaNormB = VisionOps.GetTensorArray(gguf, p + "cross_attn_norm.bias"),

                FfnUpW = VisionOps.GetTensor(gguf, p + "ffn_up.weight"),
                FfnUpB = VisionOps.GetTensorArray(gguf, p + "ffn_up.bias"),
                FfnDownW = VisionOps.GetTensor(gguf, p + "ffn_down.weight"),
                FfnDownB = VisionOps.GetTensorArray(gguf, p + "ffn_down.bias"),
                FfnIntermediate = intermediate,
                FfnNormW = VisionOps.GetTensorArray(gguf, p + "ffn_norm.weight"),
                FfnNormB = VisionOps.GetTensorArray(gguf, p + "ffn_norm.bias"),

                LinearW = VisionOps.GetTensor(gguf, p + "linear.weight"),
                LinearB = VisionOps.GetTensorArray(gguf, p + "linear.bias"),
            };
        }
    }

    public float[] Forward(float[] chw, int targetWidth, int targetHeight, int patchesX, int patchesY, out int tokenCount)
    {
        var img = new Granite4PreprocessedImage(chw, targetWidth, targetHeight, patchesX, patchesY);
        return Encode(img, out tokenCount);
    }

    public float[] Encode(Granite4PreprocessedImage img, out int tokenCount)
    {
        int numPatches = img.PatchesX * img.PatchesY;

        // --- Stage 1a: SigLIP tower, saving every layer's output (not just the final one) ---
        var hiddenStates = new float[numPatches * _embd];
        fixed (float* chwPtr = img.Chw)
        {
            ExtractPatches(chwPtr, img.TargetWidth, img.TargetHeight, img.PatchesX, img.PatchesY, hiddenStates);
        }

        if (_posEmbdF32.Length > 0)
        {
            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += _posEmbdF32[i % _embd];
        }

        var layerOuts = new float[_layers][];

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

        for (int l = 0; l < _layers; l++)
        {
            var blk = _blocks[l];

            fixed (float* ln1W = blk.Ln1W, ln1B = blk.Ln1B, attnQB = blk.AttnQB, attnKB = blk.AttnKB,
                   attnVB = blk.AttnVB, attnOutB = blk.AttnOutB, ln2W = blk.Ln2W, ln2B = blk.Ln2B,
                   ffnUpB = blk.FfnUpB, ffnDownB = blk.FfnDownB)
            {
                Array.Copy(hiddenStates, normed, hiddenStates.Length);
                VisionOps.LayerNorm(normed, numPatches, _embd, ln1W, ln1B, _eps);

                VisionOps.MatVecAny(normed, blk.AttnQW, attnQB, numPatches, _embd, _embd, qBuf);
                VisionOps.MatVecAny(normed, blk.AttnKW, attnKB, numPatches, _embd, _embd, kBuf);
                VisionOps.MatVecAny(normed, blk.AttnVW, attnVB, numPatches, _embd, _embd, vBuf);

                VisionOps.Attention(qBuf, kBuf, vBuf, numPatches, _heads, _headDim, normed);
                VisionOps.MatVecAny(normed, blk.AttnOutW, attnOutB, numPatches, _embd, _embd, attnOut);

                for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];

                Array.Copy(hiddenStates, normed, hiddenStates.Length);
                VisionOps.LayerNorm(normed, numPatches, _embd, ln2W, ln2B, _eps);

                int intermediate = blk.FfnIntermediate;
                VisionOps.MatVecAny(normed, blk.FfnUpW, ffnUpB, numPatches, _embd, intermediate, ffnMid);
                VisionOps.Gelu(ffnMid.AsSpan(0, numPatches * intermediate));
                VisionOps.MatVecAny(ffnMid, blk.FfnDownW, ffnDownB, numPatches, intermediate, _embd, attnOut);

                for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];
            }

            // Real reference saves EVERY layer's residual-stream output (post attn+FFN,
            // pre-post_ln) as layer_outs[il] -- see granite4-vision.cpp build(), "cb(cur,
            // layer_out, il); layer_outs[il] = cur;". post_ln is only applied to the SigLIP
            // tower's own final-layer output, never to the intermediate layers the QFormer
            // blocks read from, so we snapshot here (before any post_ln) rather than at the end.
            var snapshot = new float[hiddenStates.Length];
            Array.Copy(hiddenStates, snapshot, hiddenStates.Length);
            layerOuts[l] = snapshot;
        }

        // --- Stage 1b/1c: WindowQFormer blocks, concatenated along the TOKEN axis ---
        int imageSide = img.PatchesX;
        int windowSide = _m.WindowSide;
        int querySide = _m.QuerySide;
        int n = imageSide / windowSide;
        int newSide = n * querySide;
        int tokensPerBlock = newSide * newSide;
        int k = _qfBlocks.Length;

        tokenCount = k * tokensPerBlock;
        var final = new float[tokenCount * _projDim];

        // Precompute the raster<->window permutation indices once per Forward call (they depend
        // only on imageSide/windowSide/querySide, which are the same for every block; only the
        // spatial_idx phase differs per block). Matches make_win_idx/make_unwin_idx/
        // make_spatial_idx in tools/mtmd/clip.cpp lines 5144-5183.
        int[] winIdx = MakeWinIdx(imageSide, windowSide);
        int[] qwinIdx = MakeWinIdx(newSide, querySide);
        int[] unwinIdx = MakeUnwinIdx(newSide, querySide);

        for (int b = 0; b < k; b++)
        {
            var blk = _qfBlocks[b];
            var h = layerOuts[blk.FeatureLayer];
            RunBlock(blk, h, imageSide, windowSide, querySide, n, newSide, winIdx, qwinIdx, unwinIdx,
                final, b * tokensPerBlock * _projDim);
        }

        return final;
    }

    private void RunBlock(
        QfBlockWeights blk,
        float[] h,
        int imageSide,
        int windowSide,
        int querySide,
        int n,
        int newSide,
        int[] winIdx,
        int[] qwinIdx,
        int[] unwinIdx,
        float[] outBuf,
        int outOffset)
    {
        int nEmbd = _embd;
        int nWindows = n * n;
        int encLen = windowSide * windowSide;
        int queryLen = querySide * querySide;
        int numPatches = imageSide * imageSide;
        int newTokens = newSide * newSide;

        // 1. Top-level LN (model eps, not qformer eps -- build_norm(h, qf_proj_norm_*, ..., eps, bid)).
        var x = new float[h.Length];
        Array.Copy(h, x, h.Length);
        fixed (float* nw = blk.NormW, nb = blk.NormB)
        {
            VisionOps.LayerNorm(x, numPatches, nEmbd, nw, nb, _eps);
        }

        // 2. enc = _win(x): window-gather to (n_windows*enc_len, n_embd).
        var enc = new float[nWindows * encLen * nEmbd];
        VisionOps.GatherRows(x, winIdx, nEmbd, enc);

        // 3. downsampled = downsampler(x): either avg-pool or an indexed checkerboard gather.
        var d = new float[newTokens * nEmbd];
        if (blk.SpatialOffset >= 0)
        {
            int[] spatialIdx = MakeSpatialIdx(imageSide, blk.SpatialOffset);
            VisionOps.GatherRows(x, spatialIdx, nEmbd, d);
        }
        else
        {
            VisionOps.AvgPoolDownsample2D(x, imageSide, newSide, nEmbd, d);
        }

        // 4. query_embeds = query + _win(d): (n_windows*query_len, n_embd).
        var dw = new float[nWindows * queryLen * nEmbd];
        VisionOps.GatherRows(d, qwinIdx, nEmbd, dw);
        var qIn = new float[nWindows * queryLen * nEmbd];
        for (int w = 0; w < nWindows; w++)
        {
            for (int i = 0; i < queryLen; i++)
            {
                int row = (w * queryLen + i) * nEmbd;
                int qrow = i * nEmbd;
                for (int e = 0; e < nEmbd; e++) qIn[row + e] = dw[row + e] + blk.Query[qrow + e];
            }
        }

        // 5. encoder_embeds = enc + image_positions: (n_windows*enc_len, n_embd).
        var eIn = new float[nWindows * encLen * nEmbd];
        for (int w = 0; w < nWindows; w++)
        {
            for (int i = 0; i < encLen; i++)
            {
                int row = (w * encLen + i) * nEmbd;
                int prow = i * nEmbd;
                for (int e = 0; e < nEmbd; e++) eIn[row + e] = enc[row + e] + blk.ImgPos[prow + e];
            }
        }

        int nq = nWindows * queryLen;
        int nkv = nWindows * encLen;
        int nHead = nEmbd / QfHeadDim;

        // 6. QFormer sub-layer.
        var q = new float[qIn.Length];
        Array.Copy(qIn, q, qIn.Length);
        fixed (float* pnw = blk.PostNormW, pnb = blk.PostNormB)
        {
            VisionOps.LayerNorm(q, nq, nEmbd, pnw, pnb, QformerEps);
        }

        // 6a. Self-attention.
        var saQ = new float[nq * nEmbd];
        var saK = new float[nq * nEmbd];
        var saV = new float[nq * nEmbd];
        var saAttn = new float[nq * nEmbd];
        var saOut = new float[nq * nEmbd];
        fixed (float* qb = blk.SaQB, kb = blk.SaKB, vb = blk.SaVB, ob = blk.SaOB)
        {
            VisionOps.MatVecAny(q, blk.SaQW, qb, nq, nEmbd, nEmbd, saQ);
            VisionOps.MatVecAny(q, blk.SaKW, kb, nq, nEmbd, nEmbd, saK);
            VisionOps.MatVecAny(q, blk.SaVW, vb, nq, nEmbd, nEmbd, saV);
            VisionOps.WindowedAttention(saQ, saK, saV, nWindows, queryLen, queryLen, nHead, QfHeadDim, saAttn);
            VisionOps.MatVecAny(saAttn, blk.SaOW, ob, nq, nEmbd, nEmbd, saOut);
        }
        for (int i = 0; i < saOut.Length; i++) saOut[i] += q[i];
        fixed (float* snw = blk.SaNormW, snb = blk.SaNormB)
        {
            VisionOps.LayerNorm(saOut, nq, nEmbd, snw, snb, QformerEps);
        }

        // 6b. Cross-attention: Q from sa_out, K/V from encoder_embeds, windowed.
        var caQ = new float[nq * nEmbd];
        var caK = new float[nkv * nEmbd];
        var caV = new float[nkv * nEmbd];
        var caAttn = new float[nq * nEmbd];
        var caOut = new float[nq * nEmbd];
        fixed (float* qb = blk.CaQB, kb = blk.CaKB, vb = blk.CaVB, ob = blk.CaOB)
        {
            VisionOps.MatVecAny(saOut, blk.CaQW, qb, nq, nEmbd, nEmbd, caQ);
            VisionOps.MatVecAny(eIn, blk.CaKW, kb, nkv, nEmbd, nEmbd, caK);
            VisionOps.MatVecAny(eIn, blk.CaVW, vb, nkv, nEmbd, nEmbd, caV);
            VisionOps.WindowedAttention(caQ, caK, caV, nWindows, queryLen, encLen, nHead, QfHeadDim, caAttn);
            VisionOps.MatVecAny(caAttn, blk.CaOW, ob, nq, nEmbd, nEmbd, caOut);
        }
        for (int i = 0; i < caOut.Length; i++) caOut[i] += saOut[i];
        fixed (float* cnw = blk.CaNormW, cnb = blk.CaNormB)
        {
            VisionOps.LayerNorm(caOut, nq, nEmbd, cnw, cnb, QformerEps);
        }

        // 6c. FFN (gelu_erf, per the real reference's ggml_gelu_erf call).
        int ffnDim = blk.FfnIntermediate;
        var ffnMid = new float[nq * ffnDim];
        var ffnOut = new float[nq * nEmbd];
        fixed (float* ub = blk.FfnUpB, db = blk.FfnDownB)
        {
            VisionOps.MatVecAny(caOut, blk.FfnUpW, ub, nq, nEmbd, ffnDim, ffnMid);
            for (int i = 0; i < ffnMid.Length; i++) ffnMid[i] = GeluErf(ffnMid[i]);
            VisionOps.MatVecAny(ffnMid, blk.FfnDownW, db, nq, ffnDim, nEmbd, ffnOut);
        }
        for (int i = 0; i < ffnOut.Length; i++) ffnOut[i] += caOut[i];
        fixed (float* fnw = blk.FfnNormW, fnb = blk.FfnNormB)
        {
            VisionOps.LayerNorm(ffnOut, nq, nEmbd, fnw, fnb, QformerEps);
        }

        // 7. _unwin back to raster spatial order: (n_windows*query_len, n_embd) -> (new_side^2, n_embd).
        var unwinned = new float[newTokens * nEmbd];
        VisionOps.GatherRows(ffnOut, unwinIdx, nEmbd, unwinned);

        // 8. out_linear: 1152 -> projDim (2560), written directly into this block's token range
        //    of the final concatenated-along-tokens output buffer.
        var outSlice = new float[newTokens * _projDim];
        fixed (float* lb = blk.LinearB)
        {
            VisionOps.MatVecAny(unwinned, blk.LinearW, lb, newTokens, nEmbd, _projDim, outSlice);
        }
        Array.Copy(outSlice, 0, outBuf, outOffset, outSlice.Length);
    }

    /// <summary>
    /// Raster-to-window permutation: dst[w*(win*win)+p] = source raster index, for a (side,side)
    /// grid split into (side/win)^2 windows of (win,win) tokens each. Matches make_win_idx in
    /// tools/mtmd/clip.cpp lines 5144-5161 exactly (same loop structure and indexing formula).
    /// </summary>
    private static int[] MakeWinIdx(int side, int win)
    {
        int nn = side / win;
        var idx = new int[side * side];
        for (int wy = 0; wy < nn; wy++)
        {
            for (int wx = 0; wx < nn; wx++)
            {
                for (int iy = 0; iy < win; iy++)
                {
                    for (int ix = 0; ix < win; ix++)
                    {
                        int w = wy * nn + wx;
                        int p = iy * win + ix;
                        int y = wy * win + iy;
                        int x = wx * win + ix;
                        idx[w * (win * win) + p] = y * side + x;
                    }
                }
            }
        }
        return idx;
    }

    /// <summary>Inverse permutation of <see cref="MakeWinIdx"/>. Matches make_unwin_idx (clip.cpp lines 5163-5170).</summary>
    private static int[] MakeUnwinIdx(int side, int win)
    {
        var fwd = MakeWinIdx(side, win);
        var inv = new int[fwd.Length];
        for (int i = 0; i < fwd.Length; i++) inv[fwd[i]] = i;
        return inv;
    }

    /// <summary>
    /// 2x2-checkerboard-phase subsample index: idx[y*newS+x] = (y*2+off_y)*side + (x*2+off_x).
    /// Matches make_spatial_idx (clip.cpp lines 5172-5183) exactly.
    /// </summary>
    private static int[] MakeSpatialIdx(int side, int offset)
    {
        int offY = (offset >> 1) & 1;
        int offX = offset & 1;
        int newS = side / 2;
        var idx = new int[newS * newS];
        for (int y = 0; y < newS; y++)
        {
            for (int x = 0; x < newS; x++)
            {
                idx[y * newS + x] = (y * 2 + offY) * side + (x * 2 + offX);
            }
        }
        return idx;
    }

    /// <summary>Real erf-based GELU: 0.5*x*(1+erf(x/sqrt(2))), matching the reference's ggml_gelu_erf.</summary>
    private static float GeluErf(float x) => 0.5f * x * (1.0f + Erf(x * 0.7071067811865476f));

    // Abramowitz-Stegun 7.1.26 approximation, max error ~1.5e-7.
    private static float Erf(float x)
    {
        float sign = x < 0 ? -1f : 1f;
        x = MathF.Abs(x);
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f;
        const float a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float t = 1.0f / (1.0f + p * x);
        float y = 1.0f - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);
        return sign * y;
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
