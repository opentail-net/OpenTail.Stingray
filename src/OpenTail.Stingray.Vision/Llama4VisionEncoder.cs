using System.Numerics.Tensors;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// The Llama 4 (E4B Scout/Maverick, <c>clip.projector_type=llama4</c>) ViT encoder + pixel-shuffle
/// merger + projector. Turns one preprocessed square tile into LLM soft-token embeddings.
///
/// Ported stage-by-stage from the real reference (<c>tools/mtmd/models/llama4.cpp</c>'s
/// <c>clip_graph_llama4::build()</c> plus the shared <c>clip_graph::build_vit</c>/
/// <c>build_attn</c>/<c>build_ffn</c>/<c>build_rope_2d</c> helpers in <c>tools/mtmd/clip.cpp</c>),
/// verified against the real <c>models/mmproj-llama-4-scout-17b-16e-instruct-f16.gguf</c> -- see
/// docs/06-llama4-vision-plan.md.
///
/// <para>Distinctive relative to <see cref="Gemma4VVisionEncoder"/>/<see cref="Gemma3VisionEncoder"/>:
/// a real [CLS] token (concatenated after patch embed, dropped again before the merger), a
/// learned absolute position table AND 2D half-RoPE active simultaneously (not either/or), real
/// pre- and post-layernorm (both present in this checkpoint, unlike gemma4v which has neither), a
/// pixel-shuffle (space-to-depth) merger instead of average-pooling, and NORM/interleaved rope
/// pairing (<c>clip_graph::build_rope_2d</c>'s default) rather than gemma4v's hand-rolled NEOX
/// split-half convention -- confirmed directly from <c>clip.cpp</c>, see
/// <see cref="ApplyRope2DHalfNorm"/>'s doc comment. This checkpoint's per-block FFN has NO gate
/// tensor at all (confirmed via list-tensors) -- plain <c>down(act(up(x)))</c>, not gated, despite
/// <c>llama4.cpp</c>'s generic <c>build_ffn</c> call being able to support gating if present.</para>
///
/// <para>Per-tile pipeline: patch embed (F16 matvec, no bias) -&gt; concat [CLS] as the last token
/// -&gt; add learned position table (direct add, all n_patches+1 rows including CLS) -&gt; pre_ln
/// -&gt; 34 blocks -&gt; post_ln -&gt; drop [CLS] -&gt; pixel-shuffle merge (kernel=stride=NMerge=2)
/// -&gt; 2-layer GELU MLP (mlp.1 -&gt; gelu -&gt; mlp.2 -&gt; gelu, both no bias, GELU applied after
/// EACH layer per <c>Llama4VisionMLP2</c>) -&gt; final linear projector (mm.model.fc, no bias, no
/// activation).</para>
///
/// <para>Per block (real pre-norm, no sandwich norm, no QK-norm, no V-norm): LayerNorm(ln1, with
/// bias) -&gt; separate Q/K/V projections (each with a bias-add) -&gt; 2D half-RoPE on Q/K only
/// (NORM/interleaved pairing, first half of head-dim rotated by column position, second half by
/// row position; [CLS] gets position (0,0) so it is effectively unrotated -- theta^0=1, angle=0)
/// -&gt; standard bidirectional attention over all n_patches+1 tokens, scale=1/sqrt(head_dim) -&gt;
/// output projection (with bias) -&gt; residual add -&gt; LayerNorm(ln2, with bias) -&gt; plain FFN
/// (<c>down(act(up(x)))</c>, both with bias-adds, act = this checkpoint's <c>clip.use_gelu=true</c>
/// -&gt; plain tanh-GELU) -&gt; residual add.</para>
///
/// <para><b>Scope</b>: processes exactly one fixed <c>ImageSize x ImageSize</c> square tile.
/// llama.cpp's own multi-tile ("llava-uhd") preprocessing -- deciding how many tiles a source
/// image needs, slicing it, and interleaving tile-boundary tokens in the prompt -- is explicitly
/// out of scope, same precedent as decoder splice (Phase V4) being out of scope for the other two
/// encoders. See docs/06-llama4-vision-plan.md.</para>
///
/// <para><b>Unverified end-to-end</b>: sourced from the real reference graph and the real mmproj's
/// tensor/metadata inventory, not inferred, but there is no working oracle on this machine to run
/// this encoder + Llama 4 Scout's text decoder end-to-end for numerical parity, and llama.cpp's
/// own code carries a runtime warning that this exact projector is known to have degraded quality
/// (ggml-org/llama.cpp#13282) -- a ceiling independent of whether this port itself is correct.</para>
/// </summary>
public sealed unsafe class Llama4VisionEncoder
{
    private readonly Llama4VisionModel _m;
    private readonly int _embd, _heads, _headDim, _ffLen, _blockCount, _patch, _imageSize, _projDim, _nMerge;
    private readonly float _eps, _kqScale;
    private readonly int _gridSize;
    private readonly Llama4FfnActivation _ffnActivation;

    private readonly byte* _patchEmbdW;    // F16 [patchVec, embd]
    private readonly float[] _classEmbedding;
    private readonly float[] _posTable;    // [(nPatches+1) * embd], direct elementwise add
    private readonly float[]? _preLnW, _preLnB, _postLnW, _postLnB;
    private readonly byte* _mlp1W, _mlp2W, _projW; // F16 merger weights, no bias

    private sealed class BlockData
    {
        public required float[] Ln1W;
        public required float[]? Ln1B;
        public required float[] Ln2W;
        public required float[]? Ln2B;
        public required float[]? AttnQBias, AttnKBias, AttnVBias, AttnOutBias, FfnUpBias, FfnDownBias;
        public required byte* AttnQ, AttnK, AttnV, AttnOut, FfnUp, FfnDown;
    }

    private readonly BlockData[] _blocks;

    /// <remarks>
    /// Borrows <paramref name="model"/>'s memory-mapped weights (caches raw pointers into them),
    /// so <paramref name="model"/> must outlive this encoder and must not be disposed before the
    /// last <see cref="Forward"/> call.
    /// </remarks>
    public Llama4VisionEncoder(Llama4VisionModel model)
    {
        _m = model;
        _embd = model.EmbeddingLength;
        _heads = model.HeadCount;
        _headDim = _embd / _heads;
        _ffLen = model.FeedForwardLength;
        _blockCount = model.BlockCount;
        _patch = model.PatchSize;
        _imageSize = model.ImageSize;
        _projDim = model.ProjectionDim;
        _nMerge = model.NMerge;
        _eps = model.LayerNormEps;
        _gridSize = _imageSize / _patch;
        _kqScale = 1f / MathF.Sqrt(_headDim);
        _ffnActivation = model.FfnActivation;

        _patchEmbdW = model.Gguf.GetTensorDataPtr(model.PatchEmbdWeight);
        _classEmbedding = model.LoadFloats(model.ClassEmbedding);
        _posTable = model.LoadFloats(model.PositionEmbedding);
        _preLnW = model.PreLnWeight is { } plw ? model.LoadFloats(plw) : null;
        _preLnB = model.PreLnBias is { } plb ? model.LoadFloats(plb) : null;
        _postLnW = model.PostLnWeight is { } polw ? model.LoadFloats(polw) : null;
        _postLnB = model.PostLnBias is { } polb ? model.LoadFloats(polb) : null;
        _mlp1W = model.Gguf.GetTensorDataPtr(model.MmModelMlp1Weight);
        _mlp2W = model.Gguf.GetTensorDataPtr(model.MmModelMlp2Weight);
        _projW = model.Gguf.GetTensorDataPtr(model.MmModelProj);

        _blocks = new BlockData[_blockCount];
        for (var i = 0; i < _blockCount; i++)
        {
            var b = model.Blocks[i];
            _blocks[i] = new BlockData
            {
                Ln1W = model.LoadFloats(b.Ln1W),
                Ln1B = b.Ln1B is { } l1b ? model.LoadFloats(l1b) : null,
                Ln2W = model.LoadFloats(b.Ln2W),
                Ln2B = b.Ln2B is { } l2b ? model.LoadFloats(l2b) : null,
                AttnQBias = b.AttnQBias is { } aqb ? model.LoadFloats(aqb) : null,
                AttnKBias = b.AttnKBias is { } akb ? model.LoadFloats(akb) : null,
                AttnVBias = b.AttnVBias is { } avb ? model.LoadFloats(avb) : null,
                AttnOutBias = b.AttnOutBias is { } aob ? model.LoadFloats(aob) : null,
                FfnUpBias = b.FfnUpBias is { } fub ? model.LoadFloats(fub) : null,
                FfnDownBias = b.FfnDownBias is { } fdb ? model.LoadFloats(fdb) : null,
                AttnQ = model.Gguf.GetTensorDataPtr(b.AttnQ),
                AttnK = model.Gguf.GetTensorDataPtr(b.AttnK),
                AttnV = model.Gguf.GetTensorDataPtr(b.AttnV),
                AttnOut = model.Gguf.GetTensorDataPtr(b.AttnOut),
                FfnUp = model.Gguf.GetTensorDataPtr(b.FfnUp),
                FfnDown = model.Gguf.GetTensorDataPtr(b.FfnDown),
            };
        }
    }

    /// <summary>Soft-token count this encoder produces for its fixed input grid.</summary>
    public int TokenCount => (_gridSize / _nMerge) * (_gridSize / _nMerge);

    private static readonly bool s_debug = Environment.GetEnvironmentVariable("STINGRAY_LLAMA4_DEBUG") == "1";

    /// <summary>
    /// Run the encoder on one preprocessed square tile. <paramref name="chw"/> is planar CHW,
    /// already resized to <c>ImageSize x ImageSize</c> and channel-normalized to this mmproj's
    /// declared mean/std. Returns <see cref="TokenCount"/> x <c>ProjectionDim</c> soft-token
    /// embeddings, row-major.
    /// </summary>
    public float[] Forward(ReadOnlySpan<float> chw)
    {
        _m.EnsureNotDisposed();
        var plane = _imageSize * _imageSize;
        if (chw.Length < 3L * plane)
            throw new ArgumentException($"chw length ({chw.Length}) is too small for a 3x{_imageSize}x{_imageSize} image.");

        var nPatches = _gridSize * _gridSize;
        var nPos = nPatches + 1; // +1 for [CLS], appended as the LAST token
        var patchVec = _patch * _patch * 3;
        if (s_debug) Console.Error.WriteLine($"[LLAMA4-DBG] Forward start: nPatches={nPatches} nPos={nPos}");

        // 1. im2col (raster order, x fastest -- matches clip.cpp's pos_h/pos_w fill loop) + patch
        //    embed (F16 matvec, no bias) + [CLS] concat (last token) + learned position table
        //    (direct add, all nPos rows including CLS's own trailing row).
        var patches = new float[nPatches * patchVec];
        for (var p = 0; p < nPatches; p++)
        {
            var pr = p / _gridSize;
            var pc = p % _gridSize;
            var baseY = pr * _patch;
            var baseX = pc * _patch;
            var dst = p * patchVec;
            for (var c = 0; c < 3; c++)
            {
                var cPlane = c * plane;
                var cOff = dst + c * _patch * _patch;
                for (var ky = 0; ky < _patch; ky++)
                {
                    var srcRow = cPlane + (baseY + ky) * _imageSize + baseX;
                    var dstRow = cOff + ky * _patch;
                    for (var kx = 0; kx < _patch; kx++)
                        patches[dstRow + kx] = chw[srcRow + kx];
                }
            }
        }

        var hidden = new float[nPos * _embd];
        fixed (float* pPatches = patches, pHidden = hidden, pPos = _posTable, pCls = _classEmbedding)
        {
            for (var p = 0; p < nPatches; p++)
                SimdKernels.MatVec(pHidden + p * _embd, _patchEmbdW, pPatches + p * patchVec, _embd, patchVec, DType.Float16);
            new Span<float>(pCls, _embd).CopyTo(new Span<float>(pHidden + nPatches * _embd, _embd));

            for (var p = 0; p < nPos; p++)
            {
                var row = new Span<float>(pHidden + p * _embd, _embd);
                TensorPrimitives.Add(row, new ReadOnlySpan<float>(pPos + p * _embd, _embd), row);
            }
        }
        if (s_debug) Console.Error.WriteLine("[LLAMA4-DBG] patch embed + CLS + position done");

        // pre_ln (present in this checkpoint, unlike gemma4v/gemma3 -- applies once, to all nPos
        // tokens, before the block loop, per build_vit's `if (model.pre_ln_w)` gate).
        if (_preLnW is not null)
        {
            fixed (float* pHidden = hidden, w = _preLnW, b = _preLnB)
            {
                for (var p = 0; p < nPos; p++)
                    SimdKernels.LayerNorm(pHidden + p * _embd, pHidden + p * _embd, w, b, _embd, _eps);
            }
        }

        // Precompute per-token 2D rope positions once: patches get (col+1, row+1), [CLS] gets
        // (0,0) -- confirmed exactly from clip.cpp's PROJECTOR_TYPE_LLAMA4 pos_h/pos_w fill loop
        // (1-indexed patches, CLS left at the vector's zero-initialized default).
        var posW = new int[nPos]; // first half of head-dim (X/column)
        var posH = new int[nPos]; // second half of head-dim (Y/row)
        for (var p = 0; p < nPatches; p++)
        {
            posH[p] = (p / _gridSize) + 1;
            posW[p] = (p % _gridSize) + 1;
        }
        // posH[nPatches] = posW[nPatches] = 0 (CLS), already zero-initialized.

        // 2. transformer blocks.
        var q = new float[nPos * _embd];
        var k = new float[nPos * _embd];
        var v = new float[nPos * _embd];
        var attnConcat = new float[nPos * _embd];

        fixed (float* pHidden = hidden, pQ = q, pK = k, pV = v, pAttnConcat = attnConcat)
        {
            for (var layer = 0; layer < _blockCount; layer++)
            {
                if (s_debug) Console.Error.WriteLine($"[LLAMA4-DBG] block {layer}/{_blockCount} start");
                var blk = _blocks[layer];
                fixed (float* ln1w = blk.Ln1W, ln1b = blk.Ln1B, ln2w = blk.Ln2W, ln2b = blk.Ln2B)
                {
                    // ln1 -> Q/K/V (each with an optional bias-add) -> 2D half-RoPE on Q/K only
                    // (NORM/interleaved pairing -- see ApplyRope2DHalfNorm).
                    {
                        nint hiddenBase = (nint)pHidden, qBase2 = (nint)pQ, kBase2 = (nint)pK, vBase2 = (nint)pV;
                        nint ln1wBase = (nint)ln1w, ln1bBase = (nint)ln1b;
                        var attnQL = blk.AttnQ; var attnKL = blk.AttnK; var attnVL = blk.AttnV;
                        var qBiasArr = blk.AttnQBias; var kBiasArr = blk.AttnKBias; var vBiasArr = blk.AttnVBias;
                        var posWL = posW; var posHL = posH;
                        int headsL = _heads, headDimL = _headDim;
                        float ropeTheta = Llama4VisionModel.RopeTheta;
                        Parallel.For(0, nPos, p =>
                        {
                            var hiddenL = (float*)hiddenBase;
                            var qL = (float*)qBase2;
                            var kL = (float*)kBase2;
                            var vL = (float*)vBase2;
                            var ln1wL = (float*)ln1wBase;
                            var ln1bL = ln1bBase == 0 ? null : (float*)ln1bBase;
                            var localNormed = new float[_embd];
                            fixed (float* pLocalNormed = localNormed)
                            {
                                var hp = hiddenL + p * _embd;
                                SimdKernels.LayerNorm(pLocalNormed, hp, ln1wL, ln1bL, _embd, _eps);

                                var qp = qL + p * _embd;
                                var kp = kL + p * _embd;
                                var vp = vL + p * _embd;
                                SimdKernels.MatVec(qp, attnQL, pLocalNormed, _embd, _embd, DType.Float16);
                                SimdKernels.MatVec(kp, attnKL, pLocalNormed, _embd, _embd, DType.Float16);
                                SimdKernels.MatVec(vp, attnVL, pLocalNormed, _embd, _embd, DType.Float16);
                                if (qBiasArr is not null) { var s = new Span<float>(qp, _embd); TensorPrimitives.Add(s, qBiasArr, s); }
                                if (kBiasArr is not null) { var s = new Span<float>(kp, _embd); TensorPrimitives.Add(s, kBiasArr, s); }
                                if (vBiasArr is not null) { var s = new Span<float>(vp, _embd); TensorPrimitives.Add(s, vBiasArr, s); }

                                var quarterDim = headDimL / 4;
                                var pw = posWL[p];
                                var ph = posHL[p];
                                for (var h = 0; h < headsL; h++)
                                {
                                    var qh = qp + h * headDimL;
                                    var kh = kp + h * headDimL;
                                    ApplyRope2DHalfNorm(qh, quarterDim, pw, ropeTheta);
                                    ApplyRope2DHalfNorm(qh + headDimL / 2, quarterDim, ph, ropeTheta);
                                    ApplyRope2DHalfNorm(kh, quarterDim, pw, ropeTheta);
                                    ApplyRope2DHalfNorm(kh + headDimL / 2, quarterDim, ph, ropeTheta);
                                }
                            }
                        });
                    }

                    // Bidirectional multi-head attention over all nPos tokens (including [CLS]),
                    // standard scale = 1/sqrt(head_dim). Parallelized across heads.
                    if (s_debug) Console.Error.WriteLine($"[LLAMA4-DBG] block {layer} QKV+rope done, starting attention");
                    nint qBase = (nint)pQ, kBase = (nint)pK, vBase = (nint)pV, concatBase = (nint)pAttnConcat;
                    int headDimL2 = _headDim, embdL = _embd, nPosL = nPos;
                    float kqScaleL = _kqScale;
                    Parallel.For(0, _heads, h =>
                    {
                        var qL = (float*)qBase;
                        var kL = (float*)kBase;
                        var vL = (float*)vBase;
                        var concatL = (float*)concatBase;
                        var localScores = new float[nPosL];
                        var localTemp = new float[headDimL2];
                        fixed (float* pLocalScores = localScores, pLocalTemp = localTemp)
                        {
                            var off = h * headDimL2;
                            for (var i = 0; i < nPosL; i++)
                            {
                                var qi = new ReadOnlySpan<float>(qL + i * embdL + off, headDimL2);
                                for (var j = 0; j < nPosL; j++)
                                {
                                    var kj = new ReadOnlySpan<float>(kL + j * embdL + off, headDimL2);
                                    pLocalScores[j] = TensorPrimitives.Dot(qi, kj) * kqScaleL;
                                }
                                SimdKernels.SoftmaxInPlace(pLocalScores, nPosL);

                                var outSpan = new Span<float>(concatL + i * embdL + off, headDimL2);
                                outSpan.Clear();
                                var tempSpan = new Span<float>(pLocalTemp, headDimL2);
                                for (var j = 0; j < nPosL; j++)
                                {
                                    var vj = new ReadOnlySpan<float>(vL + j * embdL + off, headDimL2);
                                    TensorPrimitives.Multiply(vj, pLocalScores[j], tempSpan);
                                    TensorPrimitives.Add(outSpan, tempSpan, outSpan);
                                }
                            }
                        }
                    });

                    // Output projection (optional bias) -> residual (no post-attn norm).
                    {
                        nint concatBase2 = (nint)pAttnConcat, hiddenBase2 = (nint)pHidden;
                        var attnOutL = blk.AttnOut;
                        var oBiasArr = blk.AttnOutBias;
                        Parallel.For(0, nPos, p =>
                        {
                            var concatL2 = (float*)concatBase2;
                            var hiddenL2 = (float*)hiddenBase2;
                            var localProj = new float[_embd];
                            fixed (float* pLocalProj = localProj)
                            {
                                SimdKernels.MatVec(pLocalProj, attnOutL, concatL2 + p * _embd, _embd, _embd, DType.Float16);
                                if (oBiasArr is not null)
                                {
                                    var s = new Span<float>(pLocalProj, _embd);
                                    TensorPrimitives.Add(s, oBiasArr, s);
                                }
                                var hp = hiddenL2 + p * _embd;
                                for (var c = 0; c < _embd; c++) hp[c] += pLocalProj[c];
                            }
                        });
                    }

                    // ln2 -> plain FFN (down(act(up(x))), no gate in this checkpoint) -> residual.
                    {
                        nint hiddenBase3 = (nint)pHidden, ln2wBase = (nint)ln2w, ln2bBase = (nint)ln2b;
                        var ffnUpL = blk.FfnUp; var ffnDownL = blk.FfnDown;
                        var upBiasArr = blk.FfnUpBias; var downBiasArr = blk.FfnDownBias;
                        int ffLenL = _ffLen;
                        var activation = _ffnActivation;
                        Parallel.For(0, nPos, p =>
                        {
                            var hiddenL3 = (float*)hiddenBase3;
                            var ln2wL = (float*)ln2wBase;
                            var ln2bL = ln2bBase == 0 ? null : (float*)ln2bBase;
                            var localFfnIn = new float[_embd];
                            var localUp = new float[ffLenL];
                            var localFfnOut = new float[_embd];
                            fixed (float* pLocalFfnIn = localFfnIn, pLocalUp = localUp, pLocalFfnOut = localFfnOut)
                            {
                                var hp = hiddenL3 + p * _embd;
                                SimdKernels.LayerNorm(pLocalFfnIn, hp, ln2wL, ln2bL, _embd, _eps);

                                SimdKernels.MatVec(pLocalUp, ffnUpL, pLocalFfnIn, ffLenL, _embd, DType.Float16);
                                if (upBiasArr is not null)
                                {
                                    var s = new Span<float>(pLocalUp, ffLenL);
                                    TensorPrimitives.Add(s, upBiasArr, s);
                                }
                                switch (activation)
                                {
                                    case Llama4FfnActivation.Gelu: SimdKernels.GeluInPlace(pLocalUp, ffLenL); break;
                                    case Llama4FfnActivation.GeluQuick: SimdKernels.GeluQuickInPlace(pLocalUp, ffLenL); break;
                                    default: // Silu
                                        for (var i = 0; i < ffLenL; i++) pLocalUp[i] *= 1f / (1f + MathF.Exp(-pLocalUp[i]));
                                        break;
                                }

                                SimdKernels.MatVec(pLocalFfnOut, ffnDownL, pLocalUp, _embd, ffLenL, DType.Float16);
                                if (downBiasArr is not null)
                                {
                                    var s = new Span<float>(pLocalFfnOut, _embd);
                                    TensorPrimitives.Add(s, downBiasArr, s);
                                }

                                for (var c = 0; c < _embd; c++) hp[c] += pLocalFfnOut[c];
                            }
                        });
                    }
                }
            }
        }

        // post_ln (present in this checkpoint), applied once to all nPos tokens.
        if (_postLnW is not null)
        {
            fixed (float* pHidden = hidden, w = _postLnW, b = _postLnB)
            {
                for (var p = 0; p < nPos; p++)
                    SimdKernels.LayerNorm(pHidden + p * _embd, pHidden + p * _embd, w, b, _embd, _eps);
            }
        }

        // 3. Drop [CLS] (the last token) -- the merger only sees the n_patches grid tokens.
        // Pixel-shuffle merge: for output token (yg,xg) with 0<=dy,dx<NMerge, merged feature
        // layout is [dy][dx][channel] (dy slowest, channel fastest) sourced from input patch
        // (row=yg*NMerge+dy, col=xg*NMerge+dx) -- derived exactly from llama4.cpp's
        // reshape/permute/reshape sequence (Llama4VisionPixelShuffleMLP), see the class doc.
        var outSide = _gridSize / _nMerge;
        var nOut = outSide * outSide;
        var mergedWidth = _embd * _nMerge * _nMerge;
        var merged = new float[nOut * mergedWidth];
        for (var yg = 0; yg < outSide; yg++)
        {
            for (var xg = 0; xg < outSide; xg++)
            {
                var t = yg * outSide + xg;
                var dstBase = t * mergedWidth;
                for (var dy = 0; dy < _nMerge; dy++)
                {
                    for (var dx = 0; dx < _nMerge; dx++)
                    {
                        var srcPatch = (yg * _nMerge + dy) * _gridSize + (xg * _nMerge + dx);
                        var srcOff = srcPatch * _embd;
                        var dstOff = dstBase + dy * _embd * _nMerge + dx * _embd;
                        Array.Copy(hidden, srcOff, merged, dstOff, _embd);
                    }
                }
            }
        }
        if (s_debug) Console.Error.WriteLine("[LLAMA4-DBG] blocks + post_ln + pixel-shuffle done");

        // 4. Llama4VisionMLP2 (mlp.1 -> gelu -> mlp.2 -> gelu, both no bias) -> Llama4MultiModalProjector
        //    (mm.model.fc, no bias, no activation). Widths are whatever the real tensors declared
        //    (validated self-consistently at load time), not hardcoded here.
        var mlp1Out = (int)_m.MmModelMlp1Weight.Dimensions[1];
        var mlp2Out = (int)_m.MmModelMlp2Weight.Dimensions[1];
        var result = new float[nOut * _projDim];
        fixed (float* pMerged = merged, pResult = result)
        {
            var stage1 = new float[mlp1Out];
            var stage2 = new float[mlp2Out];
            fixed (float* pStage1 = stage1, pStage2 = stage2)
            {
                for (var t = 0; t < nOut; t++)
                {
                    SimdKernels.MatVec(pStage1, _mlp1W, pMerged + t * mergedWidth, mlp1Out, mergedWidth, DType.Float16);
                    SimdKernels.GeluInPlace(pStage1, mlp1Out);
                    SimdKernels.MatVec(pStage2, _mlp2W, pStage1, mlp2Out, mlp1Out, DType.Float16);
                    SimdKernels.GeluInPlace(pStage2, mlp2Out);
                    SimdKernels.MatVec(pResult + t * _projDim, _projW, pStage2, _projDim, mlp2Out, DType.Float16);
                }
            }
        }
        return result;
    }

    /// <summary>
    /// 2D-RoPE half, NORM/interleaved pairing: <c>(2i, 2i+1)</c> for <c>i in [0, quarterDim)</c>,
    /// rotated by <c>position * ropeTheta^(-2i/(2*quarterDim))</c>. This is the SHARED
    /// <c>clip_graph::build_rope_2d</c> convention llama4.cpp calls as-is -- confirmed from
    /// <c>clip.cpp</c>: its inner <c>ggml_rope_ext</c> calls use mode 0 (NORM/interleaved), NOT
    /// <c>GGML_ROPE_TYPE_NEOX</c>. This is genuinely different from
    /// <see cref="Gemma4VVisionEncoder"/>'s <c>ApplyRope2DHalf</c>, which implements NEOX
    /// split-half pairing <c>(j, j+quarterDim)</c> -- gemma4v.cpp hand-rolls its OWN 2D-rope
    /// function specifically because it needs NEOX ordering instead of this shared helper's
    /// default. Applied twice per head here too -- once to the leading half-of-head-dim with the
    /// patch's column as <paramref name="position"/>, once to the trailing half with its row --
    /// never to V.
    /// </summary>
    private static void ApplyRope2DHalfNorm(float* half, int quarterDim, int position, float ropeTheta)
    {
        var nDims = quarterDim * 2;
        for (var i = 0; i < quarterDim; i++)
        {
            var thetaScale = MathF.Pow(ropeTheta, -2f * i / nDims);
            var angle = position * thetaScale;
            var cos = MathF.Cos(angle);
            var sin = MathF.Sin(angle);
            var x0 = half[2 * i];
            var x1 = half[2 * i + 1];
            half[2 * i] = x0 * cos - x1 * sin;
            half[2 * i + 1] = x0 * sin + x1 * cos;
        }
    }
}
