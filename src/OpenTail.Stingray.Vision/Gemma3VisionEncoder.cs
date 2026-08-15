using System.Numerics.Tensors;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// The Gemma 3 (SigLIP, <c>clip.vision.projector_type=gemma3</c>) ViT encoder + token-reduction
/// pooler + projector. Turns a preprocessed image into LLM soft-token embeddings.
///
/// Ported stage-by-stage from the real reference (<c>examples/llama.cpp/llama.cpp</c>,
/// <c>tools/mtmd/models/siglip.cpp</c>'s <c>PROJECTOR_TYPE_GEMMA3</c> branch plus the shared
/// <c>clip_graph::build_vit</c>/<c>build_attn</c>/<c>build_ffn</c> helpers in
/// <c>tools/mtmd/clip.cpp</c>), verified against the real
/// <c>models/mmproj-gemma-3-4b-it-f16.gguf</c> -- see docs/03-gemma4-e4b-vision-plan.md's Gemma 3
/// addendum for the full derivation.
///
/// <para>Genuinely simpler than the Gemma 4 E4B <c>gemma4v</c> encoder
/// (<see cref="Gemma4VVisionEncoder"/>): no 2D RoPE, no per-head QK-norm, no V-norm, no per-block
/// QAT clamp, a single learned position table added once (not two stacked x/y tables looked up
/// per patch), a plain (not gated) FFN with the ordinary tanh-GELU already used elsewhere in this
/// codebase (<c>clip.use_gelu=true</c> in the real mmproj, unlike gemma4v's quick-GELU default),
/// and the standard <c>1/sqrt(head_dim)</c> attention scale (no override). Has its own new
/// wrinkles instead: every per-block linear (Q/K/V/O, FFN up/down) carries a real bias; weights
/// are F16, not BF16; the final projection tensor (<c>mm.input_projection.weight</c>) is stored
/// TRANSPOSED relative to this codebase's row-major convention (materialized into the
/// conventional layout once at construction, see <see cref="TransposeToRowMajor"/>); and the
/// input grid is much larger (896px / 14px patches = 4096 patches, 27 blocks, head_dim 72) --
/// attention here uses <see cref="TensorPrimitives.Dot{T}"/> for the O(n^2) score/weighted-sum
/// work rather than gemma4v's naive scalar loops (fine at 196 patches, far too slow at 4096).</para>
///
/// <para>Per-patch pipeline: patch conv (WITH bias) -&gt; add learned position table (direct
/// elementwise add, no lookup -- the table is already patch-ordered) -&gt; 27 blocks -&gt; real
/// post-layernorm (<c>v.post_ln</c>, unlike gemma4v which has none) -&gt; average-pool token
/// reduction (kernel=stride=<c>NMerge</c>=4, exact divide at 64/4=16, no dropped patches unlike
/// gemma4v's 14/3) -&gt; weighted RMSNorm (<c>mm.soft_emb_norm</c>) -&gt; transposed linear
/// projection. NO post-pool <c>sqrt(embd)</c> scale (gemma4v has one, this projector doesn't).</para>
///
/// <para>Per block (real pre-norm, NOT gemma4v's sandwich norm -- there is no attn_post_norm or
/// ffn_post_norm tensor at all for this projector): LayerNorm(ln1, with bias) -&gt; separate Q/K/V
/// projections (each with a bias-add, NO norm, NO rotation) -&gt; standard bidirectional attention,
/// scale=1/sqrt(head_dim) -&gt; output projection (with bias) -&gt; residual add -&gt;
/// LayerNorm(ln2, with bias) -&gt; FFN: <c>down(gelu(up(x)))</c> (both with bias-adds, no gate) -&gt;
/// residual add.</para>
///
/// <para><b>Unverified end-to-end</b>: every constant and mechanism here is sourced from the real
/// reference graph and the real mmproj's tensor/metadata inventory, not inferred, but there is
/// still no working oracle on this machine to run this encoder + its paired
/// <c>gemma-3-4b-it-Q4_K_M.gguf</c> text model end-to-end for numerical parity. Do not treat this
/// as parity-verified until that comparison exists.</para>
/// </summary>
public sealed unsafe class Gemma3VisionEncoder
{
    private readonly Gemma3VisionModel _m;
    private readonly int _embd, _heads, _headDim, _ffLen, _blockCount, _patch, _imageSize, _projDim;
    private readonly float _eps, _kqScale;
    private readonly int _gridSize;
    private readonly int _nMerge;

    private readonly byte* _patchEmbdW;    // F32 [embd, patchVec]
    private readonly float[] _patchEmbdBias;
    private readonly float[] _posTable;    // [nPatches * embd], direct elementwise add
    private readonly float[] _postLnW, _postLnB;
    private readonly float[] _mmSoftEmbNorm;
    private readonly float[] _mmProjT;     // materialized [projDim, embd] row-major (transposed from storage)

    private sealed class BlockData
    {
        public required float[] Ln1W, Ln1B, Ln2W, Ln2B;
        public required float[] AttnQBias, AttnKBias, AttnVBias, AttnOutBias, FfnUpBias, FfnDownBias;
        // Gemma3VisionModel's loader already binds FfnUp/FfnDown by FUNCTIONAL role (this
        // checkpoint's GGUF tensor NAMES "ffn_up"/"ffn_down" are swapped relative to their actual
        // role -- confirmed via bias length, not a storage-transpose issue), so these are used
        // directly here via the raw F16 pointer, same as attn_q/k/v/out -- no transpose needed.
        public required byte* AttnQ, AttnK, AttnV, AttnOut, FfnUp, FfnDown;
    }

    private readonly BlockData[] _blocks;

    /// <remarks>
    /// Borrows <paramref name="model"/>'s memory-mapped weights (caches raw pointers into them),
    /// so <paramref name="model"/> must outlive this encoder and must not be disposed before the
    /// last <see cref="Forward"/> call.
    /// </remarks>
    public Gemma3VisionEncoder(Gemma3VisionModel model)
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
        _eps = model.LayerNormEps;
        _gridSize = _imageSize / _patch;
        _nMerge = model.NMerge;
        _kqScale = 1f / MathF.Sqrt(_headDim);

        _patchEmbdW = model.Gguf.GetTensorDataPtr(model.PatchEmbdWeight);
        _patchEmbdBias = model.LoadFloats(model.PatchEmbdBias);
        _posTable = model.LoadFloats(model.PositionEmbedding);
        _postLnW = model.LoadFloats(model.PostLnWeight);
        _postLnB = model.LoadFloats(model.PostLnBias);
        _mmSoftEmbNorm = model.LoadFloats(model.MmSoftEmbNorm);
        _mmProjT = TransposeToRowMajor(model.LoadFloats(model.MmInputProjection), _projDim, _embd);

        _blocks = new BlockData[_blockCount];
        for (var i = 0; i < _blockCount; i++)
        {
            var b = model.Blocks[i];
            _blocks[i] = new BlockData
            {
                Ln1W = model.LoadFloats(b.Ln1W),
                Ln1B = model.LoadFloats(b.Ln1B),
                Ln2W = model.LoadFloats(b.Ln2W),
                Ln2B = model.LoadFloats(b.Ln2B),
                AttnQBias = model.LoadFloats(b.AttnQBias),
                AttnKBias = model.LoadFloats(b.AttnKBias),
                AttnVBias = model.LoadFloats(b.AttnVBias),
                AttnOutBias = model.LoadFloats(b.AttnOutBias),
                FfnUpBias = model.LoadFloats(b.FfnUpBias),
                FfnDownBias = model.LoadFloats(b.FfnDownBias),
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

    /// <summary>
    /// Run the encoder. <paramref name="chw"/> is the preprocessor's output: planar CHW, already
    /// resized to <c>ImageSize x ImageSize</c> and channel-normalized to this mmproj's declared
    /// mean=[0.5,0.5,0.5]/std=[0.5,0.5,0.5] (i.e. already in [-1,1] -- unlike gemma4v, this
    /// encoder applies no additional range fix). Returns <see cref="TokenCount"/> x
    /// <c>ProjectionDim</c> soft-token embeddings, row-major.
    /// </summary>
    // TEMP diagnostic (STINGRAY_GEMMA3_DEBUG=1) for isolating a stall while running this encoder.
    private static readonly bool s_debug = Environment.GetEnvironmentVariable("STINGRAY_GEMMA3_DEBUG") == "1";

    public float[] Forward(ReadOnlySpan<float> chw)
    {
        _m.EnsureNotDisposed();
        var plane = _imageSize * _imageSize;
        if (chw.Length < 3L * plane)
            throw new ArgumentException($"chw length ({chw.Length}) is too small for a 3x{_imageSize}x{_imageSize} image.");

        var nPatches = _gridSize * _gridSize;
        var patchVec = _patch * _patch * 3;
        if (s_debug) Console.Error.WriteLine($"[GEMMA3-DBG] Forward start: nPatches={nPatches} patchVec={patchVec}");

        // 1. im2col (no range fix -- preprocessor already emits [-1,1] via mean=std=0.5) + patch
        //    embed (F32 matvec WITH bias) + learned position table (direct add, already patch-ordered).
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

        var hidden = new float[nPatches * _embd];
        fixed (float* pPatches = patches, pHidden = hidden, pPos = _posTable)
        {
            for (var p = 0; p < nPatches; p++)
                SimdKernels.MatVec(pHidden + p * _embd, _patchEmbdW, pPatches + p * patchVec, _embd, patchVec, DType.Float32);
            for (var p = 0; p < nPatches; p++)
            {
                var row = new Span<float>(pHidden + p * _embd, _embd);
                TensorPrimitives.Add(row, _patchEmbdBias, row);
                TensorPrimitives.Add(row, new ReadOnlySpan<float>(pPos + p * _embd, _embd), row);
            }
        }
        if (s_debug) Console.Error.WriteLine("[GEMMA3-DBG] patch embed + position done");

        // 2. transformer blocks (real pre-norm, no post-attn/post-ffn norm, no rotation, no
        // per-head norm -- see the class doc comment for the full per-block order). Every
        // per-patch stage below is parallelized (Parallel.For across patches or heads, each task
        // with its own local scratch), so there is no single shared per-token scratch buffer at
        // this level -- only the arrays every stage reads/writes disjoint slices of.
        var q = new float[nPatches * _embd];
        var k = new float[nPatches * _embd];
        var v = new float[nPatches * _embd];
        var attnConcat = new float[nPatches * _embd];

        fixed (float* pHidden = hidden, pQ = q, pK = k, pV = v, pAttnConcat = attnConcat)
        {
            for (var layer = 0; layer < _blockCount; layer++)
            {
                if (s_debug) Console.Error.WriteLine($"[GEMMA3-DBG] block {layer}/{_blockCount} start");
                var blk = _blocks[layer];
                fixed (float* ln1w = blk.Ln1W, ln1b = blk.Ln1B, ln2w = blk.Ln2W, ln2b = blk.Ln2B,
                       qBias = blk.AttnQBias, kBias = blk.AttnKBias, vBias = blk.AttnVBias,
                       oBias = blk.AttnOutBias, upBias = blk.FfnUpBias, downBias = blk.FfnDownBias)
                {
                    // ln1 -> Q/K/V (each with a bias-add, no norm, no rotation). Parallelized
                    // across patches -- measured as the dominant cost (with output-proj/FFN
                    // below), ~89% of a block's wall-clock time, far more than attention itself.
                    // 4096 independent patches (each reads/writes its own disjoint slice) is a
                    // much better parallelization target than the 16 heads above, since it isn't
                    // capped by this machine's 12 logical processors the way 16-way head
                    // parallelism nearly is. Same nint round-trip as the attention section
                    // (pointers can't be captured by a lambda); each task gets its own `normed`
                    // scratch instead of sharing pNormed, which would race.
                    {
                        nint hiddenBase = (nint)pHidden, qBase2 = (nint)pQ, kBase2 = (nint)pK, vBase2 = (nint)pV;
                        nint ln1wBase = (nint)ln1w, ln1bBase = (nint)ln1b;
                        var attnQL = blk.AttnQ; var attnKL = blk.AttnK; var attnVL = blk.AttnV;
                        var qBiasArr = blk.AttnQBias; var kBiasArr = blk.AttnKBias; var vBiasArr = blk.AttnVBias;
                        Parallel.For(0, nPatches, p =>
                        {
                            var hiddenL = (float*)hiddenBase;
                            var qL = (float*)qBase2;
                            var kL = (float*)kBase2;
                            var vL = (float*)vBase2;
                            var ln1wL = (float*)ln1wBase;
                            var ln1bL = (float*)ln1bBase;
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
                                var qSpan = new Span<float>(qp, _embd);
                                var kSpan = new Span<float>(kp, _embd);
                                var vSpan = new Span<float>(vp, _embd);
                                TensorPrimitives.Add(qSpan, qBiasArr, qSpan);
                                TensorPrimitives.Add(kSpan, kBiasArr, kSpan);
                                TensorPrimitives.Add(vSpan, vBiasArr, vSpan);
                            }
                        });
                    }

                    // Bidirectional multi-head attention, standard scale = 1/sqrt(head_dim).
                    // Parallelized across heads (fully independent: each head reads/writes a
                    // disjoint headDim-wide slice of Q/K/V/attnConcat) -- at 4096 patches this is
                    // ~1 trillion MACs per encoder call, impractically slow single-threaded
                    // (~35 min measured). Pointers can't be captured by a lambda even in unsafe
                    // code, so they're round-tripped through nint; each task gets its own
                    // scores/temp scratch instead of sharing pScores/pTemp, which would race.
                    if (s_debug) Console.Error.WriteLine($"[GEMMA3-DBG] block {layer} QKV done, starting attention");
                    nint qBase = (nint)pQ, kBase = (nint)pK, vBase = (nint)pV, concatBase = (nint)pAttnConcat;
                    int headDimL = _headDim, embdL = _embd, npL = nPatches, layerL = layer;
                    float kqScaleL = _kqScale;
                    Parallel.For(0, _heads, h =>
                    {
                        if (s_debug && h % 4 == 0) Console.Error.WriteLine($"[GEMMA3-DBG] block {layerL} attention head {h}/{_heads}");
                        var qL = (float*)qBase;
                        var kL = (float*)kBase;
                        var vL = (float*)vBase;
                        var concatL = (float*)concatBase;
                        var localScores = new float[npL];
                        var localTemp = new float[headDimL];
                        fixed (float* pLocalScores = localScores, pLocalTemp = localTemp)
                        {
                            var off = h * headDimL;
                            for (var i = 0; i < npL; i++)
                            {
                                var qi = new ReadOnlySpan<float>(qL + i * embdL + off, headDimL);
                                for (var j = 0; j < npL; j++)
                                {
                                    var kj = new ReadOnlySpan<float>(kL + j * embdL + off, headDimL);
                                    pLocalScores[j] = TensorPrimitives.Dot(qi, kj) * kqScaleL;
                                }
                                SimdKernels.SoftmaxInPlace(pLocalScores, npL);

                                var outSpan = new Span<float>(concatL + i * embdL + off, headDimL);
                                outSpan.Clear();
                                var tempSpan = new Span<float>(pLocalTemp, headDimL);
                                for (var j = 0; j < npL; j++)
                                {
                                    var vj = new ReadOnlySpan<float>(vL + j * embdL + off, headDimL);
                                    TensorPrimitives.Multiply(vj, pLocalScores[j], tempSpan);
                                    TensorPrimitives.Add(outSpan, tempSpan, outSpan);
                                }
                            }
                        }
                    });

                    // Output projection (with bias) -> residual (NO post-attn norm for this
                    // projector). Parallelized across patches, same reasoning as the QKV loop
                    // above -- each patch's output row and residual-add are fully independent.
                    {
                        nint concatBase2 = (nint)pAttnConcat, hiddenBase2 = (nint)pHidden;
                        var attnOutL = blk.AttnOut;
                        var oBiasArr = blk.AttnOutBias;
                        Parallel.For(0, nPatches, p =>
                        {
                            var concatL2 = (float*)concatBase2;
                            var hiddenL2 = (float*)hiddenBase2;
                            var localProj = new float[_embd];
                            fixed (float* pLocalProj = localProj)
                            {
                                SimdKernels.MatVec(pLocalProj, attnOutL, concatL2 + p * _embd, _embd, _embd, DType.Float16);
                                var projSpan = new Span<float>(pLocalProj, _embd);
                                TensorPrimitives.Add(projSpan, oBiasArr, projSpan);
                                var hp = hiddenL2 + p * _embd;
                                for (var c = 0; c < _embd; c++) hp[c] += pLocalProj[c];
                            }
                        });
                    }

                    // ln2 -> plain FFN (down(gelu(up(x))), both with bias-adds, no gate) ->
                    // residual. Parallelized across patches -- this is the single largest cost in
                    // the block (~40B MACs/block from the 1152<->4304 projections alone, more than
                    // attention's own ~38.7B), so it needed the same treatment as QKV above.
                    {
                        nint hiddenBase3 = (nint)pHidden, ln2wBase = (nint)ln2w, ln2bBase = (nint)ln2b;
                        var ffnUpL = blk.FfnUp; var ffnDownL = blk.FfnDown;
                        var upBiasArr = blk.FfnUpBias; var downBiasArr = blk.FfnDownBias;
                        int ffLenL = _ffLen;
                        Parallel.For(0, nPatches, p =>
                        {
                            var hiddenL3 = (float*)hiddenBase3;
                            var ln2wL = (float*)ln2wBase;
                            var ln2bL = (float*)ln2bBase;
                            var localFfnIn = new float[_embd];
                            var localUp = new float[ffLenL];
                            var localFfnOut = new float[_embd];
                            fixed (float* pLocalFfnIn = localFfnIn, pLocalUp = localUp, pLocalFfnOut = localFfnOut)
                            {
                                var hp = hiddenL3 + p * _embd;
                                SimdKernels.LayerNorm(pLocalFfnIn, hp, ln2wL, ln2bL, _embd, _eps);

                                SimdKernels.MatVec(pLocalUp, ffnUpL, pLocalFfnIn, ffLenL, _embd, DType.Float16);
                                var upSpan = new Span<float>(pLocalUp, ffLenL);
                                TensorPrimitives.Add(upSpan, upBiasArr, upSpan);
                                SimdKernels.GeluInPlace(pLocalUp, ffLenL);

                                SimdKernels.MatVec(pLocalFfnOut, ffnDownL, pLocalUp, _embd, ffLenL, DType.Float16);
                                var downSpan = new Span<float>(pLocalFfnOut, _embd);
                                TensorPrimitives.Add(downSpan, downBiasArr, downSpan);

                                for (var c = 0; c < _embd; c++) hp[c] += pLocalFfnOut[c];
                            }
                        });
                    }
                }
            }
        }

        // 3. real post-layernorm.
        fixed (float* pHidden = hidden, w = _postLnW, b = _postLnB)
        {
            for (var p = 0; p < nPatches; p++)
                SimdKernels.LayerNorm(pHidden + p * _embd, pHidden + p * _embd, w, b, _embd, _eps);
        }

        // 4. average-pool token reduction (kernel = stride = NMerge, exact divide -- no dropped
        // patches here, unlike gemma4v's non-divisible 14/3) -> weighted RMSNorm
        // (mm.soft_emb_norm) -> projection via the pre-transposed weight. NO post-pool sqrt(embd)
        // scale for this projector.
        var outSide = _gridSize / _nMerge;
        var nOut = outSide * outSide;
        var pooled = new float[nOut * _embd];
        var kernelArea = (float)(_nMerge * _nMerge);
        for (var oy = 0; oy < outSide; oy++)
        {
            for (var ox = 0; ox < outSide; ox++)
            {
                var dst = (oy * outSide + ox) * _embd;
                for (var ky = 0; ky < _nMerge; ky++)
                {
                    for (var kx = 0; kx < _nMerge; kx++)
                    {
                        var srcPatch = (oy * _nMerge + ky) * _gridSize + (ox * _nMerge + kx);
                        var src = srcPatch * _embd;
                        for (var c = 0; c < _embd; c++)
                            pooled[dst + c] += hidden[src + c];
                    }
                }
                for (var c = 0; c < _embd; c++)
                    pooled[dst + c] /= kernelArea;
            }
        }

        var result = new float[nOut * _projDim];
        fixed (float* pPooled = pooled, pResult = result, normW = _mmSoftEmbNorm, pProjT = _mmProjT)
        {
            for (var t = 0; t < nOut; t++)
                SimdKernels.RmsNorm(pPooled + t * _embd, pPooled + t * _embd, normW, _embd, _eps);
            for (var t = 0; t < nOut; t++)
                SimdKernels.MatVec(pResult + t * _projDim, (byte*)pProjT, pPooled + t * _embd, _projDim, _embd, DType.Float32);
        }
        return result;
    }

    /// <summary>
    /// <c>mm.input_projection.weight</c> is stored as [projDim, embd] with projDim contiguous --
    /// the OPPOSITE of this codebase's row-major [outputRow, inputCol] convention every other
    /// weight tensor follows. siglip.cpp explicitly transposes it before use
    /// (<c>ggml_cont(ggml_transpose(...))</c>); this does the same at construction time, once,
    /// into a plain row-major float array so the encoder can use the ordinary
    /// <see cref="SimdKernels.MatVec"/> path afterward instead of a bespoke strided dot product.
    /// </summary>
    private static float[] TransposeToRowMajor(float[] storedProjDimFast, int rows, int cols)
    {
        // storedProjDimFast is [cols(=rows here, embd), rows(=cols here, projDim)] flattened with
        // the FIRST index (embd) as the slow dimension and projDim fast -- i.e.
        // storedProjDimFast[embdIdx * cols + projIdx]. We want row-major [rows=projDim, cols=embd]:
        // result[projIdx * embd + embdIdx].
        var result = new float[rows * cols];
        for (var embdIdx = 0; embdIdx < cols; embdIdx++)
        {
            var srcOff = embdIdx * rows;
            for (var projIdx = 0; projIdx < rows; projIdx++)
                result[projIdx * cols + embdIdx] = storedProjDimFast[srcOff + projIdx];
        }
        return result;
    }
}
