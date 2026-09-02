namespace OpenTail.Stingray.Engine;

// ============================================================================================
// ALPHA / UNTESTED -- see DeepSeek4Alpha.cs's file header for the overall status/scope note;
// everything there applies here too. This file is the persistent compressed-KV state cache that
// CSA/HCA/LID (lightning indexer) attention need -- flagged in
// docs/058-deepseek-full-lineage-implementation-plan.md as the one mechanism with no existing
// analog anywhere in this codebase.
//
// IMPORTANT SCOPE NOTE, read before wiring this in: the real reference
// (examples/llama.cpp/llama.cpp/src/llama-kv-cache-dsv4.h/.cpp) is NOT primarily a compression
// algorithm -- most of its ~400 lines are rollback/snapshot bookkeeping (state_restore /
// state_snapshot / state_persist, a "reserve_plan" mechanism, multi-stream sequence-copy support
// via stream_copy_info) built for llama.cpp's speculative-decode-with-rewind and multi-sequence
// batched-inference model. This class deliberately does NOT port that machinery. It implements
// only the minimal single-sequence, no-rewind version: persist a compressed block once computed,
// read it back later. This is almost certainly enough for a first straight-line greedy-decode
// verification pass (the same kind doc 032's investigation used for deepseek2 -- one prompt,
// temp 0, no speculative decoding, no batched multi-sequence serving) but will NOT support this
// engine's speculative decoding (MtpDecoder/DSparkDecoder) or ContinuousBatchingEngine's
// multi-sequence serving without real extension work. Flagged here explicitly so nobody wires
// this into either of those paths assuming rollback support exists -- it does not.
// ============================================================================================

/// <summary>
/// ALPHA/UNTESTED. One layer's persistent compressed-KV-block state for a single decode
/// sequence -- the storage CSA/HCA (and separately, the lightning indexer, which keeps its own
/// instance of this same structure) need to remember previously-compressed KV blocks across
/// decode positions. Simplified relative to the C++ reference's <c>llama_dsv4_comp_state</c>
/// (see this file's header) to a plain growable ring of compressed rows, addressed by absolute
/// block index -- no rollback/snapshot support.
/// </summary>
public sealed class DeepSeek4CompressedLayerState
{
    private readonly int _headDim;
    private readonly List<float[]> _kvBlocks = [];
    private readonly List<float[]> _scoreBlocks = [];

    public DeepSeek4CompressedLayerState(int headDim)
    {
        _headDim = headDim;
    }

    /// <summary>Number of compressed blocks persisted so far for this layer.</summary>
    public int BlockCount => _kvBlocks.Count;

    /// <summary>
    /// Appends a newly-compressed [headDim] KV block and its paired score row, becoming block
    /// index <see cref="BlockCount"/> - 1 after the call. Corresponds to the reference's
    /// <c>state_persist_*_idxs</c>-driven commit (deepseek4.cpp:1062-1071 for CSA,
    /// 1131-1140 for LID, 1212-1221 for HCA) with rollback support dropped per this file's
    /// scope note.
    /// </summary>
    public void Persist(ReadOnlySpan<float> kv, ReadOnlySpan<float> score)
    {
        if (kv.Length != _headDim || score.Length != _headDim)
        {
            throw new ArgumentException($"Expected length {_headDim}, got kv={kv.Length} score={score.Length}");
        }
        _kvBlocks.Add(kv.ToArray());
        _scoreBlocks.Add(score.ToArray());
    }

    /// <summary>
    /// Reads back the [headDim] KV and score rows for a previously-persisted block index.
    /// Corresponds to the reference's <c>get_kv</c>/<c>get_score</c> (llama-kv-cache-dsv4.h:44-45)
    /// plus the state-restore gather (deepseek4.cpp:217-238) collapsed into direct indexed access,
    /// since this simplified version has no rollback planes to restore from.
    /// </summary>
    public (ReadOnlyMemory<float> Kv, ReadOnlyMemory<float> Score) GetBlock(int blockIndex)
    {
        if ((uint)blockIndex >= (uint)_kvBlocks.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(blockIndex));
        }
        return (_kvBlocks[blockIndex], _scoreBlocks[blockIndex]);
    }

    /// <summary>
    /// Copies the KV/score rows for a contiguous range of blocks
    /// [<paramref name="startBlock"/>, <paramref name="startBlock"/> + <paramref name="count"/>)
    /// into caller-provided [count, headDim] buffers, for feeding directly into
    /// <see cref="DeepSeek4Graph.HcaCompressBlock"/>/<see cref="DeepSeek4Graph.CsaCompressBlock"/>
    /// as the "already-gathered rows" those methods expect (see their doc comments).
    /// </summary>
    public void CopyRange(int startBlock, int count, Span<float> kvOut, Span<float> scoreOut)
    {
        for (int i = 0; i < count; i++)
        {
            var (kv, score) = GetBlock(startBlock + i);
            kv.Span.CopyTo(kvOut.Slice(i * _headDim, _headDim));
            score.Span.CopyTo(scoreOut.Slice(i * _headDim, _headDim));
        }
    }

    /// <summary>Discards all state for this layer (sequence reset / new conversation).</summary>
    public void Clear()
    {
        _kvBlocks.Clear();
        _scoreBlocks.Clear();
    }
}

/// <summary>
/// ALPHA/UNTESTED. Per-layer collection of <see cref="DeepSeek4CompressedLayerState"/>, one for
/// each of the three independent compressed-state streams DeepSeek-V4 needs: CSA (ratio-4
/// layers' compressed KV), HCA (ratio-128 layers' compressed KV), and LID (the lightning
/// indexer's own separately-compressed key stream, used regardless of which ratio a layer has,
/// per deepseek4.cpp:1073-1140 building lid_state_kv/lid_state_score alongside csa_state_kv/
/// csa_state_score in the same CSA-ratio branch). Corresponds to
/// <c>llama_kv_cache_dsv4</c>'s three <c>get_csa_state()</c>/<c>get_hca_state()</c>/
/// <c>get_lid_state()</c> accessors (llama-kv-cache-dsv4.h:147-149).
/// </summary>
public sealed class DeepSeek4CompressedState
{
    private readonly DeepSeek4CompressedLayerState?[] _csa;
    private readonly DeepSeek4CompressedLayerState?[] _hca;
    private readonly DeepSeek4CompressedLayerState?[] _lid;

    /// <param name="numLayers">Total trunk layer count (excludes MTP tail layers).</param>
    /// <param name="headDim">Attention head width, used to size CSA/HCA block storage.</param>
    /// <param name="indexerHeadDim">Lightning-indexer per-head width, used to size LID block storage.</param>
    /// <param name="compressRatios">Per-layer compression ratio (0/4/128), from <see cref="DeepSeek4Hyperparams.CompressRatios"/>.</param>
    public DeepSeek4CompressedState(int numLayers, int headDim, int indexerHeadDim, IReadOnlyList<int> compressRatios)
    {
        _csa = new DeepSeek4CompressedLayerState?[numLayers];
        _hca = new DeepSeek4CompressedLayerState?[numLayers];
        _lid = new DeepSeek4CompressedLayerState?[numLayers];

        for (int il = 0; il < numLayers; il++)
        {
            int ratio = il < compressRatios.Count ? compressRatios[il] : 0;
            if (ratio == 0) continue;

            if (ratio == 4)
            {
                _csa[il] = new DeepSeek4CompressedLayerState(headDim);
                _lid[il] = new DeepSeek4CompressedLayerState(indexerHeadDim);
            }
            else if (ratio == 128)
            {
                _hca[il] = new DeepSeek4CompressedLayerState(headDim);
            }
            // Any other ratio is invalid per the reference (deepseek4.cpp:148-150) -- the caller
            // that built compressRatios is responsible for having already validated this; this
            // constructor silently treats an unrecognized ratio as "no compression" rather than
            // throwing, since hyperparameter validation belongs at the loader, not here.
        }
    }

    public DeepSeek4CompressedLayerState? Csa(int layer) => _csa[layer];
    public DeepSeek4CompressedLayerState? Hca(int layer) => _hca[layer];
    public DeepSeek4CompressedLayerState? Lid(int layer) => _lid[layer];

    /// <summary>Resets every layer's compressed state (new sequence / conversation).</summary>
    public void Clear()
    {
        foreach (var s in _csa) s?.Clear();
        foreach (var s in _hca) s?.Clear();
        foreach (var s in _lid) s?.Clear();
    }
}
