using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Mlx;

/// <summary>
/// <see cref="IForwardPass"/> adapter over <see cref="MlxDenseModel"/> -- the engine/planner
/// integration point. Most of this class's control flow (cache reset, rewind-length validation,
/// startPos validation) is plain C# with no native call in it, so it is fully tested for real;
/// only the calls that actually reach <see cref="MlxDenseModel.Prefill"/>/<see cref="MlxDenseModel.Decode"/>
/// can only be checked as "reaches the first native call and fails predictably" (no Mac here). See
/// docs/4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md.
/// </summary>
public sealed class MlxForwardPass : IForwardPass
{
    private readonly MlxDenseModel _model;
    private readonly MlxKvCache[] _caches;
    private readonly int _maxSeqLen;
    private int _length;
    private float[] _lastLogits = [];

    public MlxForwardPass(MlxDenseModel model, int maxSeqLen)
    {
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
        _model = model;
        _maxSeqLen = maxSeqLen;
        _caches = MakeCaches();
    }

    private MlxKvCache[] MakeCaches()
    {
        var caches = new MlxKvCache[_model.Layers.Length];
        for (int i = 0; i < caches.Length; i++)
            caches[i] = new MlxKvCache(_model.Layers[i].NumKvHeads, _maxSeqLen, _model.Layers[i].HeadDim);
        return caches;
    }

    public int VocabSize => _model.VocabSize;
    public int MaxSeqLen => _maxSeqLen;

    /// <summary>Caches are plain host arrays (no native resource to free); see <see cref="MlxKvCache"/>'s own type doc comment for why.</summary>
    public void ResetCache()
    {
        for (int i = 0; i < _caches.Length; i++)
            _caches[i] = new MlxKvCache(_model.Layers[i].NumKvHeads, _maxSeqLen, _model.Layers[i].HeadDim);
        _length = 0;
    }

    /// <summary>
    /// Per <see cref="IForwardPass.Prefill"/>'s own contract, <paramref name="tokens"/> at
    /// <paramref name="startPos"/> &gt; 0 is the NEW suffix beyond what the cache already holds, not
    /// a repeat of it -- so <paramref name="startPos"/> must equal the cache's current length exactly
    /// (0 for a fresh prefill, or the length left by an earlier <see cref="Prefill"/>/<see cref="Forward"/>
    /// call for a continuation), and this always does real work for a non-empty token list. The actual
    /// continuation support -- an offset causal mask instead of the aligned built-in one -- lives in
    /// <see cref="MlxDenseLayer.Forward"/>'s <c>cache</c> parameter and
    /// <see cref="MlxOps.Attention"/>'s <c>queryOffset</c>; this method is just the thin
    /// <c>IForwardPass</c>-shaped entry point to it.
    /// </summary>
    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        if (startPos != _length)
            throw new NotSupportedException(
                $"Prefill(startPos={startPos}) does not match the cache's current length ({_length}) -- " +
                "this port only supports prefilling a new suffix starting exactly where the cache left off.");

        int[] tokenArray = [.. tokens];
        if (tokenArray.Length == 0)
            return _lastLogits.AsSpan(_lastLogits.Length - VocabSize);

        _lastLogits = _model.Prefill(tokenArray, _caches);
        _length += tokenArray.Length;
        return _lastLogits.AsSpan(_lastLogits.Length - VocabSize);
    }

    public ReadOnlySpan<float> Forward(int token, int position)
    {
        if (position != _length)
            throw new NotSupportedException(
                $"Forward(position={position}) does not match the cache's current length ({_length}) -- " +
                "this port only supports advancing one token at a time from exactly where the cache left off.");
        if (_length >= _maxSeqLen)
            throw new InvalidOperationException($"Cache is full ({_maxSeqLen} positions); cannot advance further.");

        _lastLogits = _model.Decode(token, position, _caches);
        _length++;
        return _lastLogits;
    }

    /// <summary>
    /// Only <c>length == 0</c> (full reset) or <c>length == current length</c> (no-op) are accepted,
    /// per <see cref="IForwardPass.TruncateTo"/>'s documented contract for a
    /// <see cref="IForwardPass.SupportsPartialRewind"/> == <c>false</c> backend (the default this
    /// class leaves unoverridden) -- arbitrary intermediate rewind would need to physically shrink
    /// <see cref="MlxKvCache"/>'s filled length, not built here.
    /// </summary>
    public void TruncateTo(int length)
    {
        if (length == _length) return;
        if (length == 0) { ResetCache(); return; }
        throw new NotSupportedException(
            $"TruncateTo({length}): only 0 (full reset) or the current length ({_length}, a no-op) are " +
            "supported -- SupportsPartialRewind is false for this backend.");
    }

    public void Dispose() { }
}
