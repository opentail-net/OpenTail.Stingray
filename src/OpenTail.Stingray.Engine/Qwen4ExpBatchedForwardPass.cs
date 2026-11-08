namespace OpenTail.Stingray.Engine;

/// <summary>
/// Adapter that lets <see cref="ContinuousBatchingEngine"/> drive a qwen4exp model (Phase 2 of the Qwen 3.8 plan).
/// Each admitted request gets its own <see cref="Qwen4ExpSequenceState"/>; the model weights are shared. The first
/// version runs the requests of a batch one after another through the per-sequence forward pass, which is the
/// sequential path the plan asks for. Batched GEMM is not used, so no numerics change relative to single-sequence runs.
/// </summary>
public sealed class Qwen4ExpBatchedForwardPass : IBatchedForwardPass, IPrefixCacheableBatchedForwardPass
{
    private readonly Qwen4ExpForwardPass _pass;
    private readonly int _kvBytesPerToken;
    private readonly int _maxSeqLen;

    public Qwen4ExpBatchedForwardPass(Qwen4ExpForwardPass pass, Qwen4ExpHyperparams hp, int numLayers)
    {
        _pass = pass ?? throw new ArgumentNullException(nameof(pass));
        ArgumentNullException.ThrowIfNull(hp);
        _maxSeqLen = hp.ContextLength;

        // Growing per-token state: K and V rows of each attention (QSA) layer, plus the indexer keys.
        int attentionLayers = 0;
        for (int l = 0; l < numLayers; l++)
            if (!hp.IsRecurrentLayer(l)) attentionLayers++;
        long perToken = (long)attentionLayers * 2 * hp.NumHeadsKv * hp.HeadDim * sizeof(float);
        _kvBytesPerToken = (int)Math.Min(int.MaxValue, perToken);
    }

    public ISequenceKvCache CreateCache() => new Cache(_pass.CreateSequenceState());

    public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
    {
        var state = ToState(cache);
        ReadOnlySpan<float> last = default;
        for (int i = 0; i < tokens.Count; i++)
            last = _pass.Forward(tokens[i], startPos + i, state);
        return last;
    }

    public float[]?[] PrefillPackedMulti(
        ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits)
    {
        var result = new float[]?[chunks.Length];
        for (int s = 0; s < chunks.Length; s++)
        {
            var state = ToState(caches[s]);
            var span = chunks[s].Span;
            ReadOnlySpan<float> last = default;
            for (int i = 0; i < span.Length; i++)
                last = _pass.Forward(span[i], startPos[s] + i, state);
            if (wantLogits[s]) result[s] = last.ToArray();
        }
        return result;
    }

    public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
    {
        var result = new float[tokens.Length][];
        for (int s = 0; s < tokens.Length; s++)
            result[s] = _pass.Forward(tokens[s], positions[s], ToState(caches[s])).ToArray();
        return result;
    }

    public bool SnapKvEnabled => false;

    public long KvBytesPerToken => _kvBytesPerToken;

    public int MaxSeqLen => _maxSeqLen;

    public bool PrefillDequantCacheActive => false;

    /// <summary>Recurrent state has no pages: a prefix is any token count, captured as a deep copy.</summary>
    public int PrefixCacheBlockSize => 1;

    /// <summary>Recurrent state cannot be rolled back, so the engine must capture at the exact boundary.</summary>
    public bool RequiresExactPrefixBoundary => true;

    /// <summary>
    /// Deep copy of the sequence at exactly <paramref name="prefixLength"/> tokens. The recurrent GDN state cannot be
    /// rolled back, so capture is only valid when the sequence stands at that position.
    /// </summary>
    public ISequenceKvCache CapturePrefix(ISequenceKvCache cache, int prefixLength)
    {
        var state = ToState(cache);
        if (state.TokensProcessed != prefixLength)
            throw new InvalidOperationException(
                $"qwen4exp prefix capture at {prefixLength} but the sequence stands at {state.TokensProcessed}.");
        return new Cache(state.Clone());
    }

    /// <summary>A writable sequence starting from a copy of the retained prefix.</summary>
    public ISequenceKvCache ForkPrefix(ISequenceKvCache prefix) => new Cache(ToState(prefix).Clone());

    public void BindToCurrentThread() { }

    private static Qwen4ExpSequenceState ToState(ISequenceKvCache cache) =>
        cache is Cache c ? c.State : throw new ArgumentException("Cache was not created by this Qwen4Exp forward pass.", nameof(cache));

    private sealed class Cache(Qwen4ExpSequenceState state) : ISequenceKvCache
    {
        public Qwen4ExpSequenceState State { get; } = state;

        public void Dispose() => State.Dispose();
    }
}
