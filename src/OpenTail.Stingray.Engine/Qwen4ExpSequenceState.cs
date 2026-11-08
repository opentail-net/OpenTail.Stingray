namespace OpenTail.Stingray.Engine;

/// <summary>
/// Mutable, conversation-specific state of one qwen4exp sequence (Phase 1 of the Qwen 3.8 plan).
/// Model weights and model-wide configuration are not stored here. Execution scratch (the per-token
/// residual stream, matmul buffers) stays in <see cref="Qwen4ExpForwardPass"/> because it is rebuilt on every call.
/// Clone, Restore and Dispose are deep: nothing is shared between two states except immutable model tables.
/// </summary>
public sealed class Qwen4ExpSequenceState : IDisposable
{
    private bool _disposed;

    /// <summary>GDN depthwise conv history per layer (empty for attention layers).</summary>
    public float[][] GdnConvState { get; private set; }

    /// <summary>GDN recurrent matrix state per layer, FP32 (empty for attention layers).</summary>
    public float[][] GdnState { get; private set; }

    /// <summary>PLE depthwise conv history, [histSlots * hcDim].</summary>
    public float[] PleConvHistory { get; private set; }

    /// <summary>Number of tokens pushed through the PLE path so far.</summary>
    public int PleTokensSeen { get; set; }

    /// <summary>Absolute position after the last forwarded token (0 for a fresh sequence).</summary>
    public int TokensProcessed { get; set; }

    /// <summary>PLE n-gram token history (null when the model has no PLE table).</summary>
    public Qwen4ExpPleHasher? PleHasher { get; private set; }

    /// <summary>QSA attention K/V and indexer history, one list per layer.</summary>
    public List<float[]>[] QsaKeyCache { get; private set; }
    public List<float[]>[] QsaValCache { get; private set; }
    public List<float[]>[] QsaRawIndexKCache { get; private set; }
    public List<float[]>[] QsaPooledKeys { get; private set; }

    public Qwen4ExpSequenceState(
        int numLayers,
        Func<int, bool> isRecurrent,
        int gdnConvLength,
        int gdnStateLength,
        int pleHistoryLength,
        Qwen4ExpPleHasher? pleHasher)
    {
        GdnConvState = new float[numLayers][];
        GdnState = new float[numLayers][];
        for (int l = 0; l < numLayers; l++)
        {
            if (isRecurrent(l))
            {
                GdnConvState[l] = new float[Math.Max(1, gdnConvLength)];
                GdnState[l] = new float[Math.Max(1, gdnStateLength)];
            }
            else
            {
                GdnConvState[l] = Array.Empty<float>();
                GdnState[l] = Array.Empty<float>();
            }
        }

        PleConvHistory = new float[pleHistoryLength];
        PleHasher = pleHasher;

        QsaKeyCache = EmptyLists(numLayers);
        QsaValCache = EmptyLists(numLayers);
        QsaRawIndexKCache = EmptyLists(numLayers);
        QsaPooledKeys = EmptyLists(numLayers);
    }

    private Qwen4ExpSequenceState(Qwen4ExpSequenceState source)
    {
        GdnConvState = CopyArrays(source.GdnConvState);
        GdnState = CopyArrays(source.GdnState);
        PleConvHistory = (float[])source.PleConvHistory.Clone();
        PleTokensSeen = source.PleTokensSeen;
        TokensProcessed = source.TokensProcessed;
        PleHasher = source.PleHasher?.Clone();
        QsaKeyCache = CopyLists(source.QsaKeyCache);
        QsaValCache = CopyLists(source.QsaValCache);
        QsaRawIndexKCache = CopyLists(source.QsaRawIndexKCache);
        QsaPooledKeys = CopyLists(source.QsaPooledKeys);
    }

    /// <summary>
    /// Bytes a sequence holds before its first token, with the same sizing rule as the constructor: the GDN conv and
    /// recurrent state of each recurrent layer, and the PLE conv history. The QSA caches and the PLE hasher grow with
    /// tokens and are not included. Used to charge a retained prefix's fixed state against its cache budget.
    /// </summary>
    public static long FixedFloatBytes(
        int numLayers, Func<int, bool> isRecurrent, int gdnConvLength, int gdnStateLength, int pleHistoryLength)
    {
        long floats = pleHistoryLength;
        for (int l = 0; l < numLayers; l++)
            if (isRecurrent(l)) floats += Math.Max(1, gdnConvLength) + Math.Max(1, gdnStateLength);
        return floats * sizeof(float);
    }

    /// <summary>
    /// Bytes one token adds to a sequence's growing state, by the rows the forward pass writes per token: K and V for
    /// every attention layer; for each attention layer that has an indexer, the raw indexer key (one per token) and the
    /// pooled key (one per <paramref name="indexerKPool"/> tokens, amortised and rounded up); and, when PLE is present,
    /// one token in its history (an int).
    /// </summary>
    public static long PerTokenBytes(
        int attentionLayers, int indexerLayers, int numHeadsKv, int headDim, int indexerKeyLength, int indexerKPool, bool hasPle)
    {
        long kv = 2L * numHeadsKv * headDim * sizeof(float);
        long keyRow = (long)indexerKeyLength * sizeof(float);
        long pooledPerToken = indexerKPool > 0 ? (keyRow + indexerKPool - 1) / indexerKPool : keyRow;
        long bytes = attentionLayers * kv + indexerLayers * (keyRow + pooledPerToken);
        if (hasPle) bytes += sizeof(int);
        return bytes;
    }

    /// <summary>
    /// Bytes this state holds beyond the QSA rows it shares with its clones: the GDN and PLE conv buffers, one reference
    /// per row in each QSA list (the list spines), and this copy's PLE token history. Counted from the real arrays and
    /// lists, so it is exact for the state as it stands.
    /// </summary>
    public long HeldBytes
    {
        get
        {
            long floats = PleConvHistory.Length;
            for (int l = 0; l < GdnConvState.Length; l++)
                floats += GdnConvState[l].Length + GdnState[l].Length;

            long references = 0;
            for (int l = 0; l < QsaKeyCache.Length; l++)
                references += QsaKeyCache[l].Count + QsaValCache[l].Count + QsaRawIndexKCache[l].Count + QsaPooledKeys[l].Count;

            long hasherBytes = (long)(PleHasher?.HistoryLength ?? 0) * sizeof(int);
            return floats * sizeof(float) + references * IntPtr.Size + hasherBytes;
        }
    }

    /// <summary>Deep copy: the clone and this sequence can continue independently.</summary>
    public Qwen4ExpSequenceState Clone()
    {
        ThrowIfDisposed();
        return new Qwen4ExpSequenceState(this);
    }

    /// <summary>Snapshot of the current state; restore it later with <see cref="Restore"/>.</summary>
    public Qwen4ExpSequenceState Snapshot() => Clone();

    /// <summary>Replaces this sequence's state with a deep copy of <paramref name="snapshot"/>.</summary>
    public void Restore(Qwen4ExpSequenceState snapshot)
    {
        ThrowIfDisposed();
        var copy = snapshot.Clone();
        GdnConvState = copy.GdnConvState;
        GdnState = copy.GdnState;
        PleConvHistory = copy.PleConvHistory;
        PleTokensSeen = copy.PleTokensSeen;
        TokensProcessed = copy.TokensProcessed;
        PleHasher = copy.PleHasher;
        QsaKeyCache = copy.QsaKeyCache;
        QsaValCache = copy.QsaValCache;
        QsaRawIndexKCache = copy.QsaRawIndexKCache;
        QsaPooledKeys = copy.QsaPooledKeys;
    }

    /// <summary>Returns the sequence to the state of a fresh sequence (no tokens seen).</summary>
    public void Reset()
    {
        ThrowIfDisposed();
        PleTokensSeen = 0;
        TokensProcessed = 0;
        PleHasher?.Reset();
        Array.Clear(PleConvHistory);
        for (int l = 0; l < GdnConvState.Length; l++)
        {
            if (GdnConvState[l].Length > 0) Array.Clear(GdnConvState[l]);
            if (GdnState[l].Length > 0) Array.Clear(GdnState[l]);
            QsaKeyCache[l].Clear();
            QsaValCache[l].Clear();
            QsaRawIndexKCache[l].Clear();
            QsaPooledKeys[l].Clear();
        }
    }

    /// <summary>True once the sequence is disposed; a disposed sequence rejects further use.</summary>
    public bool IsDisposed => _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        GdnConvState = Array.Empty<float[]>();
        GdnState = Array.Empty<float[]>();
        PleConvHistory = Array.Empty<float>();
        PleHasher = null;
        QsaKeyCache = Array.Empty<List<float[]>>();
        QsaValCache = Array.Empty<List<float[]>>();
        QsaRawIndexKCache = Array.Empty<List<float[]>>();
        QsaPooledKeys = Array.Empty<List<float[]>>();
    }

    internal void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(Qwen4ExpSequenceState));
    }

    private static List<float[]>[] EmptyLists(int n)
    {
        var lists = new List<float[]>[n];
        for (int l = 0; l < n; l++) lists[l] = new List<float[]>();
        return lists;
    }

    private static float[][] CopyArrays(float[][] src)
    {
        var dst = new float[src.Length][];
        for (int l = 0; l < src.Length; l++) dst[l] = (float[])src[l].Clone();
        return dst;
    }

    /// <summary>
    /// Copies the QSA history lists but shares their rows. A row is written once, when it is appended, and never
    /// changed afterwards (the forward pass only appends and reads), so a copy can share the rows and copy only the
    /// list spine. Without this, every snapshot duplicated the whole attention history (the per-session doubling).
    /// </summary>
    private static List<float[]>[] CopyLists(List<float[]>[] src)
    {
        var dst = new List<float[]>[src.Length];
        for (int l = 0; l < src.Length; l++)
            dst[l] = new List<float[]>(src[l]);
        return dst;
    }
}
