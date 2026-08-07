using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Tests for PagedKvCache: paged memory layout, cross-page access, soft truncate, prefix reuse.
/// </summary>
public sealed unsafe class PagedKvCacheTests : IDisposable
{
    // 2 layers, 2 KV heads, head dim 4 → kvDim = 8, PageSize = 16
    private readonly PagedKvCache _cache = new(numLayers: 2, numKvHeads: 2, headDim: 4);

    public void Dispose() => _cache.Dispose();

    private static void Append(PagedKvCache cache, int layer, float k, float v)
    {
        float[] ks = [k, k, k, k, k, k, k, k];
        float[] vs = [v, v, v, v, v, v, v, v];
        cache.Append(layer, ks, vs);
    }

    private static void AppendToken(PagedKvCache cache, float k, float v)
    {
        Append(cache, 0, k, v);
        Append(cache, 1, k, v);
        cache.IncrementPosition();
    }

    /// <summary>
    /// Values are stored transposed inside a page ([numKvHeads][PageSize][headDim]) so each KV
    /// head's slices are contiguous across positions. The other tests here write one repeated
    /// value per token, which cannot distinguish a head or position mix-up, so this writes a
    /// distinct value per (position, head, d) and reads every one back. Dimensions are
    /// deliberately not powers of two to catch stride arithmetic that only works when they are.
    /// </summary>
    [Fact]
    public void TransposedValues_RoundTripForEveryHeadAndPosition()
    {
        const int kvHeads = 3, headDim = 5, kvDim = kvHeads * headDim;
        const int positions = PagedKvCache.PageSize * 2 + 3;   // spans three pages, ends mid-page
        using var cache = new PagedKvCache(numLayers: 1, numKvHeads: kvHeads, headDim: headDim);

        static float Expected(int p, int h, int d) => 1000f + p * 100f + h * 10f + d;

        for (int p = 0; p < positions; p++)
        {
            float[] k = new float[kvDim];
            float[] v = new float[kvDim];
            for (int h = 0; h < kvHeads; h++)
                for (int d = 0; d < headDim; d++)
                {
                    k[h * headDim + d] = Expected(p, h, d);
                    v[h * headDim + d] = -Expected(p, h, d);
                }
            cache.Append(0, k, v);
            cache.IncrementPosition();
        }

        for (int p = 0; p < positions; p++)
        {
            float* key = cache.KeyAt(0, p);
            for (int h = 0; h < kvHeads; h++)
            {
                float* val = cache.ValueAtHead(0, p, h);
                for (int d = 0; d < headDim; d++)
                {
                    Assert.Equal(Expected(p, h, d), key[h * headDim + d]);
                    Assert.Equal(-Expected(p, h, d), val[d]);
                }
            }
        }
    }

    [Fact]
    public void NewCache_LengthIsZero()
    {
        Assert.Equal(0, _cache.Length);
    }

    [Fact]
    public void Append_IncreasesLength()
    {
        AppendToken(_cache, 1f, 2f);
        Assert.Equal(1, _cache.Length);
    }

    [Fact]
    public void KeyAt_ReturnsCorrectValue()
    {
        AppendToken(_cache, 42f, 0f);

        float* k = _cache.KeyAt(0, 0);
        Assert.Equal(42f, k[0]);
        Assert.Equal(42f, k[7]);
    }

    [Fact]
    public void ValueAt_ReturnsCorrectValue()
    {
        AppendToken(_cache, 0f, 99f);

        // Values are stored transposed within the page ([numKvHeads][PageSize][headDim]), so a
        // whole kvDim-wide row is not contiguous — read per KV head. Index 7 of the old row-major
        // layout is head 1 / d 3.
        Assert.Equal(99f, _cache.ValueAtHead(0, 0, 0)[0]);
        Assert.Equal(99f, _cache.ValueAtHead(0, 0, 1)[3]);
    }

    [Fact]
    public void ForkSharedPrefix_RetainsAlignedPagesAfterSourceDisposal_AndAppendsPrivately()
    {
        var source = new PagedKvCache(numLayers: 2, numKvHeads: 2, headDim: 4);
        for (int i = 0; i < PagedKvCache.PageSize; i++) AppendToken(source, i + 1, 0f);

        using var snapshot = source.ForkSharedPrefix(PagedKvCache.PageSize);
        source.Dispose(); // a retained snapshot, not the request which created it, owns the lifetime now
        using var request = snapshot.ForkSharedPrefix(PagedKvCache.PageSize);
        AppendToken(request, 99f, 0f);

        Assert.Equal(1f, snapshot.KeyAt(0, 0)[0]);
        Assert.Equal(PagedKvCache.PageSize, snapshot.Length);
        Assert.Equal(99f, request.KeyAt(0, PagedKvCache.PageSize)[0]);
    }

    [Fact]
    public void ForkSharedPrefix_RewriteWithinSharedPage_IsCopyOnWrite()
    {
        using var source = new PagedKvCache(numLayers: 2, numKvHeads: 2, headDim: 4);
        for (int i = 0; i < PagedKvCache.PageSize; i++) AppendToken(source, i + 1, 0f);
        using var snapshot = source.ForkSharedPrefix(PagedKvCache.PageSize);
        using var request = snapshot.ForkSharedPrefix(PagedKvCache.PageSize);

        request.TruncateTo(PagedKvCache.PageSize - 1);
        AppendToken(request, 777f, 0f);

        Assert.Equal(16f, snapshot.KeyAt(0, PagedKvCache.PageSize - 1)[0]);
        Assert.Equal(777f, request.KeyAt(0, PagedKvCache.PageSize - 1)[0]);
    }

    /// <summary>
    /// Gemma-style caches size pages for the widest layer while each layer's transposed V planes
    /// use its own head stride. A prefix fork must retain that geometry; losing it makes higher KV
    /// heads read a plausible but unrelated part of the V region.
    /// </summary>
    [Fact]
    public void PerLayerHeadDim_ForkRetainsTransposedValueStrides()
    {
        const int kvHeads = 2, maxHeadDim = 4;
        using var source = new PagedKvCache(numLayers: 2, numKvHeads: kvHeads, headDim: maxHeadDim,
            layerHeadDim: [2, 4]);

        for (int pos = 0; pos < PagedKvCache.PageSize; pos++)
        {
            float[] key = new float[kvHeads * maxHeadDim];
            float[] narrowValue = [100 + pos, 101 + pos, 200 + pos, 201 + pos, 0, 0, 0, 0];
            float[] wideValue = [300 + pos, 301 + pos, 302 + pos, 303 + pos,
                                 400 + pos, 401 + pos, 402 + pos, 403 + pos];
            source.Append(0, key, narrowValue);
            source.Append(1, key, wideValue);
            source.IncrementPosition();
        }

        using var fork = source.ForkSharedPrefix(PagedKvCache.PageSize);
        source.Dispose();

        Assert.Equal(101f, fork.ValueAtHead(0, 0, 0)[1]);
        Assert.Equal(201f, fork.ValueAtHead(0, 0, 1)[1]);
        Assert.Equal(303f, fork.ValueAtHead(1, 0, 0)[3]);
        Assert.Equal(403f, fork.ValueAtHead(1, 0, 1)[3]);
    }

    /// <summary>
    /// Regression for the geometry that exposed the per-layer V-stride defect: Gemma-style SWA
    /// layers use 256-wide heads while global layers use 512-wide heads, with eight KV heads.
    /// Earlier 2-head / 4-wide fixtures exercised the layout abstraction but made an accidental
    /// model-wide stride much harder to recognize. Read the first and final KV heads across a page
    /// boundary: head zero stays correct under either stride, whereas head seven immediately
    /// exposes a 256-vs-512 V-plane error.
    /// </summary>
    [Fact]
    public void PerLayerHeadDim_ProductionGemmaGeometry_RoundTripsEveryValuePlane()
    {
        const int kvHeads = 8, narrowHeadDim = 256, wideHeadDim = 512;
        const int positions = PagedKvCache.PageSize + 1;
        using var cache = new PagedKvCache(numLayers: 2, numKvHeads: kvHeads, headDim: wideHeadDim,
            layerHeadDim: [narrowHeadDim, wideHeadDim]);

        static float Expected(int layer, int position, int head, int dimension) =>
            layer * 1_000_000f + position * 10_000f + head * 1_000f + dimension;

        for (int position = 0; position < positions; position++)
        {
            float[] key = new float[kvHeads * wideHeadDim];
            float[] narrow = new float[kvHeads * narrowHeadDim];
            float[] wide = new float[kvHeads * wideHeadDim];
            for (int head = 0; head < kvHeads; head++)
            {
                for (int dimension = 0; dimension < narrowHeadDim; dimension++)
                    narrow[head * narrowHeadDim + dimension] = Expected(0, position, head, dimension);
                for (int dimension = 0; dimension < wideHeadDim; dimension++)
                    wide[head * wideHeadDim + dimension] = Expected(1, position, head, dimension);
            }
            cache.Append(0, key, narrow);
            cache.Append(1, key, wide);
            cache.IncrementPosition();
        }

        foreach (int position in new[] { 0, PagedKvCache.PageSize - 1, PagedKvCache.PageSize })
        foreach (int head in new[] { 0, kvHeads - 1 })
        {
            Assert.Equal(Expected(0, position, head, 0), cache.ValueAtHead(0, position, head)[0]);
            Assert.Equal(Expected(0, position, head, narrowHeadDim - 1),
                cache.ValueAtHead(0, position, head)[narrowHeadDim - 1]);
            Assert.Equal(Expected(1, position, head, 0), cache.ValueAtHead(1, position, head)[0]);
            Assert.Equal(Expected(1, position, head, wideHeadDim - 1),
                cache.ValueAtHead(1, position, head)[wideHeadDim - 1]);
        }
    }

    [Fact]
    public void PerLayerHeadDim_Bf16StoreUsesLayerValueStride()
    {
        using var cache = new PagedKvCache(numLayers: 2, numKvHeads: 2, headDim: 4,
            bf16Store: true, layerHeadDim: [2, 4]);
        float[] key = new float[8];
        cache.Append(0, key, [10f, 11f, 20f, 21f, 0f, 0f, 0f, 0f]);
        cache.Append(1, key, [30f, 31f, 32f, 33f, 40f, 41f, 42f, 43f]);
        cache.IncrementPosition();

        static float Widen(ushort value) => BitConverter.UInt32BitsToSingle((uint)value << 16);
        Assert.Equal(21f, Widen(cache.Bf16ValueAtHead(0, 0, 1)[1]));
        Assert.Equal(43f, Widen(cache.Bf16ValueAtHead(1, 0, 1)[3]));
    }

    [Fact]
    public void PerLayerHeadDim_PersistedStateRequiresMatchingGeometry()
    {
        using var source = new PagedKvCache(numLayers: 2, numKvHeads: 2, headDim: 4,
            layerHeadDim: [2, 4]);
        float[] key = new float[8];
        source.Append(0, key, [10f, 11f, 20f, 21f, 0f, 0f, 0f, 0f]);
        source.Append(1, key, [30f, 31f, 32f, 33f, 40f, 41f, 42f, 43f]);
        source.IncrementPosition();
        byte[] state = source.ExportKvState();

        using var compatible = new PagedKvCache(numLayers: 2, numKvHeads: 2, headDim: 4,
            layerHeadDim: [2, 4]);
        compatible.ImportKvState(state);
        Assert.Equal(21f, compatible.ValueAtHead(0, 0, 1)[1]);
        Assert.Equal(43f, compatible.ValueAtHead(1, 0, 1)[3]);

        using var incompatible = new PagedKvCache(numLayers: 2, numKvHeads: 2, headDim: 4);
        Assert.Throws<InvalidOperationException>(() => incompatible.ImportKvState(state));
    }

    [Fact]
    public void PersistedState_Version1UniformGeometry_RemainsImportable()
    {
        using var source = new PagedKvCache(numLayers: 1, numKvHeads: 1, headDim: 2);
        source.Append(0, [1f, 2f], [3f, 4f]);
        source.IncrementPosition();

        // Version 2 adds a layer-geometry flag immediately after the original 35-byte header.
        // Remove the false flag from a uniform-cache stream to model an existing v1 artifact.
        byte[] v2 = source.ExportKvState();
        const int V1HeaderBytes = 35;
        byte[] v1 = new byte[v2.Length - 1];
        v2.AsSpan(0, V1HeaderBytes).CopyTo(v1);
        v2.AsSpan(V1HeaderBytes + 1).CopyTo(v1.AsSpan(V1HeaderBytes));
        BitConverter.TryWriteBytes(v1.AsSpan(sizeof(uint)), (ushort)1);

        using var restored = new PagedKvCache(numLayers: 1, numKvHeads: 1, headDim: 2);
        restored.ImportKvState(v1);
        Assert.Equal(1f, restored.KeyAt(0, 0)[0]);
        Assert.Equal(4f, restored.ValueAtHead(0, 0, 0)[1]);
    }

    [Fact]
    public void LayersAreIndependent()
    {
        Append(_cache, 0, 11f, 12f);
        Append(_cache, 1, 21f, 22f);
        _cache.IncrementPosition();

        Assert.Equal(11f, _cache.KeyAt(0, 0)[0]);
        Assert.Equal(12f, _cache.ValueAtHead(0, 0, 0)[0]);
        Assert.Equal(21f, _cache.KeyAt(1, 0)[0]);
        Assert.Equal(22f, _cache.ValueAtHead(1, 0, 0)[0]);
    }

    [Fact]
    public void MultiplePositions_CorrectLayout()
    {
        for (int i = 0; i < 5; i++)
            AppendToken(_cache, i, -i);

        for (int i = 0; i < 5; i++)
        {
            Assert.Equal((float)i, _cache.KeyAt(0, i)[0]);
            Assert.Equal((float)-i, _cache.ValueAtHead(0, i, 0)[0]);
        }
    }

    [Fact]
    public void CrossPageBoundary_CorrectAccess()
    {
        // Fill exactly two pages (PageSize = 16, so positions 0..31)
        for (int i = 0; i < PagedKvCache.PageSize * 2; i++)
            AppendToken(_cache, i, i * 10f);

        Assert.Equal(PagedKvCache.PageSize * 2, _cache.Length);

        // Check first position of second page
        int p = PagedKvCache.PageSize;
        Assert.Equal((float)p, _cache.KeyAt(0, p)[0]);
        Assert.Equal((float)(p * 10), _cache.ValueAtHead(0, p, 0)[0]);

        // Check last position of second page
        int last = PagedKvCache.PageSize * 2 - 1;
        Assert.Equal((float)last, _cache.KeyAt(0, last)[0]);
    }

    [Fact]
    public void TruncateTo_SoftTruncate_PreservesExistingPages()
    {
        for (int i = 0; i < 5; i++)
            AppendToken(_cache, i, 0f);

        _cache.TruncateTo(2);
        Assert.Equal(2, _cache.Length);

        // Pages are still valid — can read position 0 and 1
        Assert.Equal(0f, _cache.KeyAt(0, 0)[0]);
        Assert.Equal(1f, _cache.KeyAt(0, 1)[0]);
    }

    [Fact]
    public void TruncateTo_ThenAppend_OverwritesCorrectly()
    {
        for (int i = 0; i < 5; i++)
            AppendToken(_cache, i, 0f);

        _cache.TruncateTo(3);

        // Overwrite position 3 with a new value
        AppendToken(_cache, 99f, 0f);

        Assert.Equal(4, _cache.Length);
        Assert.Equal(99f, _cache.KeyAt(0, 3)[0]);
        // Positions before 3 are unchanged
        Assert.Equal(2f, _cache.KeyAt(0, 2)[0]);
    }

    [Fact]
    public void Reset_ReturnsLengthToZero()
    {
        for (int i = 0; i < 8; i++)
            AppendToken(_cache, i, 0f);

        _cache.Reset();
        Assert.Equal(0, _cache.Length);
    }

    [Fact]
    public void Reset_AllowsReuseOfPages()
    {
        // Fill, reset, and fill again — pages should be reused from warm pool
        for (int i = 0; i < PagedKvCache.PageSize; i++)
            AppendToken(_cache, (float)i, 0f);

        _cache.Reset();

        for (int i = 0; i < PagedKvCache.PageSize; i++)
            AppendToken(_cache, (float)(i + 100), 0f);

        Assert.Equal(PagedKvCache.PageSize, _cache.Length);
        Assert.Equal(100f, _cache.KeyAt(0, 0)[0]);
        Assert.Equal(115f, _cache.KeyAt(0, 15)[0]);
    }

    [Fact]
    public void PrefixReuse_TruncateAndContinue()
    {
        // Simulate prefix caching: fill 32 positions (2 pages), truncate to 16, continue from 16
        for (int i = 0; i < 32; i++)
            AppendToken(_cache, (float)i, 0f);

        // New request: same prefix (0..15 preserved), truncate to prefix length
        _cache.TruncateTo(16);

        // Fill positions 16..19 with new values
        for (int i = 0; i < 4; i++)
            AppendToken(_cache, (float)(200 + i), 0f);

        Assert.Equal(20, _cache.Length);

        // Prefix positions still valid
        Assert.Equal(0f, _cache.KeyAt(0, 0)[0]);
        Assert.Equal(15f, _cache.KeyAt(0, 15)[0]);

        // New suffix positions have new values
        Assert.Equal(200f, _cache.KeyAt(0, 16)[0]);
        Assert.Equal(203f, _cache.KeyAt(0, 19)[0]);
    }

    [Fact]
    public void MaxSeqLen_ReflectsMaxBlocks()
    {
        using var small = new PagedKvCache(numLayers: 1, numKvHeads: 1, headDim: 4, maxBlocks: 4);
        Assert.Equal(4 * PagedKvCache.PageSize, small.MaxSeqLen);
    }

    [Fact]
    public void ReserveBlock_AllowsLayerOneToAppendFirst()
    {
        // ReserveBlock makes the "layer 0 must call Append first" invariant optional —
        // hybrid models can call any layer's Append after a ReserveBlock at the page boundary.
        _cache.ReserveBlock();
        Append(_cache, 1, 7f, 11f);            // layer 1 first, no layer-0 write
        // Layer 1 should now read back what it wrote.
        Assert.Equal(7f,  _cache.KeyAt(1, 0)[0]);
        Assert.Equal(11f, _cache.ValueAtHead(1, 0, 0)[0]);
        _cache.IncrementPosition();
        Assert.Equal(1, _cache.Length);
    }

    [Fact]
    public void ReserveBlock_IdempotentWithinSameBlock()
    {
        // Multiple ReserveBlock calls inside the same PageSize window should be no-ops.
        _cache.ReserveBlock();
        _cache.ReserveBlock();
        _cache.ReserveBlock();
        Append(_cache, 0, 1f, 2f);
        _cache.IncrementPosition();
        Append(_cache, 0, 3f, 4f);
        _cache.IncrementPosition();
        Assert.Equal(1f, _cache.KeyAt(0, 0)[0]);
        Assert.Equal(3f, _cache.KeyAt(0, 1)[0]);
        Assert.Equal(2, _cache.Length);
    }

    [Fact]
    public void ReserveBlock_AcrossPageBoundary_AllocatesNewBlock()
    {
        // Fill page 0 with 16 tokens, then ReserveBlock at the boundary and verify page 1
        // is usable from layer 1 only (layer 0's page-1 slot stays unallocated).
        for (int i = 0; i < PagedKvCache.PageSize; i++)
            AppendToken(_cache, i, i + 100f);
        Assert.Equal(PagedKvCache.PageSize, _cache.Length);

        _cache.ReserveBlock();                 // crosses into page 1
        Append(_cache, 1, 42f, 99f);           // layer 1 only writes page 1
        _cache.IncrementPosition();

        Assert.Equal(PagedKvCache.PageSize + 1, _cache.Length);
        Assert.Equal(42f, _cache.KeyAt(1, PagedKvCache.PageSize)[0]);
        Assert.Equal(99f, _cache.ValueAtHead(1, PagedKvCache.PageSize, 0)[0]);
    }

    // ── SnapKV (issue #51) compaction ──────────────────────────────────────

    [Fact]
    public void Compact_KeepEverything_NoOp()
    {
        for (int i = 0; i < 20; i++) AppendToken(_cache, i, -i);
        var keep = new int[20];
        for (int i = 0; i < 20; i++) keep[i] = i;

        _cache.Compact(keep);

        Assert.Equal(20, _cache.Length);
        Assert.Equal(20, _cache.LogicalLength);
        for (int i = 0; i < 20; i++)
        {
            Assert.Equal((float)i, _cache.KeyAt(0, i)[0]);
            Assert.Equal((float)-i, _cache.ValueAtHead(0, i, 0)[0]);
        }
    }

    [Fact]
    public void Compact_DropsEvictedPositions_KeepsOrder()
    {
        // 20 tokens at positions 0..19; keep {0, 5, 11, 17, 19}.
        for (int i = 0; i < 20; i++) AppendToken(_cache, i + 1f, -(i + 1f));
        int[] keep = { 0, 5, 11, 17, 19 };

        _cache.Compact(keep);

        Assert.Equal(5, _cache.Length);              // slot count drops
        Assert.Equal(20, _cache.LogicalLength);      // RoPE frame preserved
        // Slot i now holds what was at position keep[i].
        for (int i = 0; i < keep.Length; i++)
        {
            float expectedK = keep[i] + 1f;
            float expectedV = -(keep[i] + 1f);
            Assert.Equal(expectedK, _cache.KeyAt(0, i)[0]);
            Assert.Equal(expectedV, _cache.ValueAtHead(0, i, 0)[0]);
            Assert.Equal(expectedK, _cache.KeyAt(1, i)[0]);
            Assert.Equal(expectedV, _cache.ValueAtHead(1, i, 0)[0]);
        }
    }

    [Fact]
    public void Compact_ThenAppend_NewTokenLandsAtCompactedTail()
    {
        for (int i = 0; i < 20; i++) AppendToken(_cache, i + 1f, 0f);
        int[] keep = { 0, 5, 11, 17, 19 };
        _cache.Compact(keep);

        // A decode-side append should write at slot 5 (the new tail), while
        // LogicalLength advances from 20 to 21 — the decode caller will RoPE
        // the new K at position 20 (the original sequence frame).
        AppendToken(_cache, 999f, 0f);

        Assert.Equal(6, _cache.Length);
        Assert.Equal(21, _cache.LogicalLength);
        Assert.Equal(999f, _cache.KeyAt(0, 5)[0]);
        // Pre-compaction survivors untouched.
        Assert.Equal(6f, _cache.KeyAt(0, 1)[0]);  // was position 5, value 6f
    }

    [Fact]
    public void Compact_RejectsOutOfRange()
    {
        for (int i = 0; i < 8; i++) AppendToken(_cache, i, 0f);
        // 8 stored — position 8 is out of range.
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            _cache.Compact(new[] { 0, 8 }));
    }

    [Fact]
    public void Compact_RejectsUnsorted()
    {
        for (int i = 0; i < 8; i++) AppendToken(_cache, i, 0f);
        Assert.Throws<ArgumentException>(() =>
            _cache.Compact(new[] { 3, 1 }));
    }

    /// <summary>
    /// Issue #130 gate contract: <see cref="PagedKvCache.Length"/> and
    /// <see cref="PagedKvCache.LogicalLength"/> stay equal through all normal operations
    /// (Append/IncrementPosition, TruncateTo, Reset) and diverge ONLY after eviction
    /// (Compact / CompactLengthOnly). The <c>SupportsBatchVerify</c> gate keys off
    /// <c>Length != LogicalLength</c> as its "eviction occurred" signal, so this invariant
    /// is load-bearing: if a future cache change broke it, the gate would mis-fire and only
    /// the heavy dev-box-only GPU oracle would catch it. This locks it in CI (no GPU/model).
    /// </summary>
    [Fact]
    public void LengthAndLogicalLength_EqualExceptAfterEviction()
    {
        // Normal appends advance both in lockstep.
        for (int i = 0; i < 8; i++) AppendToken(_cache, i, 0f);
        Assert.Equal(8, _cache.Length);
        Assert.Equal(_cache.Length, _cache.LogicalLength);

        // CompactLengthOnly (the SnapKV host-bookkeeping path) drops the physical
        // length to the budget K while leaving the logical RoPE position untouched.
        _cache.CompactLengthOnly(3);
        Assert.Equal(3, _cache.Length);
        Assert.Equal(8, _cache.LogicalLength);
        Assert.NotEqual(_cache.Length, _cache.LogicalLength); // gate would now fire

        // TruncateTo re-synchronizes both (speculative rewind / per-layer prefill reset).
        _cache.TruncateTo(5);
        Assert.Equal(5, _cache.Length);
        Assert.Equal(5, _cache.LogicalLength);

        // Reset zeroes both.
        _cache.Reset();
        Assert.Equal(0, _cache.Length);
        Assert.Equal(0, _cache.LogicalLength);

        // Position-based Compact (keep a subset) also leaves logical length at the
        // pre-compaction value while physical length shrinks to the kept count.
        for (int i = 0; i < 8; i++) AppendToken(_cache, i, 0f);
        _cache.Compact(new[] { 0, 2, 4, 6 });
        Assert.Equal(4, _cache.Length);
        Assert.Equal(8, _cache.LogicalLength);
    }
}
