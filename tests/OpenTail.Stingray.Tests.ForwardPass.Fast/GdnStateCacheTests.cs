
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Tests for GdnStateCache: per-sequence Gated DeltaNet recurrent state lifecycle
/// (allocate, layer mapping, reset, restricted truncate).
/// </summary>
public sealed unsafe class GdnStateCacheTests
{
    // Realistic qwen35moe trunk shape: 40 layers, full attention when (i+1) % 4 == 0.
    // → indices 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 are Attention (10 layers).
    // → the remaining 30 layers are GatedDeltaNet.
    private static LayerType[] RealisticLayerTypes()
    {
        var types = new LayerType[40];
        for (int i = 0; i < 40; i++)
            types[i] = ((i + 1) % 4 == 0) ? LayerType.Attention : LayerType.GatedDeltaNet;
        return types;
    }

    // Realistic GDN config from qwen35moe.
    private static GdnConfig RealisticGdn() => new(
        NumKHeads: 16,
        NumVHeads: 32,
        HeadDim: 128,
        InnerSize: 4096,
        ConvKernel: 4,
        FullAttentionInterval: 4);

    [Fact]
    public void Construct_RealisticHybridShape_AllocatesExpectedLayerCount()
    {
        using var cache = new GdnStateCache(RealisticLayerTypes(), RealisticGdn());

        Assert.Equal(30, cache.NumGdnLayers);
        Assert.Equal(0, cache.Length);

        // Every GDN layer slot is non-null and unique across both buffers.
        var seen = new HashSet<nuint>();
        for (int g = 0; g < cache.NumGdnLayers; g++)
        {
            float* cp = cache.ConvStateAt(g);
            float* sp = cache.ScanStateAt(g);
            Assert.True(cp != null, $"Conv state for GDN layer {g} is null");
            Assert.True(sp != null, $"Scan state for GDN layer {g} is null");
            Assert.True(seen.Add((nuint)cp), $"Conv state for GDN layer {g} aliases another buffer");
            Assert.True(seen.Add((nuint)sp), $"Scan state for GDN layer {g} aliases another buffer");
        }
    }

    [Fact]
    public void GdnLayerOf_AttentionReturnsMinusOne_GdnReturnsDenseIndex()
    {
        using var cache = new GdnStateCache(RealisticLayerTypes(), RealisticGdn());

        // Trunk layers 0, 1, 2 are GDN → dense indices 0, 1, 2.
        Assert.Equal(0, cache.GdnLayerOf(0));
        Assert.Equal(1, cache.GdnLayerOf(1));
        Assert.Equal(2, cache.GdnLayerOf(2));

        // Trunk layer 3 is full attention.
        Assert.Equal(-1, cache.GdnLayerOf(3));

        // Trunk layer 4 is the next GDN → dense index 3.
        Assert.Equal(3, cache.GdnLayerOf(4));

        // Trunk layer 39 (last) is attention.
        Assert.Equal(-1, cache.GdnLayerOf(39));

        // Trunk layer 38 is GDN; it should be the 29th (zero-indexed) dense entry.
        Assert.Equal(29, cache.GdnLayerOf(38));
    }

    [Fact]
    public void ConstructionZeroesAllState()
    {
        using var cache = new GdnStateCache(RealisticLayerTypes(), RealisticGdn());

        int convLen = cache.ConvStateFloatsPerLayer;
        int scanLen = cache.ScanStateFloatsPerLayer;

        for (int g = 0; g < cache.NumGdnLayers; g++)
        {
            float* cp = cache.ConvStateAt(g);
            float* sp = cache.ScanStateAt(g);

            // First and last 64 floats of each buffer should be zero.
            for (int i = 0; i < 64; i++)
                Assert.Equal(0f, cp[i]);
            for (int i = 0; i < 64; i++)
                Assert.Equal(0f, cp[convLen - 1 - i]);

            for (int i = 0; i < 64; i++)
                Assert.Equal(0f, sp[i]);
            for (int i = 0; i < 64; i++)
                Assert.Equal(0f, sp[scanLen - 1 - i]);
        }
    }

    [Fact]
    public void Reset_ZeroesAllStateAndLengthGoesToZero()
    {
        using var cache = new GdnStateCache(RealisticLayerTypes(), RealisticGdn());

        // Write nonzero scribbles into a couple of slots and advance length.
        float* c0 = cache.ConvStateAt(0);
        float* sLast = cache.ScanStateAt(cache.NumGdnLayers - 1);
        c0[0] = 1.5f;
        c0[100] = -2.25f;
        sLast[0] = 3.5f;
        sLast[cache.ScanStateFloatsPerLayer - 1] = 9.0f;
        cache.IncrementPosition();
        cache.IncrementPosition();

        Assert.Equal(2, cache.Length);

        cache.Reset();

        Assert.Equal(0, cache.Length);
        Assert.Equal(0f, c0[0]);
        Assert.Equal(0f, c0[100]);
        Assert.Equal(0f, sLast[0]);
        Assert.Equal(0f, sLast[cache.ScanStateFloatsPerLayer - 1]);
    }

    [Fact]
    public void IncrementPosition_AdvancesLength()
    {
        using var cache = new GdnStateCache(RealisticLayerTypes(), RealisticGdn());

        cache.IncrementPosition();
        cache.IncrementPosition();
        cache.IncrementPosition();

        Assert.Equal(3, cache.Length);
    }

    [Fact]
    public void TruncateTo_ZeroLengthOrCurrentLength_Works()
    {
        using var cache = new GdnStateCache(RealisticLayerTypes(), RealisticGdn());

        cache.IncrementPosition();
        cache.IncrementPosition();
        cache.IncrementPosition();
        Assert.Equal(3, cache.Length);

        // No-op: matches the ContinuousBatchingEngine pattern.
        cache.TruncateTo(3);
        Assert.Equal(3, cache.Length);

        // Alias for Reset.
        float* c0 = cache.ConvStateAt(0);
        c0[0] = 42f;
        cache.TruncateTo(0);
        Assert.Equal(0, cache.Length);
        Assert.Equal(0f, c0[0]);
    }

    [Fact]
    public void TruncateTo_OtherLengths_Throws()
    {
        using var cache = new GdnStateCache(RealisticLayerTypes(), RealisticGdn());
        cache.IncrementPosition();
        cache.IncrementPosition();
        cache.IncrementPosition();

        var ex = Assert.Throws<InvalidOperationException>(() => cache.TruncateTo(1));
        // The message names the architecture so callers can route around it.
        Assert.True(
            ex.Message.Contains("GDN", StringComparison.Ordinal) ||
            ex.Message.Contains("DeltaNet", StringComparison.Ordinal),
            $"Expected exception message to mention GDN/DeltaNet, got: {ex.Message}");

        // Negative lengths and lengths above current both fail the same way.
        Assert.Throws<InvalidOperationException>(() => cache.TruncateTo(-1));
        Assert.Throws<InvalidOperationException>(() => cache.TruncateTo(4));
    }

    [Fact]
    public void Dispose_DoubleDisposeDoesNotThrow()
    {
        var cache = new GdnStateCache(RealisticLayerTypes(), RealisticGdn());
        cache.Dispose();
        cache.Dispose(); // must be a no-op
    }

    [Fact]
    public void SmallShape_TotalBytes_MatchesFormula()
    {
        // 1 GDN layer, NumVHeads=2, HeadDim=4, ConvKernel=2, ConvChannels=4 (= 2*2 + 4 = 8?
        // KeyDim = NumKHeads*HeadDim = 1*4 = 4, ValueDim = NumVHeads*HeadDim = 2*4 = 8,
        // ConvChannels = KeyDim*2 + ValueDim = 4*2 + 8 = 16. So the formula uses the
        // derived ConvChannels, not the user's free choice. We test against the derived
        // value to verify we're computing layout correctly.
        var layerTypes = new[] { LayerType.GatedDeltaNet };
        var tiny = new GdnConfig(
            NumKHeads: 1,
            NumVHeads: 2,
            HeadDim: 4,
            InnerSize: 8,
            ConvKernel: 2,
            FullAttentionInterval: 1);

        using var cache = new GdnStateCache(layerTypes, tiny);
        Assert.Equal(1, cache.NumGdnLayers);

        long expectedConvBytes = (long)(tiny.ConvKernel - 1) * tiny.ConvChannels * sizeof(float);
        long expectedScanBytes = (long)tiny.NumVHeads * tiny.HeadDim * tiny.HeadDim * sizeof(float);
        long expectedTotal = cache.NumGdnLayers * (expectedConvBytes + expectedScanBytes);

        Assert.Equal(expectedConvBytes / sizeof(float), cache.ConvStateFloatsPerLayer);
        Assert.Equal(expectedScanBytes / sizeof(float), cache.ScanStateFloatsPerLayer);
        Assert.Equal(expectedTotal, cache.TotalBytes);

        // Sanity: writes to one buffer don't clobber the other, and bytes are addressable
        // across the full extent.
        float* cp = cache.ConvStateAt(0);
        float* sp = cache.ScanStateAt(0);
        cp[0] = 1f;
        cp[cache.ConvStateFloatsPerLayer - 1] = 2f;
        sp[0] = 3f;
        sp[cache.ScanStateFloatsPerLayer - 1] = 4f;

        Assert.Equal(1f, cp[0]);
        Assert.Equal(2f, cp[cache.ConvStateFloatsPerLayer - 1]);
        Assert.Equal(3f, sp[0]);
        Assert.Equal(4f, sp[cache.ScanStateFloatsPerLayer - 1]);
    }

    [Fact]
    public void ConvStateAt_OutOfRange_Throws()
    {
        using var cache = new GdnStateCache(RealisticLayerTypes(), RealisticGdn());
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.ConvStateAt(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.ConvStateAt(cache.NumGdnLayers));
    }

    [Fact]
    public void ScanStateAt_OutOfRange_Throws()
    {
        using var cache = new GdnStateCache(RealisticLayerTypes(), RealisticGdn());
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.ScanStateAt(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.ScanStateAt(cache.NumGdnLayers));
    }

    [Fact]
    public void EmptyLayerTypes_DegeneratesSafely()
    {
        using var cache = new GdnStateCache(Array.Empty<LayerType>(), RealisticGdn());

        Assert.Equal(0, cache.NumGdnLayers);
        Assert.Equal(0L, cache.TotalBytes);
        cache.Reset();          // no-op
        cache.TruncateTo(0);    // no-op
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.ConvStateAt(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.ScanStateAt(0));
    }

    [Fact]
    public void AllAttentionLayers_HasNoGdnSlots()
    {
        var allAttn = new LayerType[8];
        Array.Fill(allAttn, LayerType.Attention);
        using var cache = new GdnStateCache(allAttn, RealisticGdn());

        Assert.Equal(0, cache.NumGdnLayers);
        for (int i = 0; i < 8; i++)
            Assert.Equal(-1, cache.GdnLayerOf(i));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.ConvStateAt(0));
    }

    [Fact]
    public void ConvKernel1_ConvStateIsNullScanStateIsAllocated()
    {
        var gdn = new GdnConfig(
            NumKHeads: 1,
            NumVHeads: 1,
            HeadDim: 2,
            InnerSize: 2,
            ConvKernel: 1,
            FullAttentionInterval: 4);
        using var cache = new GdnStateCache(new[] { LayerType.GatedDeltaNet }, gdn);

        Assert.Equal(1, cache.NumGdnLayers);
        Assert.True(cache.ConvStateAt(0) == null, "Conv state should be null when ConvKernel == 1 (no past tokens to store).");
        Assert.True(cache.ScanStateAt(0) != null);
    }

    [Fact]
    public void GdnStateCache_RetainedSessionLifecycle_ConformanceAcrossTurns()
    {
        // Issue #21 / 02-qwen35moe-plan.md: GDN state preservation across chat continuation turns.
        // Recurrent state is destructively updated token-by-token and does NOT support partial rewind.
        // Across turns, session state must be retained via Snapshot/Restore or Reset.
        using var cache = new GdnStateCache(RealisticLayerTypes(), RealisticGdn());

        // Turn 1: Process 5 tokens
        for (int i = 0; i < 5; i++)
            cache.IncrementPosition();
        Assert.Equal(5, cache.Length);

        // Scribble distinctive markers into first, middle, and last layer's conv and scan buffers (head and tail)
        int lastLayer = cache.NumGdnLayers - 1;
        int midLayer = cache.NumGdnLayers / 2;
        int convTail = cache.ConvStateFloatsPerLayer - 1;
        int scanTail = cache.ScanStateFloatsPerLayer - 1;

        cache.ConvStateAt(0)[0] = 12.34f;
        cache.ConvStateAt(0)[convTail] = 23.45f;
        cache.ScanStateAt(0)[0] = 56.78f;
        cache.ScanStateAt(0)[scanTail] = 67.89f;

        cache.ConvStateAt(midLayer)[0] = 34.56f;
        cache.ScanStateAt(midLayer)[0] = 78.90f;

        cache.ConvStateAt(lastLayer)[0] = 45.67f;
        cache.ConvStateAt(lastLayer)[convTail] = 56.78f;
        cache.ScanStateAt(lastLayer)[0] = 89.01f;
        cache.ScanStateAt(lastLayer)[scanTail] = 90.12f;

        // TruncateTo(Length) is a valid, documented no-op (used when sequence is at terminal position)
        cache.TruncateTo(cache.Length);
        Assert.Equal(5, cache.Length);
        Assert.Equal(12.34f, cache.ConvStateAt(0)[0]);

        // Verify partial rewind is prohibited (throws InvalidOperationException)
        Assert.Throws<InvalidOperationException>(() => cache.TruncateTo(4));
        Assert.Throws<InvalidOperationException>(() => cache.TruncateTo(1));

        // Snapshot state at end of Turn 1
        var snapshot = new byte[cache.SnapshotBytes];
        fixed (byte* pSnap = snapshot)
        {
            cache.SnapshotInto(pSnap, snapshot.Length);
        }

        // Turn 2: Advance by 3 more tokens, mutating state
        for (int i = 0; i < 3; i++)
            cache.IncrementPosition();
        Assert.Equal(8, cache.Length);
        cache.ConvStateAt(0)[0] = 999.0f;
        cache.ScanStateAt(0)[0] = -999.0f;
        cache.ConvStateAt(lastLayer)[0] = 888.0f;
        cache.ScanStateAt(lastLayer)[0] = -888.0f;

        // Restore back to Turn 1 snapshot
        fixed (byte* pSnap = snapshot)
        {
            cache.RestoreFrom(pSnap, snapshot.Length);
        }

        // State and length must be exactly restored to Turn 1 across all layers and positions
        Assert.Equal(5, cache.Length);
        Assert.Equal(12.34f, cache.ConvStateAt(0)[0]);
        Assert.Equal(23.45f, cache.ConvStateAt(0)[convTail]);
        Assert.Equal(56.78f, cache.ScanStateAt(0)[0]);
        Assert.Equal(67.89f, cache.ScanStateAt(0)[scanTail]);

        Assert.Equal(34.56f, cache.ConvStateAt(midLayer)[0]);
        Assert.Equal(78.90f, cache.ScanStateAt(midLayer)[0]);

        Assert.Equal(45.67f, cache.ConvStateAt(lastLayer)[0]);
        Assert.Equal(56.78f, cache.ConvStateAt(lastLayer)[convTail]);
        Assert.Equal(89.01f, cache.ScanStateAt(lastLayer)[0]);
        Assert.Equal(90.12f, cache.ScanStateAt(lastLayer)[scanTail]);

        // Reset (or TruncateTo(0)) wipes all state and length for a fresh session
        cache.TruncateTo(0);
        Assert.Equal(0, cache.Length);
        Assert.Equal(0f, cache.ConvStateAt(0)[0]);
        Assert.Equal(0f, cache.ScanStateAt(0)[0]);
        Assert.Equal(0f, cache.ConvStateAt(lastLayer)[0]);
        Assert.Equal(0f, cache.ScanStateAt(lastLayer)[0]);
    }

    [Fact]
    public void GdnStateCache_SnapshotRing_RollbackToIntermediatePosition_Conformance()
    {
        // Issue #21 / 02-qwen35moe-plan.md: Simulates speculative decoding or batched verification
        // where recurrent state is snapshotted per candidate step and restored upon rejection.
        using var cache = new GdnStateCache(RealisticLayerTypes(), RealisticGdn());

        var ringSnapshots = new byte[3][];
        for (int i = 0; i < 3; i++)
            ringSnapshots[i] = new byte[cache.SnapshotBytes];

        int lastLayer = cache.NumGdnLayers - 1;

        // Step 1
        cache.IncrementPosition();
        cache.ConvStateAt(0)[0] = 101f;
        cache.ScanStateAt(0)[0] = 201f;
        cache.ConvStateAt(lastLayer)[0] = 301f;
        cache.ScanStateAt(lastLayer)[0] = 401f;
        fixed (byte* p = ringSnapshots[0]) cache.SnapshotInto(p, ringSnapshots[0].Length);

        // Step 2
        cache.IncrementPosition();
        cache.ConvStateAt(0)[0] = 102f;
        cache.ScanStateAt(0)[0] = 202f;
        cache.ConvStateAt(lastLayer)[0] = 302f;
        cache.ScanStateAt(lastLayer)[0] = 402f;
        fixed (byte* p = ringSnapshots[1]) cache.SnapshotInto(p, ringSnapshots[1].Length);

        // Step 3
        cache.IncrementPosition();
        cache.ConvStateAt(0)[0] = 103f;
        cache.ScanStateAt(0)[0] = 203f;
        cache.ConvStateAt(lastLayer)[0] = 303f;
        cache.ScanStateAt(lastLayer)[0] = 403f;
        fixed (byte* p = ringSnapshots[2]) cache.SnapshotInto(p, ringSnapshots[2].Length);


        Assert.Equal(3, cache.Length);

        // Reject back to Step 1
        fixed (byte* p = ringSnapshots[0]) cache.RestoreFrom(p, ringSnapshots[0].Length);
        Assert.Equal(1, cache.Length);
        Assert.Equal(101f, cache.ConvStateAt(0)[0]);
        Assert.Equal(201f, cache.ScanStateAt(0)[0]);
        Assert.Equal(301f, cache.ConvStateAt(lastLayer)[0]);
        Assert.Equal(401f, cache.ScanStateAt(lastLayer)[0]);

        // Reject/Roll to Step 2
        fixed (byte* p = ringSnapshots[1]) cache.RestoreFrom(p, ringSnapshots[1].Length);
        Assert.Equal(2, cache.Length);
        Assert.Equal(102f, cache.ConvStateAt(0)[0]);
        Assert.Equal(202f, cache.ScanStateAt(0)[0]);
        Assert.Equal(302f, cache.ConvStateAt(lastLayer)[0]);
        Assert.Equal(402f, cache.ScanStateAt(lastLayer)[0]);
    }
}

