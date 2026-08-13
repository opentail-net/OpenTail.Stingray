using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// Parity gate for <see cref="VulkanBackend.AttentionBatchedFlash"/> against the
/// <see cref="VulkanBackend.AttentionBatched"/> kernel it is intended to replace on the prefill
/// path.
///
/// <para>The flash kernel computes the same thing — query qi attends causally over
/// [0, basePos+qi] — but streams K/V once per head with online softmax instead of once per query
/// with a materialised score array. It is therefore <b>not</b> bit-identical: the running
/// max/sum rescaling accumulates in a different order, so these assert a tolerance plus argmax-free
/// closeness rather than exact equality.</para>
///
/// <para>The cases deliberately cover what the online-softmax rewrite is easy to get wrong:
/// mid-sequence starts (basePos > 0, so masking differs per query within one workgroup), grouped
/// query attention (kv_head mapping), a query count that is not a multiple of the 8-wide tile, and
/// a sequence length that is not a multiple of the tile either.</para>
/// </summary>
public sealed class VulkanFlashAttentionTests
{
    private static VulkanBackend? TryCreate()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static float[] Rand(int n, int seed)
    {
        var r = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() * 2.0 - 1.0);
        return a;
    }

    private static void AssertMatches(
        int numHeads, int numKvHeads, int headDim, int basePos, int numQueries, int seed)
    {
        using var g = TryCreate();
        Assert.SkipUnless(g is not null, "model fixture not present in this environment");

        int qDim = numHeads * headDim;
        int kvDim = numKvHeads * headDim;
        int maxSeqLen = basePos + numQueries + 16;   // slack, never read past kvLen

        var qh = Rand(numQueries * qDim, seed);
        var kh = Rand(maxSeqLen * kvDim, seed + 1);
        var vh = Rand(maxSeqLen * kvDim, seed + 2);

        var gq = g.Upload(qh, TensorShape.D1(numQueries * qDim));
        var gk = g.Upload(kh, TensorShape.D1(maxSeqLen * kvDim));
        var gv = g.Upload(vh, TensorShape.D1(maxSeqLen * kvDim));
        var gRef = g.Allocate(TensorShape.D1(numQueries * qDim));
        var gFlash = g.Allocate(TensorShape.D1(numQueries * qDim));

        g.AttentionBatched(gq, gk, gv, gRef,
            (uint)numHeads, (uint)numKvHeads, (uint)headDim, (uint)basePos, numQueries, (uint)maxSeqLen);
        g.AttentionBatchedFlash(gq, gk, gv, gFlash,
            (uint)numHeads, (uint)numKvHeads, (uint)headDim, (uint)basePos, numQueries, (uint)maxSeqLen);

        var refOut = new float[numQueries * qDim];
        var flashOut = new float[numQueries * qDim];
        g.Download(gRef, refOut);
        g.Download(gFlash, flashOut);

        double maxAbs = 0;
        int worst = -1;
        for (int i = 0; i < refOut.Length; i++)
        {
            Assert.True(float.IsFinite(flashOut[i]),
                $"flash attention produced a non-finite value at {i} " +
                $"(heads={numHeads}/{numKvHeads} hd={headDim} basePos={basePos} q={numQueries})");
            double d = Math.Abs(refOut[i] - flashOut[i]);
            if (d > maxAbs) { maxAbs = d; worst = i; }
        }

        // Attention outputs here are convex combinations of V values drawn from [-1, 1], so they
        // are O(1); 2e-3 is loose enough for the reordered accumulation and far too tight to hide
        // a masking or head-mapping error, which would show up as O(1) divergence.
        Assert.True(maxAbs < 2e-3,
            $"flash vs batched attention diverged: max abs {maxAbs:R} at index {worst} " +
            $"(heads={numHeads}/{numKvHeads} hd={headDim} basePos={basePos} q={numQueries})");

        g.Free(gq); g.Free(gk); g.Free(gv); g.Free(gRef); g.Free(gFlash);
    }

    /// <summary>
    /// Same parity gate for the 16-bit-narrowed KV variant (perf-loop iteration 44):
    /// <c>AttentionBatchedFlash(kvBf16: true)</c> against <see cref="VulkanBackend.AttentionBatchedBf16"/>,
    /// which is what the bf16 path used before the flash variant existed.
    ///
    /// <para>Both sides read the SAME packed cache, so this isolates the flash kernel's tiling and
    /// online softmax from the narrowing itself — a wrong <c>readK</c>/<c>readV</c> word index or
    /// component selection would show as O(1) divergence rather than as quantization noise.</para>
    /// </summary>
    private static void AssertMatchesBf16(
        int numHeads, int numKvHeads, int headDim, int basePos, int numQueries, int seed)
    {
        using var g = TryCreate();
        Assert.SkipUnless(g is not null, "model fixture not present in this environment");

        int qDim = numHeads * headDim;
        int kvDim = numKvHeads * headDim;
        int maxSeqLen = basePos + numQueries + 16;

        var qh = Rand(numQueries * qDim, seed);
        // The narrowed cache is written by KvAppendBf16, so build it through that path rather than
        // packing by hand — the test then cannot disagree with production about the layout.
        var kh = Rand(maxSeqLen * kvDim, seed + 1);
        var vh = Rand(maxSeqLen * kvDim, seed + 2);

        var gq = g.Upload(qh, TensorShape.D1(numQueries * qDim));
        var gkSrc = g.Upload(kh, TensorShape.D1(maxSeqLen * kvDim));
        var gvSrc = g.Upload(vh, TensorShape.D1(maxSeqLen * kvDim));
        // Packed caches: two halves per uint → half the float slots.
        var gk = g.Allocate(TensorShape.D1(maxSeqLen * kvDim / 2));
        var gv = g.Allocate(TensorShape.D1(maxSeqLen * kvDim / 2));
        g.KvAppendBatchedBf16(gkSrc, gvSrc, gk, gv, (uint)kvDim, 0, maxSeqLen, (uint)maxSeqLen);

        var gRef = g.Allocate(TensorShape.D1(numQueries * qDim));
        var gFlash = g.Allocate(TensorShape.D1(numQueries * qDim));

        g.AttentionBatchedBf16(gq, gk, gv, gRef,
            (uint)numHeads, (uint)numKvHeads, (uint)headDim, (uint)basePos, numQueries, (uint)maxSeqLen);
        g.AttentionBatchedFlash(gq, gk, gv, gFlash,
            (uint)numHeads, (uint)numKvHeads, (uint)headDim, (uint)basePos, numQueries, (uint)maxSeqLen,
            kvBf16: true);

        var refOut = new float[numQueries * qDim];
        var flashOut = new float[numQueries * qDim];
        g.Download(gRef, refOut);
        g.Download(gFlash, flashOut);

        double maxAbs = 0;
        int worst = -1;
        for (int i = 0; i < refOut.Length; i++)
        {
            Assert.True(float.IsFinite(flashOut[i]),
                $"bf16 flash attention produced a non-finite value at {i} " +
                $"(heads={numHeads}/{numKvHeads} hd={headDim} basePos={basePos} q={numQueries})");
            double d = Math.Abs(refOut[i] - flashOut[i]);
            if (d > maxAbs) { maxAbs = d; worst = i; }
        }

        Assert.True(maxAbs < 2e-3,
            $"bf16 flash vs bf16 batched attention diverged: max abs {maxAbs:R} at index {worst} " +
            $"(heads={numHeads}/{numKvHeads} hd={headDim} basePos={basePos} q={numQueries})");

        g.Free(gq); g.Free(gkSrc); g.Free(gvSrc); g.Free(gk); g.Free(gv); g.Free(gRef); g.Free(gFlash);
    }

    [Theory]
    [InlineData(8, 8, 64, 0, 1)]         // single query
    [InlineData(8, 8, 64, 0, 16)]        // max queries, one exact tile boundary at 32
    [InlineData(8, 8, 64, 37, 11)]       // mid-sequence base, ragged tile
    [InlineData(8, 2, 64, 100, 8)]       // grouped-query attention
    [InlineData(4, 4, 128, 200, 8)]      // head_dim 128 (MAX_HD)
    public void Bf16MatchesBatchedAttentionBf16(
        int numHeads, int numKvHeads, int headDim, int basePos, int numQueries) =>
        AssertMatchesBf16(numHeads, numKvHeads, headDim, basePos, numQueries, seed: 4242);

    [Fact]
    public void MatchesBatchedAttention_SingleQuery() =>
        AssertMatches(numHeads: 8, numKvHeads: 8, headDim: 64, basePos: 0, numQueries: 1, seed: 11);

    [Fact]
    public void MatchesBatchedAttention_FullTile() =>
        AssertMatches(numHeads: 8, numKvHeads: 8, headDim: 64, basePos: 0, numQueries: 8, seed: 12);

    [Fact]
    public void MatchesBatchedAttention_MaxQueries() =>
        AssertMatches(numHeads: 8, numKvHeads: 8, headDim: 64, basePos: 0, numQueries: 16, seed: 13);

    /// <summary>Query count not a multiple of the 8-wide tile.</summary>
    [Fact]
    public void MatchesBatchedAttention_RaggedQueryCount() =>
        AssertMatches(numHeads: 8, numKvHeads: 8, headDim: 64, basePos: 0, numQueries: 5, seed: 14);

    /// <summary>
    /// Mid-sequence start: every query in the workgroup masks the same tile at a different point,
    /// which is the case a per-workgroup (rather than per-query) kernel is easiest to get wrong.
    /// basePos is deliberately not a multiple of the tile width.
    /// </summary>
    [Fact]
    public void MatchesBatchedAttention_MidSequenceBasePos() =>
        AssertMatches(numHeads: 8, numKvHeads: 8, headDim: 64, basePos: 37, numQueries: 16, seed: 15);

    [Fact]
    public void MatchesBatchedAttention_LongerContext() =>
        AssertMatches(numHeads: 8, numKvHeads: 8, headDim: 64, basePos: 501, numQueries: 16, seed: 16);

    /// <summary>Grouped-query attention — exercises the kv_head = h / (heads/kvHeads) mapping.</summary>
    [Fact]
    public void MatchesBatchedAttention_GroupedQueryAttention() =>
        AssertMatches(numHeads: 8, numKvHeads: 2, headDim: 64, basePos: 23, numQueries: 16, seed: 17);

    /// <summary>The largest head dim the kernel's shared buffers are compiled for.</summary>
    [Fact]
    public void MatchesBatchedAttention_HeadDim128() =>
        AssertMatches(numHeads: 4, numKvHeads: 4, headDim: 128, basePos: 19, numQueries: 16, seed: 18);

    [Fact]
    public void RejectsShapesItCannotServe()
    {
        Assert.False(VulkanBackend.SupportsFlashAttention(headDim: 256, numQueries: 8));
        Assert.False(VulkanBackend.SupportsFlashAttention(headDim: 64, numQueries: 17));
        Assert.False(VulkanBackend.SupportsFlashAttention(headDim: 64, numQueries: 0));
        Assert.True(VulkanBackend.SupportsFlashAttention(headDim: 128, numQueries: 16));
    }
}
