using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// Tests for the Vulkan matmul path switch and its instrumentation.
/// </summary>
/// <remarks>
/// <para>No GPU required — that is the point of putting the switch and the counters in a pure type.
/// The number these exist to protect is <see cref="VulkanMatMulStats.TokensPerDispatch"/>: it
/// decides whether a tiled-GEMM Path 2 is worth writing, so it must be right before it is trusted,
/// and it can be checked against hand-computed counts without ever touching a device.</para>
///
/// <para><see cref="BytesPerToken_IsNotDerivableFromTokenDispatches"/> pins a defect the first real
/// measurement found: dividing weight bytes by the summed <c>nTok</c> understated the cost by the
/// number of matmuls in the model (168x on a 24-layer model). The counters were wired up, run on
/// hardware, and checked against an independently computed model size before any of this was
/// believed — the instrument was measured before it was trusted.</para>
///
/// <para>Statics are safe to mutate here because the assembly sets
/// <c>parallelizeTestCollections: false</c>; each test still restores what it changed, so ordering
/// cannot leak state into the Vulkan tests that run real dispatches.</para>
/// </remarks>
public sealed class VulkanMatMulPathTests
{
    /// <summary>Restores path + stats-enable + counters, whatever a test did to them.</summary>
    private static Restore Scoped()
    {
        var r = new Restore(VulkanMatMulPathConfig.Current, VulkanMatMulPathConfig.StatsEnabled);
        VulkanMatMulStats.Reset();
        return r;
    }

    private readonly struct Restore(VulkanMatMulPath path, bool stats) : IDisposable
    {
        public void Dispose()
        {
            VulkanMatMulPathConfig.Current = path;
            VulkanMatMulPathConfig.StatsEnabled = stats;
            VulkanMatMulStats.Reset();
        }
    }

    // ── Path selection ───────────────────────────────────────────────────────

    /// <summary>
    /// Path 1 is the default, unlike the CPU's GemmPath which defaults to Path 2. Path 2 there
    /// earned the default by measuring faster end-to-end and no worse on perplexity; here it does
    /// not exist yet, so flipping this would silently route every dispatch through a decline.
    /// </summary>
    [Fact]
    public void Default_IsPath1()
    {
        using var _ = Scoped();
        Environment.SetEnvironmentVariable("STINGRAY_VULKAN_MM_PATH", null);
        Assert.Equal(VulkanMatMulPath.Path1, VulkanMatMulPathConfig.ReadFromEnvironment());
    }

    [Theory]
    [InlineData("2", VulkanMatMulPath.Path2)]
    [InlineData("path2", VulkanMatMulPath.Path2)]
    [InlineData("PATH2", VulkanMatMulPath.Path2)]
    [InlineData(" 2 ", VulkanMatMulPath.Path2)]
    [InlineData("1", VulkanMatMulPath.Path1)]
    [InlineData("path1", VulkanMatMulPath.Path1)]
    [InlineData("", VulkanMatMulPath.Path1)]
    [InlineData("   ", VulkanMatMulPath.Path1)]
    [InlineData("banana", VulkanMatMulPath.Path1)]   // unparseable must NOT silently enable Path 2
    public void EnvVar_Parses(string value, VulkanMatMulPath expected)
    {
        using var _ = Scoped();
        try
        {
            Environment.SetEnvironmentVariable("STINGRAY_VULKAN_MM_PATH", value);
            Assert.Equal(expected, VulkanMatMulPathConfig.ReadFromEnvironment());
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_VULKAN_MM_PATH", null);
        }
    }

    [Fact]
    public void UsePath2_TracksCurrent()
    {
        using var _ = Scoped();
        VulkanMatMulPathConfig.Current = VulkanMatMulPath.Path1;
        Assert.False(VulkanMatMulPathConfig.UsePath2);
        VulkanMatMulPathConfig.Current = VulkanMatMulPath.Path2;
        Assert.True(VulkanMatMulPathConfig.UsePath2);
    }

    /// <summary>
    /// The env var is registered, so <c>--help</c> / diagnostics list it and a typo in it is
    /// reported rather than silently ignored.
    /// </summary>
    [Theory]
    [InlineData("STINGRAY_VULKAN_MM_PATH")]
    [InlineData("STINGRAY_VULKAN_MM_STATS")]
    public void EnvVars_AreRegistered(string name)
        => Assert.True(KnownEnvironmentVariables.All.Contains(name), $"{name} is not registered");

    // ── Weight-byte accounting ───────────────────────────────────────────────

    /// <summary>
    /// Block sizes are the on-disk GGUF layouts the shaders index directly. Getting these wrong
    /// scales every conclusion about Path 2's prize by the same factor, so they are pinned.
    /// </summary>
    [Fact]
    public void WeightBytesFor_UsesGgufBlockSizes()
    {
        // 4096x4096 = 16,777,216 elements = 65,536 blocks of 256.
        const long elems = 4096L * 4096L;
        const long blocks = elems / 256;
        Assert.Equal(blocks * 144, VulkanMatMulStats.WeightBytesFor(4096, 4096, DType.Q4_K));
        Assert.Equal(blocks * 210, VulkanMatMulStats.WeightBytesFor(4096, 4096, DType.Q6_K));
        Assert.Equal(elems * 4, VulkanMatMulStats.WeightBytesFor(4096, 4096, DType.Float32));
    }

    [Fact]
    public void WeightBytesFor_Q4K_IsSmallerThanQ6K_AndBothBeatF32()
    {
        long q4 = VulkanMatMulStats.WeightBytesFor(1024, 1024, DType.Q4_K);
        long q6 = VulkanMatMulStats.WeightBytesFor(1024, 1024, DType.Q6_K);
        long f32 = VulkanMatMulStats.WeightBytesFor(1024, 1024, DType.Float32);
        Assert.True(q4 < q6, $"Q4_K ({q4}) must be smaller than Q6_K ({q6})");
        Assert.True(q6 < f32, $"Q6_K ({q6}) must be smaller than F32 ({f32})");
    }

    // ── Counting ─────────────────────────────────────────────────────────────

    /// <summary>Counting off by default: an unmeasured run must be byte-identical to before.</summary>
    [Fact]
    public void Stats_RecordNothing_WhenDisabled()
    {
        using var _ = Scoped();
        VulkanMatMulPathConfig.StatsEnabled = false;

        VulkanMatMulStats.RecordPath1(16, 4096, 4096, DType.Q4_K);
        VulkanMatMulStats.RecordFallback(16, 4096, 4096, DType.Float32);
        VulkanMatMulStats.RecordPath2Decline();

        Assert.Equal(0, VulkanMatMulStats.Path1Dispatches);
        Assert.Equal(0, VulkanMatMulStats.FallbackDispatches);
        Assert.Equal(0, VulkanMatMulStats.Path2Declines);
        Assert.Equal(0, VulkanMatMulStats.WeightBytes);
        Assert.Equal(0d, VulkanMatMulStats.TokensPerDispatch);
    }

    [Fact]
    public void Path1_ChargesWeightsOncePerDispatch()
    {
        using var _ = Scoped();
        VulkanMatMulPathConfig.StatsEnabled = true;

        VulkanMatMulStats.RecordPath1(16, 4096, 4096, DType.Q4_K);

        long once = VulkanMatMulStats.WeightBytesFor(4096, 4096, DType.Q4_K);
        Assert.Equal(1, VulkanMatMulStats.Path1Dispatches);
        Assert.Equal(16, VulkanMatMulStats.AmortizedTokenDispatches);
        Assert.Equal(once, VulkanMatMulStats.WeightBytes);
        Assert.Equal(16d, VulkanMatMulStats.TokensPerDispatch, 6);
    }

    /// <summary>
    /// Reproduces the real 931-token Vulkan prefill (24 layers x 7 trunk matmuls, chunked at 16)
    /// and pins the distinction that the first hardware run exposed: summed <c>nTok</c> is
    /// token-<i>dispatches</i>, and dividing weight bytes by it understates cost by the matmul
    /// count. <see cref="VulkanMatMulStats.WeightBytesPerToken"/> takes the token count explicitly
    /// precisely so this cannot be got wrong silently again.
    /// </summary>
    [Fact]
    public void BytesPerToken_IsNotDerivableFromTokenDispatches()
    {
        using var _ = Scoped();
        VulkanMatMulPathConfig.StatsEnabled = true;

        const int promptTokens = 931;
        const int chunk = 16;
        const int matmulsPerChunk = 24 * 7;

        for (int off = 0; off < promptTokens; off += chunk)
        {
            int take = Math.Min(chunk, promptTokens - off);
            for (int m = 0; m < matmulsPerChunk; m++)
                VulkanMatMulStats.RecordPath1(take, 2048, 2048, DType.Q4_K);
        }

        int chunks = (promptTokens + chunk - 1) / chunk;
        Assert.Equal(chunks * matmulsPerChunk, VulkanMatMulStats.TotalDispatches);
        Assert.Equal((long)promptTokens * matmulsPerChunk, VulkanMatMulStats.TotalTokenDispatches);

        // Amortization is bounded by the chunk size and is NOT inflated by the matmul count.
        Assert.InRange(VulkanMatMulStats.TokensPerDispatch, 15.0, 16.0);

        // The trap: token-dispatches as a denominator is 168x too large.
        double wrong = (double)VulkanMatMulStats.WeightBytes / VulkanMatMulStats.TotalTokenDispatches;
        double right = VulkanMatMulStats.WeightBytesPerToken(promptTokens);
        Assert.Equal(matmulsPerChunk, right / wrong, 3);

        // And the honest figure is one full pass over the trunk per chunk, spread over the prompt.
        long trunkBytesPerChunk = matmulsPerChunk * VulkanMatMulStats.WeightBytesFor(2048, 2048, DType.Q4_K);
        Assert.Equal((double)chunks * trunkBytesPerChunk / promptTokens, right, 3);
    }

    /// <summary>A zero/unknown token count must omit the figure, not fabricate one.</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void WeightBytesPerToken_IsZero_WhenTokenCountUnknown(int promptTokens)
    {
        using var _ = Scoped();
        VulkanMatMulPathConfig.StatsEnabled = true;
        VulkanMatMulStats.RecordPath1(16, 4096, 4096, DType.Q4_K);
        Assert.Equal(0d, VulkanMatMulStats.WeightBytesPerToken(promptTokens));
        Assert.DoesNotContain("MiB/token", VulkanMatMulStats.Report(promptTokens));
    }

    /// <summary>
    /// The fallback branch loops <c>nTok</c> single-row matvecs over the same matrix, so it must be
    /// charged <c>nTok</c> full reads. Charging it once would hide exactly the cliff these counters
    /// exist to expose.
    /// </summary>
    [Fact]
    public void Fallback_ChargesWeightsOncePerToken()
    {
        using var _ = Scoped();
        VulkanMatMulPathConfig.StatsEnabled = true;

        VulkanMatMulStats.RecordFallback(16, 4096, 4096, DType.Q4_K);

        long once = VulkanMatMulStats.WeightBytesFor(4096, 4096, DType.Q4_K);
        Assert.Equal(1, VulkanMatMulStats.FallbackDispatches);
        Assert.Equal(16, VulkanMatMulStats.FallbackTokenDispatches);
        Assert.Equal(0, VulkanMatMulStats.AmortizedTokenDispatches);
        // Per token, the fallback costs a whole matrix read — 16x what Path 1 costs.
        Assert.Equal(once * 16, VulkanMatMulStats.WeightBytes);
        Assert.Equal(once, VulkanMatMulStats.WeightBytesPerToken(16), 6);
    }

    /// <summary>
    /// The headline claim, in miniature: chunking at 16 caps amortization at 16 regardless of
    /// prompt length, where a GEMM's amortization rises with it. This is what makes the metric
    /// worth reading before any kernel is written.
    /// </summary>
    [Theory]
    [InlineData(160)]
    [InlineData(1600)]
    [InlineData(2432)]
    public void Path1_Amortization_DoesNotImproveWithPromptLength(int promptTokens)
    {
        using var _ = Scoped();
        VulkanMatMulPathConfig.StatsEnabled = true;

        const int chunk = 16;
        for (int off = 0; off < promptTokens; off += chunk)
            VulkanMatMulStats.RecordPath1(Math.Min(chunk, promptTokens - off), 4096, 4096, DType.Q4_K);

        long once = VulkanMatMulStats.WeightBytesFor(4096, 4096, DType.Q4_K);
        Assert.Equal(promptTokens, VulkanMatMulStats.TotalTokenDispatches);
        Assert.Equal(chunk, VulkanMatMulStats.TokensPerDispatch, 6);   // capped, never better
        Assert.Equal(once / (double)chunk, VulkanMatMulStats.WeightBytesPerToken(promptTokens), 6);

        // A true GEMM streaming the matrix once for the whole prompt would report this instead —
        // the ratio between them is the entire prize Path 2 is chasing.
        double gemmFloor = once / (double)promptTokens;
        Assert.True(VulkanMatMulStats.WeightBytesPerToken(promptTokens) > gemmFloor,
            "Path 1 must cost strictly more per token than a single-pass GEMM would");
    }

    [Fact]
    public void Declines_AreCounted_Separately_FromDispatches()
    {
        using var _ = Scoped();
        VulkanMatMulPathConfig.StatsEnabled = true;

        VulkanMatMulStats.RecordPath2Decline();
        VulkanMatMulStats.RecordPath2Decline();
        VulkanMatMulStats.RecordPath1(8, 1024, 1024, DType.Q6_K);

        Assert.Equal(2, VulkanMatMulStats.Path2Declines);
        Assert.Equal(0, VulkanMatMulStats.Path2Dispatches);
        Assert.Equal(1, VulkanMatMulStats.Path1Dispatches);
    }

    [Fact]
    public void Reset_ClearsEverything()
    {
        using var _ = Scoped();
        VulkanMatMulPathConfig.StatsEnabled = true;

        VulkanMatMulStats.RecordPath1(16, 4096, 4096, DType.Q4_K);
        VulkanMatMulStats.RecordFallback(4, 512, 512, DType.Float32);
        VulkanMatMulStats.RecordPath2Decline();
        Assert.True(VulkanMatMulStats.WeightBytes > 0);

        VulkanMatMulStats.Reset();

        Assert.Equal(0, VulkanMatMulStats.Path1Dispatches);
        Assert.Equal(0, VulkanMatMulStats.Path2Dispatches);
        Assert.Equal(0, VulkanMatMulStats.Path2Declines);
        Assert.Equal(0, VulkanMatMulStats.FallbackDispatches);
        Assert.Equal(0, VulkanMatMulStats.AmortizedTokenDispatches);
        Assert.Equal(0, VulkanMatMulStats.FallbackTokenDispatches);
        Assert.Equal(0, VulkanMatMulStats.WeightBytes);
        Assert.Equal(0, VulkanMatMulStats.TotalTokenDispatches);
        Assert.Equal(0, VulkanMatMulStats.TotalDispatches);
    }

    [Fact]
    public void Report_NamesThePath_AndSurvivesEmptyState()
    {
        using var _ = Scoped();
        VulkanMatMulPathConfig.Current = VulkanMatMulPath.Path1;

        string empty = VulkanMatMulStats.Report(0);   // no divide-by-zero on an untouched run
        Assert.Contains("Path1", empty);
        Assert.Contains("tokens/dispatch", empty);

        VulkanMatMulPathConfig.StatsEnabled = true;
        VulkanMatMulStats.RecordPath1(16, 4096, 4096, DType.Q4_K);
        Assert.Contains("path1=1", VulkanMatMulStats.Report(16));
    }
}
