using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Vulkan.Fast;

/// <summary>
/// The Vulkan KV-cache dtype default (perf-loop iteration 46).
///
/// <para>bf16 became the Vulkan default because it measured strictly better than the previous
/// fp32 default on every axis at long context: prefill 47.6 → 52.2 t/s, decode 6.1 → 9.4 t/s,
/// +0.023% perplexity, and it keeps the full context instead of letting SnapKV auto-evict it to a
/// 1024-position budget.</para>
///
/// <para><b>What these tests actually protect.</b> Flipping a default turns three previously
/// unreachable <c>NotSupportedException</c>s into things a user who never mentioned a KV dtype
/// could hit — TurboQuant, an explicit SnapKV budget, an odd head dim. The fix is that a
/// <i>defaulted</i> narrowing falls back to fp32 while an <i>explicit</i> one still throws, and
/// both the chooser and the constructor's safety net route through the single
/// <see cref="GpuForwardPass.CanNarrowKv"/> predicate so the two cannot drift apart. These tests
/// pin that predicate, because the end-to-end CLI checks cannot reach the TurboQuant branch on the
/// reference model (head dim 64 rejects TQ for an unrelated reason).</para>
/// </summary>
public sealed class GpuForwardPassKvDefaultTests
{
    private static ModelHyperparams Hp(int headDim = 64, int numKvHeads = 8) => new()
    {
        VocabSize = 32000,
        ContextLength = 4096,
        EmbeddingDim = 2048,
        NumLayers = 24,
        NumHeads = 32,
        NumKvHeads = numKvHeads,
        IntermediateDim = 8192,
        HeadDim = headDim,
    };

    private static SnapKvConfig NoSnapKv => new(Budget: 0, Window: 64, Recency: 64, IsBudgetExplicit: false);
    private static SnapKvConfig ExplicitSnapKv(int budget) =>
        new(Budget: budget, Window: 64, Recency: 64, IsBudgetExplicit: true);

    [Fact]
    public void DefaultsToBf16OnAnOrdinaryModel() =>
        Assert.Equal(DType.BFloat16, GpuForwardPass.ChooseDefaultKvDType(Hp(), tqEnabled: false, NoSnapKv));

    /// <summary>
    /// TurboQuant owns the KV quantization — the two cannot both narrow the store. Before the
    /// default flip this combination threw; a user running <c>--tq</c> and nothing else must get
    /// fp32, not an exception.
    /// </summary>
    [Fact]
    public void FallsBackToFp32WhenTurboQuantIsOn() =>
        Assert.Equal(DType.Float32, GpuForwardPass.ChooseDefaultKvDType(Hp(), tqEnabled: true, NoSnapKv));

    /// <summary>
    /// SnapKvScore/KvCompact read the cache as fp32, so an explicitly requested budget wins over
    /// the dtype default. (An AUTO budget does not: the narrowed store is itself the memory
    /// strategy, which is the deliberate behaviour change this default carries.)
    /// </summary>
    [Fact]
    public void FallsBackToFp32WhenASnapKvBudgetWasExplicitlyRequested() =>
        Assert.Equal(DType.Float32, GpuForwardPass.ChooseDefaultKvDType(Hp(), tqEnabled: false, ExplicitSnapKv(1024)));

    [Fact]
    public void AnExplicitZeroSnapKvBudgetDoesNotBlockBf16() =>
        Assert.Equal(DType.BFloat16, GpuForwardPass.ChooseDefaultKvDType(Hp(), tqEnabled: false, ExplicitSnapKv(0)));

    /// <summary>
    /// bf16 packs two halves per uint, so a KV head must start on a word boundary — true iff the
    /// head dim is even. No supported model has an odd head dim, but the default must not be the
    /// thing that discovers otherwise at runtime.
    /// </summary>
    [Fact]
    public void FallsBackToFp32OnAnOddHeadDim() =>
        Assert.Equal(DType.Float32, GpuForwardPass.ChooseDefaultKvDType(Hp(headDim: 65), tqEnabled: false, NoSnapKv));

    /// <summary>
    /// The chooser must never return something the constructor's safety net would reject — that
    /// equivalence is the whole point of routing both through one predicate.
    /// </summary>
    [Theory]
    [InlineData(64, 8, false, false)]
    [InlineData(64, 8, true, false)]
    [InlineData(64, 8, false, true)]
    [InlineData(65, 8, false, false)]
    [InlineData(128, 2, true, true)]
    public void ChosenDefaultIsAlwaysAcceptedByTheSafetyNet(
        int headDim, int numKvHeads, bool tq, bool explicitSnapKv)
    {
        var hp = Hp(headDim, numKvHeads);
        var snap = explicitSnapKv ? ExplicitSnapKv(1024) : NoSnapKv;

        DType chosen = GpuForwardPass.ChooseDefaultKvDType(hp, tq, snap);
        if (chosen == DType.Float32) return;   // fp32 is never narrowed, nothing to check

        Assert.True(GpuForwardPass.CanNarrowKv(chosen, hp, tq, snap),
            $"ChooseDefaultKvDType returned {chosen} for headDim={headDim} tq={tq} " +
            $"explicitSnapKv={explicitSnapKv}, but CanNarrowKv rejects it — the chooser and the " +
            "constructor's fallback have drifted apart.");
    }

    /// <summary>q8_0 is never chosen as a default; it stays opt-in via STINGRAY_KV_DTYPE.</summary>
    [Fact]
    public void Q8_0IsNeverTheDefault()
    {
        foreach (int headDim in new[] { 64, 128, 256 })
            Assert.NotEqual(DType.Q8_0, GpuForwardPass.ChooseDefaultKvDType(Hp(headDim), tqEnabled: false, NoSnapKv));
    }

    /// <summary>
    /// q8_0 requires kvDim % 32 == 0 for its per-32-element blocks to align to row boundaries.
    /// Checked here because the safety net covers explicit requests too.
    /// </summary>
    [Fact]
    public void Q8_0IsRejectedWhenKvDimIsNotBlockAligned()
    {
        Assert.False(GpuForwardPass.CanNarrowKv(DType.Q8_0, Hp(headDim: 4, numKvHeads: 1), tqEnabled: false, NoSnapKv));
        Assert.True(GpuForwardPass.CanNarrowKv(DType.Q8_0, Hp(headDim: 64, numKvHeads: 8), tqEnabled: false, NoSnapKv));
    }
}
