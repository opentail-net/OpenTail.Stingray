namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Process-global isolation: execution settings are instance state handed to a constructor, not process state, so two
/// instances configured differently in one process cannot influence each other (the failure mode of the old static,
/// environment-backed <c>PagedKvCache</c> / <c>WarmPinConfig</c> flags).
/// </summary>
public sealed unsafe class InstanceLocalSettingsTests
{
    // 1.001f is not representable in BF16; round-to-nearest-even gives 1.0f.
    private const float NotBf16 = 1.001f;

    private static void AppendToken(PagedKvCache cache, float k)
    {
        float[] ks = [k, k, k, k];
        float[] vs = [k, k, k, k];
        cache.Append(0, ks, vs);
        cache.IncrementPosition();
    }

    [Fact]
    public void KvBf16Rounding_IsPerInstance_AndConstructingBDoesNotChangeA()
    {
        using var a = new PagedKvCache(1, 1, 4, roundBf16: true);
        using var b = new PagedKvCache(1, 1, 4, roundBf16: false);   // built AFTER A, with the opposite setting

        AppendToken(a, NotBf16);
        AppendToken(b, NotBf16);
        AppendToken(a, NotBf16);   // A written again after B exists

        Assert.Equal(1.0f, a.KeyAt(0, 0)[0]);
        Assert.Equal(1.0f, a.KeyAt(0, 1)[0]);
        Assert.Equal(NotBf16, b.KeyAt(0, 0)[0]);
    }

    [Fact]
    public void KvBf16AutoNarrowThreshold_IsPerInstance()
    {
        using var early = new PagedKvCache(1, 1, 4, autoBf16: true, autoMinTokens: 2);
        using var late = new PagedKvCache(1, 1, 4, autoBf16: true, autoMinTokens: 6);

        for (int i = 0; i < 3; i++)
        {
            AppendToken(early, 1f);
            AppendToken(late, 1f);
        }

        Assert.True(early.IsBf16Store);
        Assert.False(late.IsBf16Store);
    }

    [Fact]
    public void EngineSettings_FromTwoPlans_AreIndependentObjects()
    {
        var a = new EngineSettings(
            new EngineTuning(new KvSettings("bf16", KvStoreMode.Bf16, 512), new SpeculationSettings(MtpEnabled: false),
                new PrefillSettings(PrefixSlots: 2), default),
            new MoePlan(true, CpuMoe: true, WarmPin: 3));
        var b = new EngineSettings(EngineTuning.Default, new MoePlan(true, CpuMoe: false, WarmPin: 9));

        Assert.True(a.Kv.RoundBf16);
        Assert.False(b.Kv.RoundBf16);
        Assert.False(a.Speculation.MtpEnabled);
        Assert.True(b.Speculation.MtpEnabled);
        Assert.Equal(2, a.Prefill.PrefixSlots);
        Assert.Equal(0, b.Prefill.PrefixSlots);
        Assert.Equal(true, a.Moe.CpuMoe);
        Assert.Equal(false, b.Moe.CpuMoe);
        Assert.Equal(3, a.Moe.WarmPin);   // reading B changed nothing
    }

    [Fact]
    public void WarmPinResolution_UsesTheInstancesMoePlan_NotTheEnvironment()
    {
        string? saved = Environment.GetEnvironmentVariable("STINGRAY_MOE_WARMPIN");
        try
        {
            Environment.SetEnvironmentVariable("STINGRAY_MOE_WARMPIN", "99");
            Assert.Equal(8, WarmPinConfig.ResolvePerLayer(new MoePlan(true), 8, 64, 8, 16));      // env ignored
            Assert.Equal(5, WarmPinConfig.ResolvePerLayer(new MoePlan(true, WarmPin: 5), 8, 64, 8, 16));
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_MOE_WARMPIN", saved);
        }
    }
}
