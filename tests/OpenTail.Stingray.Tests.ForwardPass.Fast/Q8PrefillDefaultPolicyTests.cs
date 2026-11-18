namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class Q8PrefillDefaultPolicyTests
{
    [Fact]
    public void BatchedQ8Prefill_IsOptInToPreservePerTokenNumerics()
    {
        Assert.Equal(
            SimdKernels.ResolveQ8PrefillDefault(Environment.GetEnvironmentVariable("STINGRAY_CPU_PREFILL_Q8")),
            SimdKernels.Q8PrefillEnabled);
        Assert.False(SimdKernels.ResolveQ8PrefillDefault(null));
        Assert.False(SimdKernels.ResolveQ8PrefillDefault("0"));
        Assert.True(SimdKernels.ResolveQ8PrefillDefault("1"));
        Assert.False(SimdKernels.ResolveQ8PrefillDefault("true"));
    }
}
