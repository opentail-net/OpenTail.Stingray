namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class Q8PrefillDefaultPolicyTests
{
    [Fact]
    public void ExactInt8Prefill_IsOnByDefault_AndOptOutWithZero()
    {
        // Pure resolver checks only: Q8PrefillEnabled is a process-wide static that other tests set and restore.
        Assert.True(SimdKernels.ResolveQ8PrefillDefault(null));
        Assert.False(SimdKernels.ResolveQ8PrefillDefault("0"));
        Assert.True(SimdKernels.ResolveQ8PrefillDefault("1"));
    }

    [Fact]
    public void FastGemm_IsOptInOnly()
    {
        Assert.Equal(
            Environment.GetEnvironmentVariable("STINGRAY_CPU_PREFILL_FAST_GEMM") == "1",
            SimdKernels.Q8PrefillFastGemm);
    }
}
