namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// <see cref="RunCommand.TryValidateCpuMoeFlags"/> validates the llama.cpp-style MoE placement flags
/// (issue #80). It is validation only: <c>--cpu-moe</c> itself travels in the ExecutionRequest/ExecutionPlan
/// and is applied by RuntimeInstance, so the CLI must never write STINGRAY_CPU_MOE.
/// </summary>
public sealed class CpuMoeFlagsTests
{
    [Fact]
    public void NoPartialSplit_IsAccepted_AndWritesNoEnvironment()
    {
        string? before = Environment.GetEnvironmentVariable("STINGRAY_CPU_MOE");
        bool ok = RunCommand.TryValidateCpuMoeFlags(nCpuMoe: null, out string? error);

        Assert.True(ok);
        Assert.Null(error);
        Assert.Equal(before, Environment.GetEnvironmentVariable("STINGRAY_CPU_MOE"));
    }

    [Theory]
    [InlineData(20)]
    [InlineData(0)]
    public void PartialSplit_IsRejected_WithRationale(int n)
    {
        bool ok = RunCommand.TryValidateCpuMoeFlags(nCpuMoe: n, out string? error);

        Assert.False(ok);
        Assert.NotNull(error);
        Assert.Contains("--cpu-moe", error);
        Assert.Contains("#80", error);
    }
}
