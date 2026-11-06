using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// <c>STINGRAY_TP_DEGREE</c> parsing (<see cref="CommonForwardPassFactory.CheckTensorParallelDegree"/>),
/// run as its own test (not through the full CUDA dense construction path, which needs real hardware
/// this project doesn't have) -- see docs/4-performance/gpu/2026-11-06-cuda-tensor-parallelism-plan.md.
/// Not parallel-safe with anything else that reads this env var (none exists yet); resets it in
/// <see cref="Dispose"/> regardless of outcome.
/// </summary>
public sealed class CudaTensorParallelFlagTests : IDisposable
{
    private const string Var = "STINGRAY_TP_DEGREE";

    public void Dispose() => Environment.SetEnvironmentVariable(Var, null);

    [Fact]
    public void Unset_DoesNotThrow()
    {
        Environment.SetEnvironmentVariable(Var, null);
        CommonForwardPassFactory.CheckTensorParallelDegree();
    }

    [Fact]
    public void DegreeOne_DoesNotThrow()
    {
        Environment.SetEnvironmentVariable(Var, "1");
        CommonForwardPassFactory.CheckTensorParallelDegree();
    }

    [Fact]
    public void DegreeTwo_ThrowsNamingWhatIsNotWiredYet()
    {
        Environment.SetEnvironmentVariable(Var, "2");
        var ex = Assert.Throws<NotSupportedException>(CommonForwardPassFactory.CheckTensorParallelDegree);
        Assert.Contains("STINGRAY_TP_DEGREE", ex.Message);
        Assert.Contains("not built yet", ex.Message);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("-1")]
    [InlineData("not-a-number")]
    public void InvalidValue_Throws(string value)
    {
        Environment.SetEnvironmentVariable(Var, value);
        Assert.Throws<NotSupportedException>(CommonForwardPassFactory.CheckTensorParallelDegree);
    }
}
