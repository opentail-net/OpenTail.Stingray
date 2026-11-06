using OpenTail.Stingray.Mlx;

namespace OpenTail.Stingray.Tests.Cuda;

/// <summary>
/// <see cref="MlxBackend.IsAvailable"/> is the only thing this skeleton can actually be checked
/// against here: no Mac exists on this development machine, so <c>libmlxc</c> is never present and
/// this must return <c>false</c> gracefully rather than throw. See
/// docs/4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md.
/// </summary>
public sealed class MlxBackendTests
{
    [Fact]
    public void IsAvailable_WithNoNativeLibraryPresent_ReturnsFalse_DoesNotThrow()
    {
        // This asserts the honest, expected-here outcome, not a mocked one: this machine genuinely
        // has no libmlxc. If a future CI runner or dev machine ever has one, this test's assumption
        // changes and it should be revisited, not silently left green for the wrong reason.
        Assert.False(MlxBackend.IsAvailable());
    }
}

/// <summary>
/// <see cref="MlxOps.RmsNorm"/>: the first real op ported from TensorSharp's MLX backend. The
/// argument-validation logic runs before any native call and is checked directly; the actual
/// mlx-c dispatch cannot be (no Mac here), so that part only checks it fails predictably rather
/// than crashing the process.
/// </summary>
public sealed class MlxOpsTests
{
    [Fact]
    public void RmsNorm_InputLengthMismatch_Throws()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            MlxOps.RmsNorm(new float[5], rows: 2, hiddenDim: 4, weight: [], eps: 1e-6f));
        Assert.Equal("input", ex.ParamName);
    }

    [Fact]
    public void RmsNorm_WeightLengthMismatch_Throws()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            MlxOps.RmsNorm(new float[8], rows: 2, hiddenDim: 4, weight: new float[3], eps: 1e-6f));
        Assert.Equal("weight", ex.ParamName);
    }

    [Fact]
    public void RmsNorm_ValidShapes_WithNoNativeLibraryPresent_FailsPredictably_DoesNotCrash()
    {
        // Honest, expected-here outcome (no libmlxc on this machine), not a mocked success:
        // the very first real mlx-c call (mlx_device_new_type) throws DllNotFoundException.
        Assert.Throws<DllNotFoundException>(() =>
            MlxOps.RmsNorm(new float[8], rows: 2, hiddenDim: 4, weight: new float[4], eps: 1e-6f));
    }
}
