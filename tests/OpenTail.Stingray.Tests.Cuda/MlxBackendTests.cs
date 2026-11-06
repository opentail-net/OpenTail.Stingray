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

    [Fact]
    public void Silu_WithNoNativeLibraryPresent_FailsPredictably_DoesNotCrash()
    {
        Assert.Throws<DllNotFoundException>(() => MlxOps.Silu(new float[4]));
    }

    [Fact]
    public void MatMul_ShapeMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => MlxOps.MatMul(new float[6], m: 2, k: 3, b: new float[5], n: 2));
    }

    [Fact]
    public void MatMul_ValidShapes_WithNoNativeLibraryPresent_FailsPredictably_DoesNotCrash()
    {
        Assert.Throws<DllNotFoundException>(() => MlxOps.MatMul(new float[6], m: 2, k: 3, b: new float[6], n: 2));
    }

    [Fact]
    public void AddMm_BiasLengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            MlxOps.AddMm(bias: new float[3], a: new float[6], m: 2, k: 3, b: new float[6], n: 2));
    }

    [Fact]
    public void AddMm_ValidShapes_WithNoNativeLibraryPresent_FailsPredictably_DoesNotCrash()
    {
        Assert.Throws<DllNotFoundException>(() =>
            MlxOps.AddMm(bias: new float[2], a: new float[6], m: 2, k: 3, b: new float[6], n: 2));
    }

    [Fact]
    public void Softmax_ShapeMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => MlxOps.Softmax(new float[5], rows: 2, cols: 3));
    }

    [Fact]
    public void Softmax_ValidShapes_WithNoNativeLibraryPresent_FailsPredictably_DoesNotCrash()
    {
        Assert.Throws<DllNotFoundException>(() => MlxOps.Softmax(new float[6], rows: 2, cols: 3));
    }

    [Fact]
    public void Attention_ShapeMismatch_Throws()
    {
        float[] qkv = new float[2 * 3 * 4]; // numHeads=2, seqLen=3, headDim=4
        Assert.Throws<ArgumentException>(() =>
            MlxOps.Attention(new float[5], qkv, qkv, numHeads: 2, seqLen: 3, headDim: 4, scale: 1f, causal: true));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Attention_ValidShapes_WithNoNativeLibraryPresent_FailsPredictably_DoesNotCrash(bool causal)
    {
        float[] qkv = new float[2 * 3 * 4];
        Assert.Throws<DllNotFoundException>(() =>
            MlxOps.Attention(qkv, qkv, qkv, numHeads: 2, seqLen: 3, headDim: 4, scale: 0.5f, causal));
    }

    [Fact]
    public void Rope_PositionsLengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            MlxOps.Rope(new float[3 * 4], seqLen: 3, headDim: 4, dims: 4, traditional: false, baseValue: 10000f, scale: 1f, positions: [0, 1]));
    }

    [Fact]
    public void Rope_OddDims_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            MlxOps.Rope(new float[3 * 4], seqLen: 3, headDim: 4, dims: 3, traditional: false, baseValue: 10000f, scale: 1f, positions: [0, 1, 2]));
    }

    [Fact]
    public void Rope_ValidShapes_WithNoNativeLibraryPresent_FailsPredictably_DoesNotCrash()
    {
        Assert.Throws<DllNotFoundException>(() =>
            MlxOps.Rope(new float[3 * 4], seqLen: 3, headDim: 4, dims: 4, traditional: false, baseValue: 10000f, scale: 1f, positions: [0, 1, 2]));
    }
}
