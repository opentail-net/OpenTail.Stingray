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
            MlxOps.Attention(new float[5], qkv, qkv, numQueryHeads: 2, numKvHeads: 2, querySeqLen: 3, kvSeqLen: 3, headDim: 4, scale: 1f, causal: true));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Attention_ValidShapes_WithNoNativeLibraryPresent_FailsPredictably_DoesNotCrash(bool causal)
    {
        float[] qkv = new float[2 * 3 * 4];
        Assert.Throws<DllNotFoundException>(() =>
            MlxOps.Attention(qkv, qkv, qkv, numQueryHeads: 2, numKvHeads: 2, querySeqLen: 3, kvSeqLen: 3, headDim: 4, scale: 0.5f, causal));
    }

    [Fact]
    public void Attention_Gqa_FewerKvHeadsThanQueryHeads_WithNoNativeLibraryPresent_FailsPredictably_DoesNotCrash()
    {
        // 4 query heads, 2 KV heads (2:1 GQA grouping) -- confirms the shape validation accepts
        // numKvHeads < numQueryHeads and reaches the real native call rather than rejecting it.
        float[] query = new float[4 * 3 * 2];
        float[] kv = new float[2 * 3 * 2];
        Assert.Throws<DllNotFoundException>(() =>
            MlxOps.Attention(query, kv, kv, numQueryHeads: 4, numKvHeads: 2, querySeqLen: 3, kvSeqLen: 3, headDim: 2, scale: 0.5f, causal: true));
    }

    [Fact]
    public void Attention_NegativeQueryOffset_Throws()
    {
        float[] qkv = new float[2 * 3 * 4];
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            MlxOps.Attention(qkv, qkv, qkv, numQueryHeads: 2, numKvHeads: 2, querySeqLen: 3, kvSeqLen: 3, headDim: 4, scale: 1f, causal: true, queryOffset: -1));
    }

    [Fact]
    public void Attention_QueryOffsetWithoutCausal_Throws()
    {
        float[] qkv = new float[2 * 3 * 4];
        Assert.Throws<ArgumentException>(() =>
            MlxOps.Attention(qkv, qkv, qkv, numQueryHeads: 2, numKvHeads: 2, querySeqLen: 3, kvSeqLen: 3, headDim: 4, scale: 1f, causal: false, queryOffset: 2));
    }

    [Fact]
    public void Attention_ContinuationWithQueryOffset_WithNoNativeLibraryPresent_FailsPredictably_DoesNotCrash()
    {
        // Multi-token cache continuation: 2 new query positions attending over a 5-position KV
        // range (3 already cached + the 2 new ones) starting at offset 3 -- confirms the offset
        // mask construction path (arange/reshape/subtract/greater) is reached, not just the
        // plain "causal" string path.
        float[] query = new float[2 * 2 * 4]; // numHeads=2, querySeqLen=2, headDim=4
        float[] kv = new float[2 * 5 * 4];    // numHeads=2, kvSeqLen=5, headDim=4
        Assert.Throws<DllNotFoundException>(() =>
            MlxOps.Attention(query, kv, kv, numQueryHeads: 2, numKvHeads: 2, querySeqLen: 2, kvSeqLen: 5, headDim: 4, scale: 0.5f, causal: true, queryOffset: 3));
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

    [Fact]
    public void Multiply_LengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => MlxOps.Multiply(new float[3], new float[4]));
    }

    [Fact]
    public void Multiply_ValidShapes_WithNoNativeLibraryPresent_FailsPredictably_DoesNotCrash()
    {
        Assert.Throws<DllNotFoundException>(() => MlxOps.Multiply(new float[4], new float[4]));
    }

    [Fact]
    public void SplitHeads_ShapeMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => MlxOps.SplitHeads(new float[5], seqLen: 2, numHeads: 2, headDim: 4));
    }

    [Fact]
    public void SplitHeads_ValidShapes_WithNoNativeLibraryPresent_FailsPredictably_DoesNotCrash()
    {
        Assert.Throws<DllNotFoundException>(() => MlxOps.SplitHeads(new float[2 * 2 * 4], seqLen: 2, numHeads: 2, headDim: 4));
    }

    [Fact]
    public void MergeHeads_ShapeMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => MlxOps.MergeHeads(new float[5], numHeads: 2, seqLen: 2, headDim: 4));
    }

    [Fact]
    public void MergeHeads_ValidShapes_WithNoNativeLibraryPresent_FailsPredictably_DoesNotCrash()
    {
        Assert.Throws<DllNotFoundException>(() => MlxOps.MergeHeads(new float[2 * 2 * 4], numHeads: 2, seqLen: 2, headDim: 4));
    }

    [Fact]
    public void MoeRouterTopKSoftmax_LogitsLengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => MlxOps.MoeRouterTopKSoftmax(new float[5], numExperts: 8, k: 2));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(-1)]
    public void MoeRouterTopKSoftmax_KOutOfRange_Throws(int k)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => MlxOps.MoeRouterTopKSoftmax(new float[8], numExperts: 8, k: k));
    }

    [Fact]
    public void MoeRouterTopKSoftmax_ValidShapes_WithNoNativeLibraryPresent_FailsPredictably_DoesNotCrash()
    {
        Assert.Throws<DllNotFoundException>(() => MlxOps.MoeRouterTopKSoftmax(new float[8], numExperts: 8, k: 2));
    }

    [Fact]
    public void GatherRows_SrcLengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => MlxOps.GatherRows(new float[5], srcRows: 2, cols: 3, indices: [0, 1]));
    }

    [Fact]
    public void GatherRows_ValidShapes_WithNoNativeLibraryPresent_FailsPredictably_DoesNotCrash()
    {
        Assert.Throws<DllNotFoundException>(() => MlxOps.GatherRows(new float[2 * 3], srcRows: 2, cols: 3, indices: [1, 0, 1]));
    }
}
