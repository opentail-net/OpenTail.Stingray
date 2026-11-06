using OpenTail.Stingray.Mlx;

namespace OpenTail.Stingray.Tests.Cuda;

/// <summary>
/// <see cref="MlxDenseLayer.Forward"/>: the first composed forward-pass building block, wiring all
/// seven ported MLX ops into one transformer block. No Mac exists on this development machine, so the
/// only thing checkable end to end is that a structurally valid call reaches the first real native
/// call and fails predictably there, rather than an earlier bug in the composition throwing something
/// else first (a wrong shape passed between ops, for instance, would throw ArgumentException from a
/// LATER op instead of DllNotFoundException from the first one). See
/// docs/4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md.
/// </summary>
public sealed class MlxDenseLayerTests
{
    private static MlxDenseLayer MakeLayer(int hiddenDim = 4, int numHeads = 1, int headDim = 4, int intermDim = 8, int? numKvHeads = null)
    {
        int kvHeads = numKvHeads ?? numHeads;
        return new()
        {
            AttnNormWeight = new float[hiddenDim],
            Wq = new float[hiddenDim * numHeads * headDim],
            Wk = new float[hiddenDim * kvHeads * headDim],
            Wv = new float[hiddenDim * kvHeads * headDim],
            Wo = new float[numHeads * headDim * hiddenDim],
            FfnNormWeight = new float[hiddenDim],
            WGate = new float[hiddenDim * intermDim],
            WUp = new float[hiddenDim * intermDim],
            WDown = new float[intermDim * hiddenDim],
            HiddenDim = hiddenDim,
            NumHeads = numHeads,
            NumKvHeads = kvHeads,
            HeadDim = headDim,
            IntermDim = intermDim,
        };
    }

    [Fact]
    public void Forward_MismatchedKvHeads_ThrowsNotSupported()
    {
        var layer = MakeLayer(hiddenDim: 4, numHeads: 2, headDim: 4, numKvHeads: 1);
        Assert.Throws<NotSupportedException>(() => layer.Forward(new float[3 * 4], seqLen: 3, positions: [0, 1, 2]));
    }

    [Fact]
    public void Forward_MultiHead_ValidShapes_WithNoNativeLibraryPresent_ReachesFirstNativeCall_FailsPredictably()
    {
        var layer = MakeLayer(hiddenDim: 8, numHeads: 2, headDim: 4, intermDim: 16);
        Assert.Throws<DllNotFoundException>(() => layer.Forward(new float[3 * 8], seqLen: 3, positions: [0, 1, 2]));
    }

    [Fact]
    public void ForwardDecode_WrongHiddenLength_Throws()
    {
        var layer = MakeLayer(hiddenDim: 8, numHeads: 2, headDim: 4);
        var cache = new MlxKvCache(numHeads: 2, maxSeqLen: 4, headDim: 4);
        Assert.Throws<ArgumentException>(() => layer.ForwardDecode(new float[5], position: 0, cache));
    }

    [Fact]
    public void ForwardDecode_CacheShapeMismatch_Throws()
    {
        var layer = MakeLayer(hiddenDim: 8, numHeads: 2, headDim: 4);
        var wrongCache = new MlxKvCache(numHeads: 1, maxSeqLen: 4, headDim: 4); // numHeads should be 2
        Assert.Throws<ArgumentException>(() => layer.ForwardDecode(new float[8], position: 0, wrongCache));
    }

    [Fact]
    public void ForwardDecode_MismatchedKvHeads_ThrowsNotSupported()
    {
        var layer = MakeLayer(hiddenDim: 8, numHeads: 2, headDim: 4, numKvHeads: 1);
        var cache = new MlxKvCache(numHeads: 1, maxSeqLen: 4, headDim: 4);
        Assert.Throws<NotSupportedException>(() => layer.ForwardDecode(new float[8], position: 0, cache));
    }

    [Fact]
    public void ForwardDecode_ValidShapes_WithNoNativeLibraryPresent_ReachesFirstNativeCall_FailsPredictably()
    {
        var layer = MakeLayer(hiddenDim: 8, numHeads: 2, headDim: 4, intermDim: 16);
        var cache = new MlxKvCache(numHeads: 2, maxSeqLen: 4, headDim: 4);
        Assert.Throws<DllNotFoundException>(() => layer.ForwardDecode(new float[8], position: 0, cache));
    }

    [Fact]
    public void Forward_HiddenLengthMismatch_Throws()
    {
        var layer = MakeLayer();
        Assert.Throws<ArgumentException>(() => layer.Forward(new float[5], seqLen: 2, positions: [0, 1]));
    }

    [Fact]
    public void Forward_PositionsLengthMismatch_Throws()
    {
        var layer = MakeLayer(hiddenDim: 4);
        Assert.Throws<ArgumentException>(() => layer.Forward(new float[2 * 4], seqLen: 2, positions: [0]));
    }

    [Fact]
    public void Forward_ValidShapes_WithNoNativeLibraryPresent_ReachesFirstNativeCall_FailsPredictably()
    {
        var layer = MakeLayer(hiddenDim: 4, headDim: 4, intermDim: 8);
        // DllNotFoundException, not any other exception, confirms every op's shape handoff into the
        // next one is consistent all the way to the first real mlx-c call (RmsNorm's).
        Assert.Throws<DllNotFoundException>(() => layer.Forward(new float[3 * 4], seqLen: 3, positions: [0, 1, 2]));
    }
}
