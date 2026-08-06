
namespace OpenTail.Stingray.Tests.Vision;

public sealed class Flux2DRoPETests
{
    [Fact]
    public void ImagePatchIds_ProducesRowMajorRowColPairs()
    {
        int[] ids = OpenTail.Stingray.Diffusion.Flux2DRoPE.ImagePatchIds(heightPatches: 2, widthPatches: 3);
        // patch 0 -> (0,0), patch 1 -> (0,1), patch 2 -> (0,2), patch 3 -> (1,0), ...
        Assert.Equal([0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 2], ids);
    }

    [Fact]
    public void BuildFreqs_Position0_CosIsOneSinIsZero()
    {
        int[] positions = [0, 0];
        var (cos, sin) = OpenTail.Stingray.Diffusion.Flux2DRoPE.BuildFreqs(positions, nPatches: 1, headDim: 8);
        Assert.All(cos, c => Assert.Equal(1f, c, precision: 5));
        Assert.All(sin, s => Assert.Equal(0f, s, precision: 5));
    }

    [Fact]
    public void BuildFreqs_ReturnsTablesShapedNPatchesByHeadDim()
    {
        int[] positions = [0, 0, 1, 2, 3, 4];
        var (cos, sin) = OpenTail.Stingray.Diffusion.Flux2DRoPE.BuildFreqs(positions, nPatches: 3, headDim: 8);
        Assert.Equal(3 * 8, cos.Length);
        Assert.Equal(3 * 8, sin.Length);
    }

    [Fact]
    public void ApplyInPlace_IsRotationThatPreservesVectorNorm()
    {
        const int nSeq = 2, nHeads = 1, headDim = 4;
        float[] x = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        double normBefore = System.Math.Sqrt(x.Sum(v => (double)v * v));

        int[] positions = [1, 2, 3, 4];
        var (cos, sin) = OpenTail.Stingray.Diffusion.Flux2DRoPE.BuildFreqs(positions, nPatches: nSeq, headDim: headDim);
        OpenTail.Stingray.Diffusion.Flux2DRoPE.ApplyInPlace(x, cos, sin, nSeq, nHeads, headDim, nImgPatches: nSeq);

        double normAfter = System.Math.Sqrt(x.Sum(v => (double)v * v));
        Assert.Equal(normBefore, normAfter, precision: 3);
    }

    [Fact]
    public void ApplyInPlace_LeavesPositionsBeyondNImgPatchesUnchanged()
    {
        const int nSeq = 2, nHeads = 1, headDim = 4;
        float[] x = [1f, 2f, 3f, 4f, 9f, 9f, 9f, 9f];
        float[] original = (float[])x.Clone();

        int[] positions = [1, 2, 3, 4];
        var (cos, sin) = OpenTail.Stingray.Diffusion.Flux2DRoPE.BuildFreqs(positions, nPatches: nSeq, headDim: headDim);
        // Only the first patch (index 0) counts as an image patch; the second (text) is untouched.
        OpenTail.Stingray.Diffusion.Flux2DRoPE.ApplyInPlace(x, cos, sin, nSeq, nHeads, headDim, nImgPatches: 1);

        Assert.Equal(original[4..], x[4..]);
    }
}

public sealed class ZImageRoPETests
{
    // Axis dims must each be even (halfD = dim/2 > 0 per axis) to avoid the degenerate
    // case where an axis contributes zero frequency slots; real configs (e.g. [32,48,48])
    // always satisfy this, so tests mirror that shape at a smaller scale: [4,2,2] sums to
    // HeadDim=8 with per-axis halfD = 2,1,1 (none zero).

    [Fact]
    public void BuildFreqs_Position0OnAllAxes_CosIsOneSinIsZero()
    {
        var p = new OpenTail.Stingray.Diffusion.ZImageParams
        {
            Dim = 16, NHeads = 2,       // HeadDim = 8
            AxesDims = [4, 2, 2],       // sum = 8 = HeadDim
            AxesLens = [4, 4, 4],
        };
        var rope = new OpenTail.Stingray.Diffusion.ZImageRoPE(p);
        int[] posIds = [0, 0, 0]; // one token, (t,h,w) = (0,0,0)
        float[] freqs = rope.BuildFreqs(posIds, nTokens: 1);

        // freqs is interleaved (cos,sin) pairs; at position 0 every angle is 0.
        for (int i = 0; i < freqs.Length; i += 2)
        {
            Assert.Equal(1f, freqs[i], precision: 5);
            Assert.Equal(0f, freqs[i + 1], precision: 5);
        }
    }

    [Fact]
    public void BuildFreqs_ReturnsLengthMatchingSumOfHalfAxesDims()
    {
        var p = new OpenTail.Stingray.Diffusion.ZImageParams
        {
            Dim = 16, NHeads = 2,
            AxesDims = [4, 2, 2],
            AxesLens = [4, 4, 4],
        };
        var rope = new OpenTail.Stingray.Diffusion.ZImageRoPE(p);
        int[] posIds = [0, 0, 0, 1, 0, 0];
        float[] freqs = rope.BuildFreqs(posIds, nTokens: 2);

        int totalHalfDim = (4 + 2 + 2) / 2; // = 4
        Assert.Equal(2 * totalHalfDim * 2, freqs.Length);
    }

    [Fact]
    public void Apply_IsRotationThatPreservesVectorNormPerHead()
    {
        var p = new OpenTail.Stingray.Diffusion.ZImageParams
        {
            Dim = 16, NHeads = 2,       // HeadDim = 8
            AxesDims = [4, 2, 2],
            AxesLens = [8, 8, 8],
        };
        var rope = new OpenTail.Stingray.Diffusion.ZImageRoPE(p);
        const int nTokens = 2, nHeads = 2;
        // [nTokens * nHeads * headDim] = 2 * 2 * 8 = 32 elements.
        float[] qk =
        [
            1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f,       // token0 head0
            1f, 1f, 1f, 1f, 2f, 2f, 2f, 2f,       // token0 head1
            3f, 1f, 4f, 1f, 5f, 9f, 2f, 6f,       // token1 head0
            0.5f, 1.5f, 2.5f, 3.5f, 4f, 3f, 2f, 1f, // token1 head1
        ];
        double normBefore = System.Math.Sqrt(qk.Sum(v => (double)v * v));

        int[] posIds = OpenTail.Stingray.Diffusion.ZImageRoPE.TextPosIds(nTokens);
        float[] freqs = rope.BuildFreqs(posIds, nTokens);
        rope.Apply(qk, nTokens, nHeads, freqs);

        double normAfter = System.Math.Sqrt(qk.Sum(v => (double)v * v));
        Assert.Equal(normBefore, normAfter, precision: 2);
    }

    [Fact]
    public void TextPosIds_StartsAtOneOnTAxisWithZeroHW()
    {
        int[] ids = OpenTail.Stingray.Diffusion.ZImageRoPE.TextPosIds(nTxt: 3);
        Assert.Equal([1, 0, 0, 2, 0, 0, 3, 0, 0], ids);
    }

    [Fact]
    public void ImagePosIds_UsesFixedTOffsetWithRowColSweep()
    {
        int[] ids = OpenTail.Stingray.Diffusion.ZImageRoPE.ImagePosIds(tOffset: 5, nRows: 2, nCols: 2);
        // (t,h,w) for each of the 4 patches, row-major.
        Assert.Equal([5, 0, 0, 5, 0, 1, 5, 1, 0, 5, 1, 1], ids);
    }
}
