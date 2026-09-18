namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Real numeric check of <see cref="OpenTail.Stingray.Diffusion.Flux2.Flux2Vae.UnnormalizeAndUnshuffle"/>'s
/// pixel-unshuffle index mapping (docs/087, candidate (d) in the FLUX.2 bug hunt) -- no real
/// weights needed, this is pure index-mapping logic. Uses identity BatchNorm (mean=0, var=1-eps)
/// so the affine step is a no-op, isolating the unshuffle itself.
/// </summary>
public sealed class Flux2VaeUnshuffleUnitTests
{
    [Fact]
    public void UnnormalizeAndUnshuffle_KnownPattern_MapsToExpectedSpatialLayout()
    {
        // 4 input channels -> 1 output channel, 1x1 spatial -> 2x2 spatial.
        // Real mapping: cIndex = c*4 + pi*2 + pj; outRow = i*2+pi; outCol = j*2+pj.
        // With c=0, i=0, j=0: input channel k maps to output (row=k/2, col=k%2).
        float[] normalizedLatent = [10f, 20f, 30f, 40f]; // channels 0,1,2,3 at the single (i=0,j=0) position
        float[] runningMean = [0f, 0f, 0f, 0f];
        float[] runningVar = [1f - OpenTail.Stingray.Diffusion.Flux2.Flux2Vae.BatchNormEps, 1f - OpenTail.Stingray.Diffusion.Flux2.Flux2Vae.BatchNormEps, 1f - OpenTail.Stingray.Diffusion.Flux2.Flux2Vae.BatchNormEps, 1f - OpenTail.Stingray.Diffusion.Flux2.Flux2Vae.BatchNormEps];

        var (output, h, w) = OpenTail.Stingray.Diffusion.Flux2.Flux2Vae.UnnormalizeAndUnshuffle(
            normalizedLatent, inH: 1, inW: 1, runningMean, runningVar);

        Assert.Equal(2, h);
        Assert.Equal(2, w);
        Assert.Equal(4, output.Length); // 1 output channel * 2 * 2

        // Real mapping: channel k=pi*2+pj -> output[row=pi, col=pj] (c=0, i=0, j=0 all zero).
        // k=0 (pi=0,pj=0) -> (row0,col0)=10; k=1 (pi=0,pj=1) -> (row0,col1)=20;
        // k=2 (pi=1,pj=0) -> (row1,col0)=30; k=3 (pi=1,pj=1) -> (row1,col1)=40.
        Assert.Equal(10f, output[0 * w + 0], precision: 4); // row0,col0
        Assert.Equal(20f, output[0 * w + 1], precision: 4); // row0,col1
        Assert.Equal(30f, output[1 * w + 0], precision: 4); // row1,col0
        Assert.Equal(40f, output[1 * w + 1], precision: 4); // row1,col1
    }

    [Fact]
    public void UnnormalizeAndUnshuffle_AppliesPerChannelAffineCorrectly()
    {
        // 4 channels -> 1 output channel, real per-channel mean/var (not identity this time).
        float[] normalizedLatent = [0f, 0f, 0f, 0f];
        float[] runningMean = [1f, 2f, 3f, 4f];
        float[] runningVar = [0f, 0f, 0f, 0f]; // scale = sqrt(0 + eps) ~ small but nonzero

        var (output, _, _) = OpenTail.Stingray.Diffusion.Flux2.Flux2Vae.UnnormalizeAndUnshuffle(
            normalizedLatent, inH: 1, inW: 1, runningMean, runningVar);

        // z=0 input -> output = 0*scale + mean = mean, for each channel.
        Assert.Equal(1f, output[0], precision: 3);
        Assert.Equal(2f, output[1], precision: 3);
        Assert.Equal(3f, output[2], precision: 3);
        Assert.Equal(4f, output[3], precision: 3);
    }
}
