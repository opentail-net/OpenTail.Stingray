using OpenTail.Stingray.Vision;

namespace OpenTail.Stingray.Tests.Vision;

public class Gemma4VImagePreprocessorTests
{
    [Fact]
    public void ResizeNormalize_AlignCorners_PacksPlanarChwAndAppliesChannelAffine()
    {
        // 2x2 RGB: red, green / blue, white. A 3x3 resize has source corners at its corners
        // and the exact four-way mean at its centre under align-corners bilinear sampling.
        byte[] rgb =
        [
            255, 0, 0,    0, 255, 0,
            0, 0, 255,    255, 255, 255,
        ];

        var image = Gemma4VImagePreprocessor.ResizeNormalize(
            rgb, 2, 2, 3, 3,
            [0.5f, 0f, 0.25f], [0.5f, 2f, 0.25f]);

        Assert.Equal(3, image.Width);
        Assert.Equal(3, image.Height);
        Assert.Equal(27, image.Chw.Length);
        // Planar R: red top-left normalises to 1; green top-right normalises to -1.
        Assert.Equal(1f, image.Chw[0], 6);
        Assert.Equal(-1f, image.Chw[2], 6);
        // Planar G: green top-right is 0.5 after std=2; red top-left is zero.
        Assert.Equal(0.5f, image.Chw[9 + 2], 6);
        Assert.Equal(0f, image.Chw[9], 6);
        // At centre each channel sees two 0s and two 1s, hence 0.5 before affine.
        Assert.Equal(0f, image.Chw[4], 6);          // (.5 - .5) / .5
        Assert.Equal(0.25f, image.Chw[9 + 4], 6);   // .5 / 2
        Assert.Equal(1f, image.Chw[18 + 4], 6);     // (.5 - .25) / .25
    }

    [Fact]
    public void ResizeNormalize_RejectsInvalidShapeOrNormalisation()
    {
        Assert.Throws<ArgumentException>(() => Gemma4VImagePreprocessor.ResizeNormalize(
            [1, 2, 3], 1, 1, 1, 1, [0f, 0f, 0f], [1f, 1f, 0f]));
        Assert.Throws<ArgumentException>(() => Gemma4VImagePreprocessor.ResizeNormalize(
            [1, 2], 1, 1, 1, 1, [0f, 0f, 0f], [1f, 1f, 1f]));
    }
}
