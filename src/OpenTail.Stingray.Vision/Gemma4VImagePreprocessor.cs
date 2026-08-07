namespace OpenTail.Stingray.Vision;

/// <summary>
/// Deterministic fixed-grid preprocessing for the encoder-full Gemma 4 E-model ViT.
/// Produces planar CHW values after the mmproj-declared per-channel normalisation.
/// </summary>
public static class Gemma4VImagePreprocessor
{
    /// <summary>
    /// <b>Parity risk to settle first:</b> this uses <c>align_corners=True</c> bilinear
    /// (<c>ratio = (source - 1) / (target - 1)</c>). Reference vision preprocessing commonly uses
    /// <c>align_corners=False</c>, and CLIP-family pipelines often use bicubic. Those choices give
    /// systematically different pixels, hence different embeddings — a mismatch that will look like
    /// a projector or encoder bug while actually being resampling. Confirm the reference's resize
    /// mode and filter BEFORE debugging anything downstream of this function.
    ///
    /// Resizes interleaved 8-bit RGB with align-corners bilinear sampling, converts it to CHW,
    /// then applies <c>(x / 255 - mean[c]) / std[c]</c>. The E4B mmproj declares a 224×224
    /// input and mean=[0,0,0], std=[1,1,1], but the operation remains metadata-driven so a
    /// differently exported compatible projector is not silently preprocessed as E4B.
    /// </summary>
    public static PreprocessedImage Preprocess(
        byte[] rgbHwc, int sourceWidth, int sourceHeight, Gemma4VVisionModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        return ResizeNormalize(rgbHwc, sourceWidth, sourceHeight, model.ImageSize, model.ImageSize,
            model.ImageMean, model.ImageStd);
    }

    /// <summary>Pure resize/normalise primitive, exposed for deterministic fixture coverage.</summary>
    public static PreprocessedImage ResizeNormalize(
        ReadOnlySpan<byte> rgbHwc, int sourceWidth, int sourceHeight, int targetWidth, int targetHeight,
        ReadOnlySpan<float> mean, ReadOnlySpan<float> std)
    {
        if (sourceWidth <= 0 || sourceHeight <= 0 || targetWidth <= 0 || targetHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(sourceWidth), "image dimensions must be positive.");
        if (rgbHwc.Length < checked(sourceWidth * sourceHeight * 3))
            throw new ArgumentException("RGB input is shorter than source dimensions require.", nameof(rgbHwc));
        if (mean.Length != 3 || std.Length != 3 || std[0] == 0 || std[1] == 0 || std[2] == 0)
            throw new ArgumentException("image mean/std must contain three non-zero-standard-deviation channels.");

        var chw = new float[checked(3 * targetWidth * targetHeight)];
        var plane = targetWidth * targetHeight;
        var xRatio = targetWidth > 1 ? (float)(sourceWidth - 1) / (targetWidth - 1) : 0f;
        var yRatio = targetHeight > 1 ? (float)(sourceHeight - 1) / (targetHeight - 1) : 0f;

        for (var y = 0; y < targetHeight; y++)
        {
            var py = y * yRatio;
            var y0 = Math.Min((int)py, sourceHeight - 1);
            var y1 = Math.Min(y0 + 1, sourceHeight - 1);
            var yf = py - y0;
            for (var x = 0; x < targetWidth; x++)
            {
                var px = x * xRatio;
                var x0 = Math.Min((int)px, sourceWidth - 1);
                var x1 = Math.Min(x0 + 1, sourceWidth - 1);
                var xf = px - x0;
                var i00 = (y0 * sourceWidth + x0) * 3;
                var i10 = (y0 * sourceWidth + x1) * 3;
                var i01 = (y1 * sourceWidth + x0) * 3;
                var i11 = (y1 * sourceWidth + x1) * 3;
                var output = y * targetWidth + x;
                for (var c = 0; c < 3; c++)
                {
                    var top = Lerp(rgbHwc[i00 + c], rgbHwc[i10 + c], xf);
                    var bottom = Lerp(rgbHwc[i01 + c], rgbHwc[i11 + c], xf);
                    chw[c * plane + output] = (Lerp(top, bottom, yf) / 255f - mean[c]) / std[c];
                }
            }
        }

        return new PreprocessedImage(chw, targetWidth, targetHeight);
    }

    private static float Lerp(float a, float b, float t) => a + (b - a) * t;
}
