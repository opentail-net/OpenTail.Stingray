
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Reusable base image preprocessor for multimodal vision models.
/// Provides bilinear resampling, planar CHW transposition, parametric mean/std normalization, and aspect ratio snapping.
/// </summary>
public static class BaseVisionPreprocessor
{
    public static readonly float[] ImageNetMean = { 0.485f, 0.456f, 0.406f };
    public static readonly float[] ImageNetStd = { 0.229f, 0.224f, 0.225f };

    public static readonly float[] ClipMean = { 0.48145466f, 0.4578275f, 0.40821073f };
    public static readonly float[] ClipStd = { 0.26862954f, 0.26130258f, 0.27577711f };

    public static readonly float[] ZeroCenterMean = { 0.5f, 0.5f, 0.5f };
    public static readonly float[] ZeroCenterStd = { 0.5f, 0.5f, 0.5f };

    /// <summary>
    /// Bilinearly resizes packed interleaved RGB bytes to target dimensions and normalizes into planar float CHW.
    /// </summary>
    public static float[] BilinearResizeAndNormalize(
        ReadOnlySpan<byte> rgb,
        int srcWidth,
        int srcHeight,
        int targetWidth,
        int targetHeight,
        ReadOnlySpan<float> mean,
        ReadOnlySpan<float> std)
    {
        if (srcWidth <= 0 || srcHeight <= 0 || targetWidth <= 0 || targetHeight <= 0)
            throw new ArgumentException("Dimensions must be positive.");

        var chw = new float[3 * targetWidth * targetHeight];
        int planeSize = targetWidth * targetHeight;

        float xRatio = (float)srcWidth / targetWidth;
        float yRatio = (float)srcHeight / targetHeight;

        for (int dy = 0; dy < targetHeight; dy++)
        {
            float sy = (dy + 0.5f) * yRatio - 0.5f;
            int y0 = Math.Clamp((int)MathF.Floor(sy), 0, srcHeight - 1);
            int y1 = Math.Clamp(y0 + 1, 0, srcHeight - 1);
            float wy1 = sy - y0;
            float wy0 = 1.0f - wy1;

            for (int dx = 0; dx < targetWidth; dx++)
            {
                float sx = (dx + 0.5f) * xRatio - 0.5f;
                int x0 = Math.Clamp((int)MathF.Floor(sx), 0, srcWidth - 1);
                int x1 = Math.Clamp(x0 + 1, 0, srcWidth - 1);
                float wx1 = sx - x0;
                float wx0 = 1.0f - wx1;

                int idx00 = (y0 * srcWidth + x0) * 3;
                int idx01 = (y0 * srcWidth + x1) * 3;
                int idx10 = (y1 * srcWidth + x0) * 3;
                int idx11 = (y1 * srcWidth + x1) * 3;

                int outIdx = dy * targetWidth + dx;

                for (int c = 0; c < 3; c++)
                {
                    float v00 = rgb[idx00 + c] / 255.0f;
                    float v01 = rgb[idx01 + c] / 255.0f;
                    float v10 = rgb[idx10 + c] / 255.0f;
                    float v11 = rgb[idx11 + c] / 255.0f;

                    float val = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
                    float norm = (val - mean[c]) / std[c];
                    chw[c * planeSize + outIdx] = norm;
                }
            }
        }

        return chw;
    }

    /// <summary>
    /// Snaps dimension to the nearest multiple of patch size.
    /// </summary>
    public static int SnapToGrid(int dim, int patchSize, int minDim = 28, int maxDim = 1024)
    {
        int snapped = ((dim + patchSize / 2) / patchSize) * patchSize;
        return Math.Clamp(snapped, minDim, maxDim);
    }
}
