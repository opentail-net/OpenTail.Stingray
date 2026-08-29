
namespace OpenTail.Stingray.Vision;

public sealed record YoutuVlPreprocessedImage(float[] Chw, int TargetWidth, int TargetHeight, int PatchesX, int PatchesY);

/// <summary>
/// Image preprocessor for YoutuVL.
/// Patch-aligned resize, zero-center ([0.5,0.5,0.5]) normalization matching SigLIP2.
/// </summary>
public static class YoutuVlImagePreprocessor
{
    public static YoutuVlPreprocessedImage Preprocess(
        ReadOnlySpan<byte> rgb, int width, int height,
        int patchSize = 14, int mergeFactor = 2, int maxDim = 980)
    {
        int align  = patchSize * mergeFactor;
        double scale = Math.Min(1.0, (double)maxDim / Math.Max(width, height));
        int targetW = Math.Max(align, (int)Math.Round(width  * scale / align) * align);
        int targetH = Math.Max(align, (int)Math.Round(height * scale / align) * align);

        int patchesX = targetW / patchSize;
        int patchesY = targetH / patchSize;

        float[] chw = BaseVisionPreprocessor.BilinearResizeAndNormalize(
            rgb, width, height, targetW, targetH,
            BaseVisionPreprocessor.ZeroCenterMean,
            BaseVisionPreprocessor.ZeroCenterStd);

        return new YoutuVlPreprocessedImage(chw, targetW, targetH, patchesX, patchesY);
    }
}
