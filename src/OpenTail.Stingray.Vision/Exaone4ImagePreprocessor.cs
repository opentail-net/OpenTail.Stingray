using System;

namespace OpenTail.Stingray.Vision;

public sealed record Exaone4PreprocessedImage(float[] Chw, int TargetWidth, int TargetHeight, int PatchesX, int PatchesY);

/// <summary>
/// Image preprocessor for EXAONE 4.5 Vision. Patch-aligned resize, zero-center normalization.
/// </summary>
public static class Exaone4ImagePreprocessor
{
    public static Exaone4PreprocessedImage Preprocess(
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

        return new Exaone4PreprocessedImage(chw, targetW, targetH, patchesX, patchesY);
    }
}
