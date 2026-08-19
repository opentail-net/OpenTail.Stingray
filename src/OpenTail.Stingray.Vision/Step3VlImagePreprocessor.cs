using System;

namespace OpenTail.Stingray.Vision;

public sealed record Step3VlPreprocessedImage(float[] Chw, int TargetWidth, int TargetHeight, int PatchesX, int PatchesY);

/// <summary>
/// Image preprocessor for Step-3 VL. SigLIP-style normalization, patch-aligned resize.
/// </summary>
public static class Step3VlImagePreprocessor
{
    public static Step3VlPreprocessedImage Preprocess(
        ReadOnlySpan<byte> rgb, int width, int height,
        int imageSize = 1024, int patchSize = 14)
    {
        double scale = Math.Min(1.0, (double)imageSize / Math.Max(width, height));
        int targetW = Math.Max(patchSize, (int)Math.Round(width  * scale / patchSize) * patchSize);
        int targetH = Math.Max(patchSize, (int)Math.Round(height * scale / patchSize) * patchSize);

        int patchesX = targetW / patchSize;
        int patchesY = targetH / patchSize;

        float[] chw = BaseVisionPreprocessor.BilinearResizeAndNormalize(
            rgb, width, height, targetW, targetH,
            BaseVisionPreprocessor.ZeroCenterMean,
            BaseVisionPreprocessor.ZeroCenterStd);

        return new Step3VlPreprocessedImage(chw, targetW, targetH, patchesX, patchesY);
    }
}
