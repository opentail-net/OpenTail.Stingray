using System;

namespace OpenTail.Stingray.Vision;

public sealed record HunyuanVlPreprocessedImage(float[] Chw, int TargetWidth, int TargetHeight, int PatchesX, int PatchesY);

/// <summary>
/// Image preprocessor for Tencent HunyuanVL.
/// Snaps to patch-size multiples, applies CLIP mean/std normalization.
/// </summary>
public static class HunyuanVlImagePreprocessor
{
    public static HunyuanVlPreprocessedImage Preprocess(
        ReadOnlySpan<byte> rgb, int width, int height,
        int imageSize = 378, int patchSize = 14)
    {
        // Fit largest side to imageSize, snap both dims to patchSize multiples
        double scale = Math.Min(1.0, (double)imageSize / Math.Max(width, height));
        int targetW = Math.Max(patchSize, (int)Math.Round(width  * scale / patchSize) * patchSize);
        int targetH = Math.Max(patchSize, (int)Math.Round(height * scale / patchSize) * patchSize);

        int patchesX = targetW / patchSize;
        int patchesY = targetH / patchSize;

        float[] chw = BaseVisionPreprocessor.BilinearResizeAndNormalize(
            rgb, width, height, targetW, targetH,
            BaseVisionPreprocessor.ClipMean,
            BaseVisionPreprocessor.ClipStd);

        return new HunyuanVlPreprocessedImage(chw, targetW, targetH, patchesX, patchesY);
    }
}
