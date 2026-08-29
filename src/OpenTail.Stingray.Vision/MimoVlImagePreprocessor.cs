
namespace OpenTail.Stingray.Vision;

public sealed record MimoVlPreprocessedImage(float[] Chw, int TargetWidth, int TargetHeight, int PatchesX, int PatchesY);

/// <summary>
/// Image preprocessor for MiMo-VL. Same patch-aligned resize and zero-center normalization as QwenVL.
/// </summary>
public static class MimoVlImagePreprocessor
{
    public static MimoVlPreprocessedImage Preprocess(
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

        return new MimoVlPreprocessedImage(chw, targetW, targetH, patchesX, patchesY);
    }
}
