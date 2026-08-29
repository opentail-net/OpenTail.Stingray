
namespace OpenTail.Stingray.Vision;

public sealed record DotsOcrPreprocessedImage(float[] Chw, int TargetWidth, int TargetHeight, int PatchesX, int PatchesY);

/// <summary>
/// Image preprocessor for Dots-OCR and PaddleOCR-VL document models.
/// </summary>
public static class DotsOcrImagePreprocessor
{
    public static DotsOcrPreprocessedImage Preprocess(ReadOnlySpan<byte> rgb, int width, int height, int patchSize = 14, int minDim = 28, int maxDim = 768)
    {
        int targetW = BaseVisionPreprocessor.SnapToGrid(width, patchSize, minDim, maxDim);
        int targetH = BaseVisionPreprocessor.SnapToGrid(height, patchSize, minDim, maxDim);
        int patchesX = targetW / patchSize;
        int patchesY = targetH / patchSize;

        var chw = BaseVisionPreprocessor.BilinearResizeAndNormalize(
            rgb, width, height, targetW, targetH,
            BaseVisionPreprocessor.ClipMean,
            BaseVisionPreprocessor.ClipStd);

        return new DotsOcrPreprocessedImage(chw, targetW, targetH, patchesX, patchesY);
    }
}
