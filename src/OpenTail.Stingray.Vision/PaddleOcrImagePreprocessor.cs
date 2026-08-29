
namespace OpenTail.Stingray.Vision;

public sealed record PaddleOcrPreprocessedImage(float[] Chw, int TargetWidth, int TargetHeight, int PatchesX, int PatchesY);

/// <summary>
/// Image preprocessor for PaddleOCR and PP-OCR vision models.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/paddleocr.cpp
/// </summary>
public static class PaddleOcrImagePreprocessor
{
    public static PaddleOcrPreprocessedImage Preprocess(
        ReadOnlySpan<byte> rgb,
        int width,
        int height,
        int patchSize = 14,
        int minDim = 28,
        int maxDim = 768)
    {
        int targetW = BaseVisionPreprocessor.SnapToGrid(width, patchSize, minDim, maxDim);
        int targetH = BaseVisionPreprocessor.SnapToGrid(height, patchSize, minDim, maxDim);
        int patchesX = targetW / patchSize;
        int patchesY = targetH / patchSize;

        var chw = BaseVisionPreprocessor.BilinearResizeAndNormalize(
            rgb, width, height, targetW, targetH,
            BaseVisionPreprocessor.ClipMean,
            BaseVisionPreprocessor.ClipStd);

        return new PaddleOcrPreprocessedImage(chw, targetW, targetH, patchesX, patchesY);
    }
}
