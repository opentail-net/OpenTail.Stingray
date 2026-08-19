using System;

namespace OpenTail.Stingray.Vision;

public sealed record LlavaPreprocessedImage(float[] Chw, int TargetWidth, int TargetHeight, int PatchesX, int PatchesY);

/// <summary>
/// Image preprocessor for LLaVA-1.5, LLaVA-NeXT, and LLaVA-OneVision multimodal vision models.
/// </summary>
public static class LlavaImagePreprocessor
{
    public static LlavaPreprocessedImage Preprocess(ReadOnlySpan<byte> rgb, int width, int height, int imageSize = 336, int patchSize = 14)
    {
        int targetW = imageSize;
        int targetH = imageSize;
        int patchesX = targetW / patchSize;
        int patchesY = targetH / patchSize;

        var chw = BaseVisionPreprocessor.BilinearResizeAndNormalize(
            rgb, width, height, targetW, targetH,
            BaseVisionPreprocessor.ClipMean,
            BaseVisionPreprocessor.ClipStd);

        return new LlavaPreprocessedImage(chw, targetW, targetH, patchesX, patchesY);
    }
}
