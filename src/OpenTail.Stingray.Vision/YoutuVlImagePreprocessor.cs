
namespace OpenTail.Stingray.Vision;

public sealed record YoutuVlPreprocessedImage(float[] Chw, int TargetWidth, int TargetHeight, int PatchesX, int PatchesY);

/// <summary>
/// Image preprocessor for YoutuVL.
/// Patch-aligned resize, zero-center ([0.5,0.5,0.5]) normalization matching SigLIP2.
/// </summary>
public static class YoutuVlImagePreprocessor
{
    /// <summary>
    /// Resize as llama.cpp's <c>mtmd_image_preprocessor_youtuvl</c>: each side is rounded UP to a
    /// multiple of <c>patch*merge</c> (32 px), and the image is only shrunk (scale steps of 0.02) when
    /// the patch count would exceed <c>maxTokens * merge^2</c> (clip.cpp sets 62500 tokens for YOUTUVL).
    /// Found 2026-09-27: the earlier port capped the long side at <c>clip.vision.image_size</c> (560) and
    /// rounded to nearest, giving 234 instead of 320 image tokens for a 640x488 image.
    /// </summary>
    public static YoutuVlPreprocessedImage Preprocess(
        ReadOnlySpan<byte> rgb, int width, int height,
        int patchSize = 16, int mergeFactor = 2, int maxTokens = 62500)
    {
        int align = patchSize * mergeFactor;
        long maxPatches = (long)maxTokens * mergeFactor * mergeFactor;
        static int Scaled(float scale, int size, int align) =>
            Math.Max(align, (int)MathF.Ceiling(size * scale / align) * align);

        float scale = 1.0f;
        int targetW = width, targetH = height;
        while (scale > 0.0f)
        {
            targetH = Scaled(scale, height, align);
            targetW = Scaled(scale, width, align);
            if ((long)(targetH / patchSize) * (targetW / patchSize) > maxPatches) scale -= 0.02f;
            else break;
        }
        int patchesX = targetW / patchSize;
        int patchesY = targetH / patchSize;
        float[] chw = BaseVisionPreprocessor.BilinearResizeAndNormalize(
            rgb, width, height, targetW, targetH,
            BaseVisionPreprocessor.ZeroCenterMean,
            BaseVisionPreprocessor.ZeroCenterStd);

        return new YoutuVlPreprocessedImage(chw, targetW, targetH, patchesX, patchesY);
    }
}
