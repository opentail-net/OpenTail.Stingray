
namespace OpenTail.Stingray.Vision;

public sealed record Granite4PreprocessedImage(float[] Chw, int TargetWidth, int TargetHeight, int PatchesX, int PatchesY);

/// <summary>
/// Image preprocessor for IBM Granite 4 Vision multimodal models.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/granite4-vision.cpp
/// </summary>
public static class Granite4ImagePreprocessor
{
    public static Granite4PreprocessedImage Preprocess(
        ReadOnlySpan<byte> rgb,
        int width,
        int height,
        int imageSize = 384,
        int patchSize = 14)
    {
        int targetW = imageSize;
        int targetH = imageSize;
        int patchesX = targetW / patchSize;
        int patchesY = targetH / patchSize;

        var chw = BaseVisionPreprocessor.BilinearResizeAndNormalize(
            rgb, width, height, targetW, targetH,
            BaseVisionPreprocessor.ClipMean,
            BaseVisionPreprocessor.ClipStd);

        return new Granite4PreprocessedImage(chw, targetW, targetH, patchesX, patchesY);
    }
}
