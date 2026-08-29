
namespace OpenTail.Stingray.Vision;

public sealed record MobileNetV5PreprocessedImage(float[] Chw, int TargetWidth, int TargetHeight, int PatchesX, int PatchesY);

/// <summary>
/// Image preprocessor for MobileNetV5 lightweight vision backbone models.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/mobilenetv5.cpp
/// </summary>
public static class MobileNetV5ImagePreprocessor
{
    public static MobileNetV5PreprocessedImage Preprocess(
        ReadOnlySpan<byte> rgb,
        int width,
        int height,
        int imageSize = 224,
        int patchSize = 16)
    {
        int targetW = imageSize;
        int targetH = imageSize;
        int patchesX = targetW / patchSize;
        int patchesY = targetH / patchSize;

        var chw = BaseVisionPreprocessor.BilinearResizeAndNormalize(
            rgb, width, height, targetW, targetH,
            BaseVisionPreprocessor.ClipMean,
            BaseVisionPreprocessor.ClipStd);

        return new MobileNetV5PreprocessedImage(chw, targetW, targetH, patchesX, patchesY);
    }
}
