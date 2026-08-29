
namespace OpenTail.Stingray.Vision;

public sealed record CogVlmPreprocessedImage(float[] Chw, int TargetWidth, int TargetHeight, int PatchesX, int PatchesY);

/// <summary>
/// Image preprocessor for CogVLM and CogAgent models.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/cogvlm.cpp
/// </summary>
public static class CogVlmImagePreprocessor
{
    public static CogVlmPreprocessedImage Preprocess(
        ReadOnlySpan<byte> rgb,
        int width,
        int height,
        int imageSize = 490,
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

        return new CogVlmPreprocessedImage(chw, targetW, targetH, patchesX, patchesY);
    }
}
