
namespace OpenTail.Stingray.Vision;

public sealed record NemotronPreprocessedImage(float[] Chw, int TargetWidth, int TargetHeight, int PatchesX, int PatchesY);

/// <summary>
/// Image preprocessor for NVIDIA Nemotron-V2-VL / Nemotron-4-Nano.
/// </summary>
public static class NemotronImagePreprocessor
{
    public static NemotronPreprocessedImage Preprocess(ReadOnlySpan<byte> rgb, int width, int height, int imageSize = 512, int patchSize = 14)
    {
        int targetW = imageSize;
        int targetH = imageSize;
        int patchesX = targetW / patchSize;
        int patchesY = targetH / patchSize;

        var chw = BaseVisionPreprocessor.BilinearResizeAndNormalize(
            rgb, width, height, targetW, targetH,
            BaseVisionPreprocessor.ClipMean,
            BaseVisionPreprocessor.ClipStd);

        return new NemotronPreprocessedImage(chw, targetW, targetH, patchesX, patchesY);
    }
}
