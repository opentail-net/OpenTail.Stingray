
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
        int imageSize,
        int patchSize,
        float[] imageMean,
        float[] imageStd)
    {
        int targetW = imageSize;
        int targetH = imageSize;
        int patchesX = targetW / patchSize;
        int patchesY = targetH / patchSize;

        // Real bug found 2026-09-11: this used to hardcode OpenAI CLIP's ImageNet-derived
        // mean/std regardless of checkpoint, but Granite 4 Vision's tower is SigLIP, not CLIP --
        // the real checkpoint's own clip.vision.image_mean/image_std GGUF metadata is
        // [0.5,0.5,0.5]/[0.5,0.5,0.5] (confirmed via `stingray list-metadata`), numerically quite
        // different from CLIP's constants. Now takes the real per-checkpoint values from
        // Granite4VisionModel instead (see its doc comment for the full story).
        var chw = BaseVisionPreprocessor.BilinearResizeAndNormalize(
            rgb, width, height, targetW, targetH,
            imageMean,
            imageStd);

        return new Granite4PreprocessedImage(chw, targetW, targetH, patchesX, patchesY);
    }
}
