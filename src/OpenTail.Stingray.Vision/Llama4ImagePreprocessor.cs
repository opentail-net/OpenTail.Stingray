namespace OpenTail.Stingray.Vision;

/// <summary>
/// Deterministic fixed-square-tile preprocessing for the Llama 4 <c>llama4</c> ViT. Thin wrapper
/// around <see cref="Gemma4VImagePreprocessor.ResizeNormalize"/> (the resize/normalize primitive
/// is metadata-driven and model-agnostic; only the calling type differs) -- with this mmproj's
/// declared mean=[0.5,0.5,0.5]/std=[0.5,0.5,0.5], the output lands directly in [-1,1], so like
/// <see cref="Gemma3ImagePreprocessor"/> (and unlike <see cref="Gemma4VVisionEncoder"/>) no
/// additional range fix is needed.
///
/// <para>Produces exactly ONE tile (<c>ImageSize x ImageSize</c>, 336x336 for the real Scout
/// mmproj). llama.cpp's own multi-tile ("llava-uhd") preprocessing -- which real Llama 4 inference
/// uses to slice a source image into up to a 3x3 grid of tiles plus one overview tile -- is out of
/// scope here; see docs/06-llama4-vision-plan.md.</para>
/// </summary>
public static class Llama4ImagePreprocessor
{
    public static PreprocessedImage Preprocess(
        byte[] rgbHwc, int sourceWidth, int sourceHeight, Llama4VisionModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        return Gemma4VImagePreprocessor.ResizeNormalize(rgbHwc, sourceWidth, sourceHeight, model.ImageSize, model.ImageSize,
            model.ImageMean, model.ImageStd);
    }
}
