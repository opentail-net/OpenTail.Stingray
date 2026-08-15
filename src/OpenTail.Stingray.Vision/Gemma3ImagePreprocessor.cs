namespace OpenTail.Stingray.Vision;

/// <summary>
/// Deterministic fixed-grid preprocessing for the Gemma 3 <c>gemma3</c> SigLIP ViT. Thin wrapper
/// around <see cref="Gemma4VImagePreprocessor.ResizeNormalize"/> (the resize/normalize primitive
/// is metadata-driven and model-agnostic; only the calling type differs) -- with this mmproj's
/// declared mean=[0.5,0.5,0.5]/std=[0.5,0.5,0.5], the output lands directly in [-1,1], so unlike
/// <see cref="Gemma4VVisionEncoder"/> the Gemma 3 encoder applies no additional range fix.
/// </summary>
public static class Gemma3ImagePreprocessor
{
    public static PreprocessedImage Preprocess(
        byte[] rgbHwc, int sourceWidth, int sourceHeight, Gemma3VisionModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        return Gemma4VImagePreprocessor.ResizeNormalize(rgbHwc, sourceWidth, sourceHeight, model.ImageSize, model.ImageSize,
            model.ImageMean, model.ImageStd);
    }
}
