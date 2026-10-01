
namespace OpenTail.Stingray.Vision;

public sealed record LlavaPreprocessedImage(float[] Chw, int TargetWidth, int TargetHeight, int PatchesX, int PatchesY);

/// <summary>
/// Image preprocessor for LLaVA-1.5, LLaVA-NeXT, and LLaVA-OneVision multimodal vision models.
/// </summary>
public static class LlavaImagePreprocessor
{
    /// <summary>
    /// llama.cpp <c>mtmd_image_preprocessor_llava_uhd</c> for an MLP projector (LLaVA-NeXT / Granite Vision):
    /// the overview is the image stretched to <paramref name="imageSize"/> (u8 bilinear, <c>PAD_NONE</c>); when
    /// either side exceeds <paramref name="imageSize"/> and grid pinpoints exist, the image is also fitted into the
    /// best pinpoint resolution (llama.cpp's u8 bicubic, <c>PAD_CEIL</c>, black) and cut into
    /// <paramref name="imageSize"/> slices, row-major. Returns CHW views, overview first, normalised with the
    /// model's own mean/std. mtmd feeds them back to back with no marker tokens (<c>MTMD_SLICE_TMPL_NONE</c>).
    /// </summary>
    public static List<float[]> PreprocessViews(ReadOnlySpan<byte> rgb, int width, int height, int imageSize,
        IReadOnlyList<(int W, int H)> pinpoints, float[] mean, float[] std)
    {
        byte[] img = rgb.ToArray();
        var views = new List<float[]>
        {
            LlamaImgTool.ToChw(LlamaImgTool.ResizeBilinear(img, width, height, imageSize, imageSize), imageSize, imageSize, mean, std),
        };
        if ((width <= imageSize && height <= imageSize) || pinpoints.Count == 0) return views;

        var (rw, rh) = SelectBestResolution(width, height, pinpoints);
        byte[] refined = LlamaImgTool.ResizePadCeil(img, width, height, rw, rh, LlamaImgTool.ResizeBicubic, 0);
        for (int y = 0; y < rh; y += imageSize)
            for (int x = 0; x < rw; x += imageSize)
            {
                int sw = Math.Min(imageSize, rw - x), sh = Math.Min(imageSize, rh - y);
                var slice = new byte[sw * sh * 3];
                for (int r = 0; r < sh; r++) Array.Copy(refined, ((y + r) * rw + x) * 3, slice, r * sw * 3, sw * 3);
                // mtmd normalises the slice as-is (append(..., true)); slices are imageSize-square for pinpoint grids.
                views.Add(LlamaImgTool.ToChw(slice, sw, sh, mean, std));
            }
        return views;
    }

    /// <summary>
    /// Reorders <see cref="PreprocessViews"/> output (overview first, then row-major slices) into the order mtmd feeds an MLP-projector
    /// image: slices first, overview LAST (<c>ov_img_first = false</c> in tools/mtmd/mtmd.cpp; Granite-4 vision sets it true and does
    /// not use this). A single view (no slicing) is returned unchanged.
    /// </summary>
    public static List<float[]> OverviewLast(List<float[]> viewsOverviewFirst)
    {
        if (viewsOverviewFirst.Count <= 1) return viewsOverviewFirst;
        var ordered = new List<float[]>(viewsOverviewFirst.Count);
        for (int i = 1; i < viewsOverviewFirst.Count; i++) ordered.Add(viewsOverviewFirst[i]);
        ordered.Add(viewsOverviewFirst[0]);
        return ordered;
    }

    /// <summary>mtmd llava_uhd select_best_resolution: most effective pixels, then least waste.</summary>
    internal static (int W, int H) SelectBestResolution(int w, int h, IReadOnlyList<(int W, int H)> candidates)
    {
        (int W, int H) best = default;
        int minWasted = int.MaxValue, maxEffective = 0;
        foreach (var c in candidates)
        {
            float scale = Math.Min((float)c.W / w, (float)c.H / h);
            int tw = (int)(w * scale), th = (int)(h * scale);
            int effective = Math.Min(tw * th, w * h);
            int wasted = c.W * c.H - effective;
            if (effective > maxEffective || (effective == maxEffective && wasted < minWasted))
            {
                maxEffective = effective;
                minWasted = wasted;
                best = c;
            }
        }
        return best;
    }

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
