
namespace OpenTail.Stingray.Vision;

public readonly record struct QwenVlPreprocessResult(
    float[] Chw,
    int TargetWidth,
    int TargetHeight,
    int PatchesX,
    int PatchesY,
    int MergedTokens);

/// <summary>
/// Preprocessor for Qwen2-VL and Qwen2.5-VL vision input tensors.
/// Resizes images to grid multiples of 28 (patch_size 14 * merge 2) and normalizes to standard CLIP/SigLIP ranges.
/// </summary>
public static class QwenVlImagePreprocessor
{
    private static readonly float[] Mean = [0.48145466f, 0.4578275f, 0.40821073f];
    private static readonly float[] Std = [0.26862954f, 0.26130258f, 0.27577711f];

    public static QwenVlPreprocessResult Preprocess(ReadOnlySpan<byte> rgbBytes, int srcWidth, int srcHeight, int patchSize = 14, int mergeFactor = 2)
    {
        int unit = patchSize * mergeFactor; // 28

        // Snap dimensions to multiples of 28
        int targetW = Math.Max(unit, (int)Math.Round((double)srcWidth / unit) * unit);
        int targetH = Math.Max(unit, (int)Math.Round((double)srcHeight / unit) * unit);

        // Clamp to sensible maximums for single image inference
        if (targetW > 896) targetW = 896;
        if (targetH > 896) targetH = 896;

        int patchesX = targetW / patchSize;
        int patchesY = targetH / patchSize;
        int mergedTokens = (patchesX / mergeFactor) * (patchesY / mergeFactor);

        var chw = new float[3 * targetH * targetW];

        // Bilinear interpolation resize with normalization
        float scaleX = (float)srcWidth / targetW;
        float scaleY = (float)srcHeight / targetH;

        for (int y = 0; y < targetH; y++)
        {
            float srcY = (y + 0.5f) * scaleY - 0.5f;
            int y0 = Math.Clamp((int)MathF.Floor(srcY), 0, srcHeight - 1);
            int y1 = Math.Clamp(y0 + 1, 0, srcHeight - 1);
            float dy = Math.Clamp(srcY - y0, 0f, 1f);

            for (int x = 0; x < targetW; x++)
            {
                float srcX = (x + 0.5f) * scaleX - 0.5f;
                int x0 = Math.Clamp((int)MathF.Floor(srcX), 0, srcWidth - 1);
                int x1 = Math.Clamp(x0 + 1, 0, srcWidth - 1);
                float dx = Math.Clamp(srcX - x0, 0f, 1f);

                for (int c = 0; c < 3; c++)
                {
                    float p00 = rgbBytes[(y0 * srcWidth + x0) * 3 + c] / 255.0f;
                    float p10 = rgbBytes[(y0 * srcWidth + x1) * 3 + c] / 255.0f;
                    float p01 = rgbBytes[(y1 * srcWidth + x0) * 3 + c] / 255.0f;
                    float p11 = rgbBytes[(y1 * srcWidth + x1) * 3 + c] / 255.0f;

                    float val = (1f - dx) * (1f - dy) * p00 +
                                dx * (1f - dy) * p10 +
                                (1f - dx) * dy * p01 +
                                dx * dy * p11;

                    // Normalize: (val - mean) / std
                    float norm = (val - Mean[c]) / Std[c];
                    chw[c * (targetH * targetW) + (y * targetW + x)] = norm;
                }
            }
        }

        return new QwenVlPreprocessResult(chw, targetW, targetH, patchesX, patchesY, mergedTokens);
    }
}
